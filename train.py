import os, glob, cv2, random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
from diffusers import StableDiffusionPipeline


# =========================
# DDP utilities
# =========================
def ddp_is_enabled() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and int(os.environ["WORLD_SIZE"]) > 1


use_ddp = ddp_is_enabled()
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

if use_ddp:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
else:
    print("Running in single-GPU (non-DDP) mode.")


# =========================
# Arguments
# =========================
parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--step_number", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--cs_ratio", type=float, default=0.1)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--testset_name", type=str, default="Set11")

# ✅ ADD: SD 本地路径（与你当前模型更匹配）
parser.add_argument("--sd_path", type=str, required=True, help="Local Stable Diffusion v1.5 folder containing model_index.json")

# ✅ ADD: 训练集目录（保留你原 pristine_images 默认）
parser.add_argument("--train_subdir", type=str, default="Set11")

# ✅ ADD: 训练时固定测量算子（组合1建议）
parser.add_argument("--fix_operator", action="store_true", help="Fix perm/Phi in training for stability")

args = parser.parse_args()


# =========================
# Reproducibility
# =========================
seed = 2025 + rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# =========================
# Hyper-parameters
# =========================
epoch = args.epoch
learning_rate = args.learning_rate
T = args.step_number
B = args.block_size
bsz = args.batch_size
psz = args.patch_size
ratio = args.cs_ratio

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

if rank == 0:
    print("cs ratio =", ratio)
    print("batch size per gpu =", bsz)
    print("patch size =", psz)
    print("block size =", B)
    print("use_ddp =", use_ddp, "world_size =", world_size, "local_rank =", local_rank)
    print("sd_path =", os.path.abspath(args.sd_path))


# =========================
# CS operator
# =========================
iter_num = 1000
N = B * B
q = int(np.ceil(ratio * N))

# ✅ FIX: svd 输出 double，Phi 强制 float32
U, S, V = torch.linalg.svd(torch.randn(N, N, device=device))
Phi = (U @ V)[:, :q].float()  # [N, q]


# =========================
# Load training images
# =========================
print("reading files...")
start_time = time()
train_dir = os.path.join(args.data_dir, args.train_subdir)
training_image_paths = glob.glob(os.path.join(train_dir, "*"))
if rank == 0:
    print("train_dir =", os.path.abspath(train_dir))
    print("training_image_num", len(training_image_paths), "read time", time() - start_time)
if len(training_image_paths) == 0:
    raise RuntimeError(f"No training images found in {train_dir}")


# =========================
# Load SD v1.5 locally
# =========================
def load_sd15(sd_path: str, device: torch.device):
    sd_dir = Path(sd_path).expanduser().resolve()
    if not sd_dir.exists():
        raise FileNotFoundError(f"[SD] Path not found: {sd_dir}")
    if not (sd_dir / "model_index.json").exists():
        raise FileNotFoundError(f"[SD] model_index.json missing in {sd_dir}")

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_dir,
        torch_dtype=torch.float32,
        local_files_only=True
    ).to(device)

    # 冻结不必要模块（更省显存/更稳定）
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    return pipe


pipe = load_sd15(args.sd_path, device)

net = Net(T, pipe.unet).to(device)
if use_ddp:
    model = DDP(net, device_ids=[local_rank] if torch.cuda.is_available() else None)
    model._set_static_graph()
else:
    model = net

if rank == 0:
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt / 1e6, "M")


# =========================
# Dataset
# =========================
class MyDataset(Dataset):
    def __getitem__(self, index):
        while True:
            path = random.choice(training_image_paths)
            img = cv2.imread(path, 1)
            if img is None:
                continue
            x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
            h, w = x.shape
            max_h, max_w = h - psz, w - psz
            if max_h < 0 or max_w < 0:
                continue
            start_h = random.randint(0, max_h)
            start_w = random.randint(0, max_w)
            patch = x[start_h:start_h + psz, start_w:start_w + psz]
            return torch.from_numpy(patch)

    def __len__(self):
        return iter_num * bsz


sampler = (
    torch.utils.data.distributed.DistributedSampler(
        MyDataset(), num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    ) if use_ddp else None
)

dataloader = DataLoader(
    MyDataset(),
    batch_size=bsz,
    sampler=sampler,
    shuffle=(sampler is None),
    num_workers=0 if os.name == "nt" else 8,
    pin_memory=torch.cuda.is_available(),
    drop_last=True
)


# =========================
# Optimizer
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# =========================
# Logging / saving
# =========================
model_dir = "./%s/R_%.2f_T_%d_B_%d" % (args.model_dir, ratio, T, B)
log_path = "./%s/R_%.2f_T_%d_B_%d.txt" % (args.log_dir, ratio, T, B)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

best_psnr = -1.0


# =========================
# Block-wise A / AT for training (✅ FIX shape mismatch)
# =========================
def make_A_AT_for_patch(psz, B, Phi, device, perm=None, perm_inv=None):
    """
    Patch: [bsz,1,psz,psz]
    Blocks: unfold -> [bsz, N, nb] -> transpose -> [bsz, nb, N]
    y: [bsz, nb, q]
    """
    nb = (psz // B) * (psz // B)
    if perm is None:
        perm = torch.arange(nb, device=device)
    if perm_inv is None:
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(perm.shape[0], device=device)

    def A(z):
        z = z.float()
        blocks = F.unfold(z, kernel_size=B, stride=B)           # [bsz, N, nb]
        blocks = blocks.transpose(1, 2)                         # [bsz, nb, N]
        blocks = blocks[:, perm, :]                             # permute blocks (optional)
        y = blocks @ Phi                                        # [bsz, nb, q]
        return y

    def AT(y):
        y = y.float()
        blocks = y @ Phi.t()                                    # [bsz, nb, N]
        blocks = blocks[:, perm_inv, :]                         # inverse perm
        blocks = blocks.transpose(1, 2)                         # [bsz, N, nb]
        x = F.fold(blocks, output_size=(psz, psz), kernel_size=B, stride=B)
        return x

    return A, AT


# =========================
# Test set
# =========================
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name, "*"))
if rank == 0:
    print("test_image_num =", len(test_image_paths))


def test():
    if use_ddp and rank != 0:
        return None, None
    if len(test_image_paths) == 0:
        return None, None

    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for path in test_image_paths:
            test_image = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2YCrCb)
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image[:, :, 0], block_size=B)

            x = torch.from_numpy(img_pad.reshape(1, 1, new_h, new_w).astype(np.float32) / 255.0).to(device)

            # 对 test：使用 block-wise A/AT；psz 用 new_h/new_w（要求可整除B）
            # 这里假设 my_zero_pad 已保证 new_h,new_w 可被 B 整除
            nb = (new_h // B) * (new_w // B)
            perm = torch.randperm(nb, device=device)
            perm_inv = torch.empty_like(perm)
            perm_inv[perm] = torch.arange(perm.shape[0], device=device)

            def A_full(z):
                z = z.float()
                blocks = F.unfold(z, kernel_size=B, stride=B)        # [1, N, nb]
                blocks = blocks.transpose(1, 2)                      # [1, nb, N]
                blocks = blocks[:, perm, :]
                y = blocks @ Phi                                     # [1, nb, q]
                return y

            def AT_full(y):
                y = y.float()
                blocks = y @ Phi.t()                                 # [1, nb, N]
                blocks = blocks[:, perm_inv, :]
                blocks = blocks.transpose(1, 2)                      # [1, N, nb]
                xrec = F.fold(blocks, output_size=(new_h, new_w), kernel_size=B, stride=B)
                return xrec

            y = A_full(x)
            x_out = model(y, A_full, AT_full, use_amp_=False)[..., :old_h, :old_w]
            x_out = (x_out.clamp(min=0.0, max=1.0) * 255.0).cpu().numpy().squeeze()

            PSNR_list.append(psnr(x_out, img))
            SSIM_list.append(ssim(x_out, img, data_range=255))

    return float(np.mean(PSNR_list)), float(np.mean(SSIM_list))


# =========================
# Training
# =========================
print("start training...")
for epoch_i in range(1, epoch + 1):
    start_time = time()
    loss_avg = 0.0
    psnr_avg = 0.0
    ssim_avg = 0.0

    if use_ddp:
        sampler.set_epoch(epoch_i)
        dist.barrier()

    # 组合1建议：训练时固定 operator（可选）
    if args.fix_operator:
        nb_train = (psz // B) * (psz // B)
        perm_fixed = torch.randperm(nb_train, device=device)
        perm_inv_fixed = torch.empty_like(perm_fixed)
        perm_inv_fixed[perm_fixed] = torch.arange(perm_fixed.shape[0], device=device)
    else:
        perm_fixed, perm_inv_fixed = None, None

    iterator = tqdm(dataloader) if rank == 0 else dataloader

    for x in iterator:
        x = x.unsqueeze(1).to(device)
        x = H(x, random.randint(0, 7))

        # A/AT (block-wise)
        if args.fix_operator:
            A, AT = make_A_AT_for_patch(psz, B, Phi, device, perm_fixed, perm_inv_fixed)
        else:
            nb_train = (psz // B) * (psz // B)
            perm = torch.randperm(nb_train, device=device)
            perm_inv = torch.empty_like(perm)
            perm_inv[perm] = torch.arange(perm.shape[0], device=device)
            A, AT = make_A_AT_for_patch(psz, B, Phi, device, perm, perm_inv)

        y = A(x)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x_out = model(y, A, AT)
            loss = (x_out - x).abs().mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_avg += float(loss.item())

        # 实时 PSNR/SSIM（训练 patch）
        if rank == 0:
            x_np = (x_out.detach().clamp(0, 1) * 255.0).cpu().numpy().squeeze()
            gt_np = (x.detach().clamp(0, 1) * 255.0).cpu().numpy().squeeze()
            cur_psnr = psnr(x_np, gt_np)
            cur_ssim = ssim(x_np, gt_np, data_range=255)
            psnr_avg += float(cur_psnr)
            ssim_avg += float(cur_ssim)
            iterator.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{cur_psnr:.2f}", ssim=f"{cur_ssim:.4f}")

    scheduler.step()

    # 平均
    loss_avg /= len(dataloader)
    if rank == 0:
        psnr_avg /= len(dataloader)
        ssim_avg /= len(dataloader)

        log_data = "[%d/%d] loss=%.6f, PSNR=%.2f, SSIM=%.4f, time=%.2fs, lr=%.2e" % (
            epoch_i, epoch, loss_avg, psnr_avg, ssim_avg, time() - start_time, scheduler.get_last_lr()[0]
        )
        print(log_data)
        with open(log_path, "a") as log_file:
            log_file.write(log_data + "\n")

        # 保存（保留你原风格 + 增加 pth）
        if epoch_i % args.save_interval == 0:
            state = model.module.state_dict() if use_ddp else model.state_dict()
            torch.save(state, "./%s/net_params_%d.pkl" % (model_dir, epoch_i))

        # always save last / best
        state = model.module.state_dict() if use_ddp else model.state_dict()
        torch.save(state, os.path.join(model_dir, "last.pth"))

        cur_psnr, cur_ssim = test()
        if cur_psnr is not None:
            log_data = "CS Ratio=%.2f, TEST PSNR=%.2f, TEST SSIM=%.4f" % (ratio, cur_psnr, cur_ssim)
            print(log_data)
            with open(log_path, "a") as log_file:
                log_file.write(log_data + "\n")

            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                torch.save(state, os.path.join(model_dir, "best.pth"))

if use_ddp:
    dist.destroy_process_group()
