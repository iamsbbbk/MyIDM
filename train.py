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

parser.add_argument("--use_weartool", action="store_true", default=True)
parser.add_argument("--weartool_npy", type=str, default="./data/WearTool/rgb_images.npy")
parser.add_argument("--sd_path", type=str, default="./sd15")
parser.add_argument("--fix_operator", action="store_true")

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
    print("use_weartool =", args.use_weartool)


# =========================
# CS operator
# =========================
iter_num = 1000
N = B * B
q = int(np.ceil(ratio * N))

U, S, V = torch.linalg.svd(torch.randn(N, N, device=device))
Phi = (U @ V)[:, :q].float()


# =========================
# ✅ ADD: block-wise CS operator (缺失函数补回)
# =========================
def make_A_AT_for_patch(psz, B, Phi, device, perm=None, perm_inv=None):
    nb = (psz // B) * (psz // B)
    if perm is None:
        perm = torch.arange(nb, device=device)
    if perm_inv is None:
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(nb, device=device)

    def A(z):
        z = z.float()
        blocks = F.unfold(z, kernel_size=B, stride=B)   # [bsz, N, nb]
        blocks = blocks.transpose(1, 2)                 # [bsz, nb, N]
        blocks = blocks[:, perm, :]
        return blocks @ Phi                             # [bsz, nb, q]

    def AT(y):
        y = y.float()
        blocks = y @ Phi.t()                            # [bsz, nb, N]
        blocks = blocks[:, perm_inv, :]
        blocks = blocks.transpose(1, 2)                 # [bsz, N, nb]
        return F.fold(blocks, output_size=(psz, psz),
                      kernel_size=B, stride=B)

    return A, AT


# =========================
# Load SD v1.5
# =========================
pipe = StableDiffusionPipeline.from_pretrained(
    args.sd_path,
    torch_dtype=torch.float32,
    local_files_only=True
).to(device)

pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

net = Net(T, pipe.unet).to(device)
model = net


# =========================
# Dataset
# =========================
class WearToolDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path)
        self.N = len(self.data)

    def __len__(self):
        return iter_num * bsz

    def __getitem__(self, index):
        img = self.data[index % self.N]
        x = img.mean(axis=2)
        h, w = x.shape
        sh = random.randint(0, h - psz)
        sw = random.randint(0, w - psz)
        patch = x[sh:sh + psz, sw:sw + psz]
        return torch.from_numpy(patch).float()


dataset = WearToolDataset(args.weartool_npy)

dataloader = DataLoader(
    dataset,
    batch_size=bsz,
    shuffle=True,
    num_workers=0 if os.name == "nt" else 8,
    pin_memory=torch.cuda.is_available(),
    drop_last=True
)


# =========================
# Optimizer
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[10, 20, 30, 40], gamma=0.5
)

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# =========================
# Training
# =========================
print("✅ start training...")
for epoch_i in range(1, epoch + 1):
    loss_avg = 0.0
    iterator = tqdm(dataloader)

    for x in iterator:
        x = x.unsqueeze(1).to(device)
        x = H(x, random.randint(0, 7))

        nb = (psz // B) ** 2
        perm = torch.randperm(nb, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(nb, device=device)

        A, AT = make_A_AT_for_patch(psz, B, Phi, device, perm, perm_inv)
        y = A(x)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x_out = model(y, A, AT)
            loss = (x_out - x).abs().mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_avg += loss.item()

    scheduler.step()
    print(f"[{epoch_i}/{epoch}] loss={loss_avg/len(dataloader):.6f}")
