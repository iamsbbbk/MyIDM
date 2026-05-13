import os
import random
import argparse
import logging
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from tqdm import tqdm

from diffusers import UNet2DConditionModel
from model import Net
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from skimage.metrics import structural_similarity as ssim_calc


# =========================================================
# 0. Basic Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_class_save_dir(save_dir: str, target_class: int) -> str:
    save_dir = os.path.normpath(save_dir)
    if os.path.basename(save_dir) != f"class{target_class}":
        save_dir = os.path.join(save_dir, f"class{target_class}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def setup_logging(save_dir, exp_name):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(
        save_dir,
        f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def call_model(model, y, A_func, AT_func, use_amp=False):
    """
    兼容不同版本的 Net.forward:
        model(y, A, AT)
        model(y, A, AT, use_amp_=False)
    """
    try:
        out = model(y, A_func, AT_func, use_amp_=use_amp)
    except TypeError:
        out = model(y, A_func, AT_func)

    if isinstance(out, (tuple, list)):
        out = out[0]

    return out


# =========================================================
# 1. Loss Functions
# =========================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class FocalFrequencyLoss(nn.Module):
    """
    频域损失：
    后期增强高频纹理和磨损特征，不建议从第 1 个 epoch 开太大。
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(self, pred, target):
        pred_freq = torch.fft.fftn(pred.float(), dim=(-2, -1))
        target_freq = torch.fft.fftn(target.float(), dim=(-2, -1))

        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], dim=-1)
        target_freq = torch.stack([target_freq.real, target_freq.imag], dim=-1)

        freq_distance = self.l1_loss(pred_freq, target_freq).mean(dim=-1)

        weight = freq_distance ** self.alpha
        weight = weight / (weight.max().detach() + 1e-8)

        return (freq_distance * weight).mean()


def orthogonality_loss(Phi: torch.Tensor) -> torch.Tensor:
    """
    Phi: [N, q]
    希望列正交:
        Phi^T Phi ≈ I
    """
    gram = Phi.t() @ Phi
    I = torch.eye(gram.size(0), device=Phi.device, dtype=Phi.dtype)
    return F.mse_loss(gram, I)


def range_regularization(x: torch.Tensor) -> torch.Tensor:
    """
    输出约束在 [0,1]。
    不直接 clamp 参与损失，避免梯度完全截断。
    """
    return (F.relu(-x) + F.relu(x - 1.0)).mean()


@torch.no_grad()
def project_phi_to_orthogonal_(cs_matrix_module):
    """
    每次更新 Phi 后重新投影到列正交空间。
    对可学习测量矩阵非常重要，能避免 Phi 退化。
    """
    q_mat, _ = torch.linalg.qr(cs_matrix_module.Phi.data, mode="reduced")
    cs_matrix_module.Phi.data.copy_(q_mat[:, :cs_matrix_module.Phi.shape[1]])


# =========================================================
# 2. Learnable CS Matrix
# =========================================================
class LearnableCSMatrix(nn.Module):
    """
    可学习压缩感知矩阵:
        Phi: [N, q]
        y = Phi^T x_block

    N = block_size * block_size
    q = ceil(cs_ratio * N)
    """
    def __init__(self, N, q, device):
        super().__init__()

        if q > N:
            raise ValueError(f"q should be <= N, got q={q}, N={N}")

        rand_mat = torch.randn(N, q, device=device)
        q_mat, _ = torch.linalg.qr(rand_mat, mode="reduced")

        self.Phi = nn.Parameter(q_mat.float())

    def forward(self):
        return self.Phi


# =========================================================
# 3. ToolWear_RGB Dataset
# =========================================================
class ToolWearExpertDataset(Dataset):
    """
    适配:
        data/ToolWear_RGB/rgb_x_0.npy
        data/ToolWear_RGB/rgb_x_1.npy
        data/ToolWear_RGB/rgb_x_2.npy
        data/ToolWear_RGB/rgb_x_3.npy

    数据真实格式:
        [N, 3, 32, 32]
        float64
        数值以 0 为中心

    改进点:
    1. 使用 mmap_mode='r'，不一次性加载全部数据。
    2. 自动兼容 NCHW / NHWC。
    3. 归一化参数保存进 checkpoint，测试时完全复用。
    4. 默认不做空间翻转增强，因为 32x32 可能是信号二维化，不是普通自然图像。
    """
    def __init__(
        self,
        root_dir,
        target_class,
        split="train",
        split_ratio=0.85,
        seed=42,
        norm="sym_p99",
        stats=None,
        augment=False,
        max_stat_samples=4096
    ):
        super().__init__()

        self.root_dir = root_dir
        self.target_class = int(target_class)
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.norm = norm
        self.augment = bool(augment and split == "train")
        self.max_stat_samples = max_stat_samples

        self.file_path = os.path.join(root_dir, f"rgb_x_{self.target_class}.npy")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        self.raw = np.load(self.file_path, mmap_mode="r", allow_pickle=False)

        if self.raw.ndim != 4:
            raise ValueError(f"Expected 4D array, got shape={self.raw.shape}")

        if self.raw.shape[1] in (1, 3):
            self.layout = "NCHW"
            self.num_samples = self.raw.shape[0]
            self.channels = self.raw.shape[1]
            self.height = self.raw.shape[2]
            self.width = self.raw.shape[3]
        elif self.raw.shape[-1] in (1, 3):
            self.layout = "NHWC"
            self.num_samples = self.raw.shape[0]
            self.channels = self.raw.shape[-1]
            self.height = self.raw.shape[1]
            self.width = self.raw.shape[2]
        else:
            raise ValueError(f"Cannot infer channel dim from shape={self.raw.shape}")

        if self.height != 32 or self.width != 32:
            raise ValueError(
                f"ToolWear_RGB should be 32x32, got H={self.height}, W={self.width}"
            )

        rng = np.random.RandomState(seed)
        all_indices = rng.permutation(self.num_samples)

        split_idx = int(self.num_samples * split_ratio)

        if split == "train":
            self.indices = all_indices[:split_idx]
        elif split in ["val", "test"]:
            self.indices = all_indices[split_idx:]
        elif split == "all":
            self.indices = all_indices
        else:
            raise ValueError(f"Unsupported split: {split}")

        if stats is None:
            self.stats = self._estimate_stats()
        else:
            self.stats = stats

    def _read_one(self, idx):
        x = np.asarray(self.raw[idx])

        if self.layout == "NHWC":
            x = np.transpose(x, (2, 0, 1))

        x = x.astype(np.float32, copy=False)
        return x

    def _read_rows(self, rows):
        x = np.asarray(self.raw[rows])

        if self.layout == "NHWC":
            x = np.transpose(x, (0, 3, 1, 2))

        x = x.astype(np.float32, copy=False)
        return x

    def _estimate_stats(self):
        """
        从样本中抽取部分行估计归一化参数。
        对 19032 个样本的 rgb_x_2.npy，不会全量载入。
        """
        n = min(self.num_samples, self.max_stat_samples)

        if n <= 0:
            raise RuntimeError("No samples found for statistics.")

        rng = np.random.RandomState(self.seed + 123)
        rows = rng.choice(self.num_samples, size=n, replace=False)
        rows = np.sort(rows)

        x = self._read_rows(rows)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        flat = x.reshape(-1)

        raw_min = float(np.min(flat))
        raw_max = float(np.max(flat))
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        abs_p99 = float(np.percentile(np.abs(flat), 99.0))
        p1 = float(np.percentile(flat, 1.0))
        p99 = float(np.percentile(flat, 99.0))

        if abs_p99 < 1e-8:
            abs_p99 = 1.0
        if std < 1e-8:
            std = 1.0
        if abs(p99 - p1) < 1e-8:
            p1, p99 = raw_min, raw_max

        return {
            "layout": self.layout,
            "shape": tuple(self.raw.shape),
            "raw_min": raw_min,
            "raw_max": raw_max,
            "mean": mean,
            "std": std,
            "abs_p99": abs_p99,
            "p1": p1,
            "p99": p99,
            "norm": self.norm,
        }

    def _normalize(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.norm == "none":
            return x

        if self.norm == "sym_p99":
            # 适合当前数据：信号以 0 为中心
            # 0 -> 0.5
            scale = float(self.stats.get("abs_p99", 1.0))
            scale = max(scale, 1e-8)
            x = np.clip(x, -scale, scale)
            x = x / (2.0 * scale) + 0.5
            return np.clip(x, 0.0, 1.0).astype(np.float32)

        if self.norm == "minmax":
            # 不做 percentile 截断，更保守
            mn = float(self.stats.get("raw_min", 0.0))
            mx = float(self.stats.get("raw_max", 1.0))
            x = (x - mn) / (mx - mn + 1e-8)
            return np.clip(x, 0.0, 1.0).astype(np.float32)

        if self.norm == "p1p99":
            p1 = float(self.stats.get("p1", 0.0))
            p99 = float(self.stats.get("p99", 1.0))
            x = np.clip(x, p1, p99)
            x = (x - p1) / (p99 - p1 + 1e-8)
            return np.clip(x, 0.0, 1.0).astype(np.float32)

        if self.norm == "zscore":
            mean = float(self.stats.get("mean", 0.0))
            std = float(self.stats.get("std", 1.0))
            x = (x - mean) / (3.0 * std + 1e-8)
            x = np.clip(x, -1.0, 1.0)
            x = x * 0.5 + 0.5
            return np.clip(x, 0.0, 1.0).astype(np.float32)

        raise ValueError(f"Unsupported norm mode: {self.norm}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x = self._read_one(real_idx)
        x = self._normalize(x)

        x = torch.from_numpy(np.ascontiguousarray(x)).float()

        if self.augment:
            # 默认关闭。
            # 如果你明确认为 32x32 是普通图像纹理，可以开启。
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])

        return x

    def info_string(self):
        return (
            f"file={self.file_path}, "
            f"layout={self.layout}, "
            f"shape={tuple(self.raw.shape)}, "
            f"split={self.split}, "
            f"samples={len(self)}, "
            f"norm={self.norm}, "
            f"stats={self.stats}"
        )


# =========================================================
# 4. CS Operator
# =========================================================
_PATCH_OP_CACHE = {}


def _get_patch_ops(h, w, block_size, device):
    key = (
        h,
        w,
        block_size,
        device.type,
        device.index if device.type == "cuda" else -1
    )

    if key not in _PATCH_OP_CACHE:
        if h % block_size != 0 or w % block_size != 0:
            raise ValueError(
                f"H,W must be divisible by block_size. "
                f"Got H={h}, W={w}, block_size={block_size}"
            )

        unfold = nn.Unfold(
            kernel_size=block_size,
            stride=block_size
        ).to(device)

        fold = nn.Fold(
            output_size=(h, w),
            kernel_size=block_size,
            stride=block_size
        ).to(device)

        _PATCH_OP_CACHE[key] = (unfold, fold)

    return _PATCH_OP_CACHE[key]


def make_A_AT_for_patch(h, w, block_size, Phi):
    """
    输入:
        x: [B, 1, H, W]

    unfold 后:
        patches: [B, N, L]
        N = block_size * block_size

    压缩:
        y = Phi^T patches
        y: [B, q, L]
    """
    unfold, fold = _get_patch_ops(h, w, block_size, Phi.device)

    def A(x):
        patches = unfold(x)
        y = torch.matmul(Phi.t(), patches)
        return y

    def AT(y):
        patches = torch.matmul(Phi, y)
        x = fold(patches)
        return x

    return A, AT


# =========================================================
# 5. Metrics / Visualization
# =========================================================
def calculate_metrics(pred, target):
    """
    pred / target: [B, C, H, W], in [0,1]

    返回:
        psnr_sum, ssim_sum, batch_size
    """
    pred = pred.detach().clamp(0, 1).cpu().numpy().astype(np.float64)
    target = target.detach().clamp(0, 1).cpu().numpy().astype(np.float64)

    pred = np.transpose(pred, (0, 2, 3, 1))
    target = np.transpose(target, (0, 2, 3, 1))

    psnr_sum = 0.0
    ssim_sum = 0.0
    batch_size = pred.shape[0]

    for i in range(batch_size):
        p = pred[i]
        t = target[i]

        if p.shape[-1] == 1:
            p = p[..., 0]
            t = t[..., 0]
            channel_axis = None
        else:
            channel_axis = -1

        min_hw = min(p.shape[0], p.shape[1])
        win_size = 7 if min_hw >= 7 else 3

        psnr_sum += psnr_calc(t, p, data_range=1.0)
        ssim_sum += ssim_calc(
            t,
            p,
            data_range=1.0,
            channel_axis=channel_axis,
            win_size=win_size
        )

    return psnr_sum, ssim_sum, batch_size


def save_validation_images(x_gt, x_rec, epoch, save_dir):
    if x_gt is None or x_rec is None:
        return

    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    x_gt = x_gt[:8].detach().cpu().clamp(0, 1)
    x_rec = x_rec[:8].detach().cpu().clamp(0, 1)
    x_err = torch.abs(x_gt - x_rec).mul(4.0).clamp(0, 1)

    comparison = torch.cat([x_gt, x_rec, x_err], dim=0)
    nrow = x_gt.shape[0]

    grid = vutils.make_grid(
        comparison,
        nrow=nrow,
        padding=2,
        normalize=False
    )

    vutils.save_image(
        grid,
        os.path.join(vis_dir, f"epoch_{epoch:04d}_gt_rec_err.png")
    )


# =========================================================
# 6. Validation
# =========================================================
@torch.no_grad()
def run_validation(model, cs_matrix_module, val_loader, args, device):
    model.eval()
    cs_matrix_module.eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    total_imgs = 0

    vis_gt = None
    vis_rec = None

    Phi = cs_matrix_module()

    for i, x_val in enumerate(val_loader):
        x_val = x_val.to(device, non_blocking=True).float()

        b, c, h, w = x_val.shape
        x_flat = x_val.reshape(b * c, 1, h, w)

        A_func, AT_func = make_A_AT_for_patch(
            h=h,
            w=w,
            block_size=args.block_size,
            Phi=Phi
        )

        y = A_func(x_flat)

        with autocast(enabled=(device.type == "cuda")):
            x_rec_flat = call_model(
                model,
                y,
                A_func,
                AT_func,
                use_amp=(device.type == "cuda")
            )

        x_rec_flat = x_rec_flat[..., :h, :w]
        x_rec = x_rec_flat.reshape(b, c, h, w).clamp(0, 1)

        p_sum, s_sum, n = calculate_metrics(x_rec, x_val)
        psnr_sum += p_sum
        ssim_sum += s_sum
        total_imgs += n

        if i == 0:
            vis_gt = x_val.detach()
            vis_rec = x_rec.detach()

    avg_psnr = psnr_sum / max(total_imgs, 1)
    avg_ssim = ssim_sum / max(total_imgs, 1)

    return avg_psnr, avg_ssim, vis_gt, vis_rec


# =========================================================
# 7. Checkpoint
# =========================================================
def save_checkpoint(
    path,
    model,
    cs_matrix_module,
    optimizer_model,
    optimizer_phi,
    scheduler_model,
    scheduler_phi,
    scaler,
    args,
    epoch,
    best_psnr,
    norm_stats
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model_state_dict": model.state_dict(),
        "matrix_state_dict": cs_matrix_module.state_dict(),
        "optimizer_model_state_dict": optimizer_model.state_dict(),
        "optimizer_phi_state_dict": optimizer_phi.state_dict(),
        "scheduler_model_state_dict": scheduler_model.state_dict(),
        "scheduler_phi_state_dict": scheduler_phi.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "norm_stats": norm_stats,
        "config": vars(args),
    }, path)


def load_unet(sd_path, logger):
    try:
        unet = UNet2DConditionModel.from_pretrained(
            sd_path,
            subfolder="unet",
            local_files_only=True
        )
        logger.info("UNet loaded from sd_path/unet")
    except Exception:
        unet = UNet2DConditionModel.from_pretrained(
            sd_path,
            local_files_only=True
        )
        logger.info("UNet loaded from sd_path root")

    return unet


# =========================================================
# 8. Main Training
# =========================================================
def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data/ToolWear_RGB")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_idm_phys")
    parser.add_argument("--sd_path", type=str, default="./sd15")
    parser.add_argument("--target_class", type=int, default=0)

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_number", type=int, default=8)

    # 如果你想严格把 32x32 当作一个完整测量块，设为 32。
    # 如果想保留局部块压缩感知基线，设为 8。
    parser.add_argument("--block_size", type=int, default=32)

    parser.add_argument("--cs_ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split_ratio", type=float, default=0.85)

    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)

    parser.add_argument("--norm", type=str, default="sym_p99",
                        choices=["sym_p99", "minmax", "p1p99", "zscore", "none"])

    parser.add_argument("--augment", action="store_true")

    # Phi warm-up
    parser.add_argument("--phi_warmup_epochs", type=int, default=8)

    # Loss weights
    parser.add_argument("--lambda_img", type=float, default=1.0)
    parser.add_argument("--lambda_meas", type=float, default=0.5)
    parser.add_argument("--lambda_orth", type=float, default=0.02)
    parser.add_argument("--lambda_range", type=float, default=0.02)

    parser.add_argument("--ffl_start_epoch", type=int, default=8)
    parser.add_argument("--ffl_max_weight", type=float, default=0.05)

    parser.add_argument("--grad_clip_model", type=float, default=1.0)
    parser.add_argument("--grad_clip_phi", type=float, default=0.25)

    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.save_dir = ensure_class_save_dir(args.save_dir, args.target_class)

    exp_name = (
        f"MyIDM_ToolWear_C{args.target_class}"
        f"_R{args.cs_ratio}_B{args.block_size}_T{args.step_number}"
    )

    logger = setup_logging(args.save_dir, exp_name)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs", exp_name))

    logger.info("=" * 80)
    logger.info("Training MyIDM on ToolWear_RGB")
    logger.info(f"Config: {vars(args)}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)

    # -----------------------------------------------------
    # 1. Data
    # -----------------------------------------------------
    train_dataset = ToolWearExpertDataset(
        root_dir=args.data_dir,
        target_class=args.target_class,
        split="train",
        split_ratio=args.split_ratio,
        seed=args.seed,
        norm=args.norm,
        stats=None,
        augment=args.augment
    )

    val_dataset = ToolWearExpertDataset(
        root_dir=args.data_dir,
        target_class=args.target_class,
        split="val",
        split_ratio=args.split_ratio,
        seed=args.seed,
        norm=args.norm,
        stats=train_dataset.stats,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0)
    )

    sample = train_dataset[0]

    logger.info(train_dataset.info_string())
    logger.info(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    logger.info(
        f"Sample shape: {tuple(sample.shape)} | "
        f"Range after norm: [{sample.min().item():.4f}, {sample.max().item():.4f}]"
    )

    _, c, h, w = 1, sample.shape[0], sample.shape[1], sample.shape[2]

    if h % args.block_size != 0 or w % args.block_size != 0:
        raise ValueError(
            f"block_size={args.block_size} must divide 32. "
            f"Recommended: 8, 16, or 32."
        )

    # -----------------------------------------------------
    # 2. Model
    # -----------------------------------------------------
    unet = load_unet(args.sd_path, logger)
    model = Net(T=args.step_number, unet=unet).to(device)

    # -----------------------------------------------------
    # 3. Learnable CS Matrix
    # -----------------------------------------------------
    N = args.block_size ** 2
    q = max(1, int(np.ceil(args.cs_ratio * N)))

    cs_matrix_module = LearnableCSMatrix(N, q, device).to(device)

    logger.info(f"Learnable CS Matrix: N={N}, q={q}, Ratio={args.cs_ratio}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # -----------------------------------------------------
    # 4. Optimizers and Schedulers
    # -----------------------------------------------------
    optimizer_model = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    optimizer_phi = torch.optim.AdamW(
        cs_matrix_module.parameters(),
        lr=args.lr * 0.02,
        weight_decay=0.0
    )

    scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_model,
        T_max=args.epoch,
        eta_min=1e-6
    )

    phi_sched_epochs = max(1, args.epoch - args.phi_warmup_epochs)

    scheduler_phi = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phi,
        T_max=phi_sched_epochs,
        eta_min=max(args.lr * 1e-3, 1e-6)
    )

    criterion_img = CharbonnierLoss().to(device)
    criterion_meas = CharbonnierLoss().to(device)
    criterion_freq = FocalFrequencyLoss().to(device)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_psnr = -1.0

    # -----------------------------------------------------
    # 5. Resume
    # -----------------------------------------------------
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        ckpt = torch.load(args.resume, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        cs_matrix_module.load_state_dict(ckpt["matrix_state_dict"], strict=True)

        if "optimizer_model_state_dict" in ckpt:
            optimizer_model.load_state_dict(ckpt["optimizer_model_state_dict"])
        if "optimizer_phi_state_dict" in ckpt:
            optimizer_phi.load_state_dict(ckpt["optimizer_phi_state_dict"])
        if "scheduler_model_state_dict" in ckpt:
            scheduler_model.load_state_dict(ckpt["scheduler_model_state_dict"])
        if "scheduler_phi_state_dict" in ckpt:
            scheduler_phi.load_state_dict(ckpt["scheduler_phi_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("best_psnr", -1.0))

        logger.info(f"Resumed from {args.resume}")
        logger.info(f"Start epoch: {start_epoch}, best_psnr={best_psnr:.4f}")

    # -----------------------------------------------------
    # 6. Training Loop
    # -----------------------------------------------------
    for epoch in range(start_epoch, args.epoch + 1):
        model.train()

        phi_trainable = epoch > args.phi_warmup_epochs
        cs_matrix_module.train(phi_trainable)
        cs_matrix_module.Phi.requires_grad_(phi_trainable)

        if epoch < args.ffl_start_epoch:
            ffl_weight = 0.0
        else:
            ffl_weight = min(
                args.ffl_max_weight,
                args.ffl_max_weight * (epoch - args.ffl_start_epoch + 1)
                / max(1, args.epoch - args.ffl_start_epoch + 1)
            )

        loss_sum = 0.0
        sample_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epoch} | PhiTrain={phi_trainable} | FFL={ffl_weight:.4f}",
            ncols=140
        )

        for step, x_gt in enumerate(pbar):
            x_gt = x_gt.to(device, non_blocking=True).float()

            b, c, h, w = x_gt.shape
            x_flat = x_gt.reshape(b * c, 1, h, w)

            Phi = cs_matrix_module()

            A_func, AT_func = make_A_AT_for_patch(
                h=h,
                w=w,
                block_size=args.block_size,
                Phi=Phi
            )

            # y 是当前 Phi 下的压缩测量。
            # 这里 detach y，避免 target measurement 自己跟着 Phi 漂移。
            with torch.no_grad():
                y = A_func(x_flat).detach()

            optimizer_model.zero_grad(set_to_none=True)
            if phi_trainable:
                optimizer_phi.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                x_rec_flat = call_model(
                    model,
                    y,
                    A_func,
                    AT_func,
                    use_amp=(device.type == "cuda")
                )

                x_rec_flat = x_rec_flat[..., :h, :w]

                loss_img = criterion_img(x_rec_flat, x_flat)
                loss_meas = criterion_meas(A_func(x_rec_flat), y)
                loss_rng = range_regularization(x_rec_flat)

                loss = (
                    args.lambda_img * loss_img
                    + args.lambda_meas * loss_meas
                    + args.lambda_range * loss_rng
                )

                loss_orth = torch.tensor(0.0, device=device)
                if phi_trainable:
                    loss_orth = orthogonality_loss(Phi)
                    loss = loss + args.lambda_orth * loss_orth

                loss_freq = torch.tensor(0.0, device=device)
                if ffl_weight > 0:
                    loss_freq = criterion_freq(x_rec_flat, x_flat)
                    loss = loss + ffl_weight * loss_freq

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer_model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip_model
            )

            if phi_trainable:
                scaler.unscale_(optimizer_phi)
                torch.nn.utils.clip_grad_norm_(
                    cs_matrix_module.parameters(),
                    max_norm=args.grad_clip_phi
                )

            scaler.step(optimizer_model)

            if phi_trainable:
                scaler.step(optimizer_phi)

            scaler.update()

            if phi_trainable:
                project_phi_to_orthogonal_(cs_matrix_module)

            loss_sum += loss.item() * b
            sample_count += b

            global_step = (epoch - 1) * len(train_loader) + step

            if step % 10 == 0:
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/Image", loss_img.item(), global_step)
                writer.add_scalar("Loss/Measurement", loss_meas.item(), global_step)
                writer.add_scalar("Loss/Range", loss_rng.item(), global_step)
                writer.add_scalar("Loss/Orth", loss_orth.item(), global_step)
                writer.add_scalar("Loss/Frequency", loss_freq.item(), global_step)
                writer.add_scalar("Schedule/FFL_Weight", ffl_weight, global_step)

            pbar.set_postfix({
                "L": f"{loss.item():.4f}",
                "Img": f"{loss_img.item():.4f}",
                "Meas": f"{loss_meas.item():.4f}",
                "Rng": f"{loss_rng.item():.4f}",
                "Frq": f"{loss_freq.item():.4f}",
                "Orth": f"{loss_orth.item():.4f}",
            })

        scheduler_model.step()

        if phi_trainable:
            scheduler_phi.step()

        avg_train_loss = loss_sum / max(sample_count, 1)

        writer.add_scalar("LR/Model", optimizer_model.param_groups[0]["lr"], epoch)
        writer.add_scalar("LR/Phi", optimizer_phi.param_groups[0]["lr"], epoch)
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)

        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"LR_Model: {optimizer_model.param_groups[0]['lr']:.2e} | "
            f"LR_Phi: {optimizer_phi.param_groups[0]['lr']:.2e}"
        )

        # -------------------------------------------------
        # 7. Validation
        # -------------------------------------------------
        if epoch % args.val_interval == 0:
            avg_psnr, avg_ssim, vis_gt, vis_rec = run_validation(
                model,
                cs_matrix_module,
                val_loader,
                args,
                device
            )

            logger.info(
                f"Epoch {epoch} | "
                f"Val PSNR: {avg_psnr:.2f} dB | "
                f"Val SSIM: {avg_ssim:.4f}"
            )

            writer.add_scalar("Val/PSNR", avg_psnr, epoch)
            writer.add_scalar("Val/SSIM", avg_ssim, epoch)

            if epoch % args.vis_interval == 0:
                save_validation_images(vis_gt, vis_rec, epoch, args.save_dir)

            latest_path = os.path.join(
                args.save_dir,
                f"latest_model_cls{args.target_class}.pth"
            )

            save_checkpoint(
                path=latest_path,
                model=model,
                cs_matrix_module=cs_matrix_module,
                optimizer_model=optimizer_model,
                optimizer_phi=optimizer_phi,
                scheduler_model=scheduler_model,
                scheduler_phi=scheduler_phi,
                scaler=scaler,
                args=args,
                epoch=epoch,
                best_psnr=best_psnr,
                norm_stats=train_dataset.stats
            )

            if epoch % args.save_interval == 0:
                periodic_path = os.path.join(
                    args.save_dir,
                    f"epoch_{epoch:04d}_model_cls{args.target_class}.pth"
                )

                save_checkpoint(
                    path=periodic_path,
                    model=model,
                    cs_matrix_module=cs_matrix_module,
                    optimizer_model=optimizer_model,
                    optimizer_phi=optimizer_phi,
                    scheduler_model=scheduler_model,
                    scheduler_phi=scheduler_phi,
                    scaler=scaler,
                    args=args,
                    epoch=epoch,
                    best_psnr=best_psnr,
                    norm_stats=train_dataset.stats
                )

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr

                best_path = os.path.join(
                    args.save_dir,
                    f"best_model_cls{args.target_class}.pth"
                )

                save_checkpoint(
                    path=best_path,
                    model=model,
                    cs_matrix_module=cs_matrix_module,
                    optimizer_model=optimizer_model,
                    optimizer_phi=optimizer_phi,
                    scheduler_model=scheduler_model,
                    scheduler_phi=scheduler_phi,
                    scaler=scaler,
                    args=args,
                    epoch=epoch,
                    best_psnr=best_psnr,
                    norm_stats=train_dataset.stats
                )

                logger.info(f"==> Best Saved: {best_path} | PSNR={best_psnr:.2f} dB")

    writer.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        print("\n" + "!" * 80)
        print("CRITICAL ERROR IN TRAINING")
        print("!" * 80)
        traceback.print_exc()
        print("!" * 80)