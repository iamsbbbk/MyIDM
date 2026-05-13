import os
import math
import random
import numpy as np
import torch


def set_random_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred, gt, data_range=1.0):
    if isinstance(pred, torch.Tensor):
        mse = torch.mean((pred - gt) ** 2).item()
    else:
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        mse = np.mean((pred - gt) ** 2)

    if mse <= 1e-12:
        return 99.0

    return 10.0 * math.log10((data_range ** 2) / mse)


def ssim_batch(pred, gt, data_range=1.0):
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return 0.0

    pred = pred.detach().float().cpu().numpy()
    gt = gt.detach().float().cpu().numpy()

    vals = []

    for i in range(pred.shape[0]):
        p = np.squeeze(pred[i])
        g = np.squeeze(gt[i])

        if p.ndim == 3:
            p = np.mean(p, axis=0)
        if g.ndim == 3:
            g = np.mean(g, axis=0)

        vals.append(ssim(p, g, data_range=data_range))

    return float(np.mean(vals))


def my_zero_pad(img, block_size=32):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / block_size) * block_size)
    new_w = int(np.ceil(w / block_size) * block_size)

    img_pad = np.zeros((new_h, new_w), dtype=img.dtype)
    img_pad[:h, :w] = img

    return img, h, w, img_pad, new_h, new_w


def normalize_toolwear_sample(x, norm="sym_p99", eps=1e-8):
    """
    ToolWear_RGB 的数值以 0 为中心。
    默认使用对称 p99 归一化：
        0 对应 0.5
        负值映射到 0~0.5
        正值映射到 0.5~1
    这样比普通 min-max 更适合振动/磨损信号。
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if norm == "none":
        return x

    if norm == "p1p99":
        p1, p99 = np.percentile(x, [1, 99])
        if abs(p99 - p1) < eps:
            return np.zeros_like(x, dtype=np.float32)
        x = np.clip(x, p1, p99)
        x = (x - p1) / (p99 - p1 + eps)
        return x.astype(np.float32)

    if norm == "sym_p99":
        scale = np.percentile(np.abs(x), 99)
        if scale < eps:
            return np.full_like(x, 0.5, dtype=np.float32)
        x = np.clip(x, -scale, scale)
        x = x / (2.0 * scale + eps) + 0.5
        return x.astype(np.float32)

    if norm == "zscore01":
        mean = np.mean(x)
        std = np.std(x)
        if std < eps:
            return np.full_like(x, 0.5, dtype=np.float32)
        x = (x - mean) / (3.0 * std + eps)
        x = np.clip(x, -1.0, 1.0)
        x = x * 0.5 + 0.5
        return x.astype(np.float32)

    raise ValueError(f"Unsupported norm mode: {norm}")


def rgb_to_gray_tensor_np(x, mode="mean"):
    """
    输入 x: C,H,W
    输出:
        mode=mean/y/first -> 1,H,W
        mode=keep         -> C,H,W
    """
    if x.ndim != 3:
        raise ValueError(f"Expected C,H,W, got shape={x.shape}")

    if mode == "keep":
        return x

    if x.shape[0] == 1:
        return x

    if mode == "first":
        return x[:1]

    if mode == "y":
        r, g, b = x[0:1], x[1:2], x[2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    if mode == "mean":
        return np.mean(x, axis=0, keepdims=True)

    raise ValueError(f"Unsupported gray mode: {mode}")


class ToolWearRGBDataset(torch.utils.data.Dataset):
    """
    适配 data/ToolWear_RGB/rgb_x_*.npy

    文件格式：
        rgb_x_0.npy: N,3,32,32
        rgb_x_1.npy: N,3,32,32
        rgb_x_2.npy: N,3,32,32
        rgb_x_3.npy: N,3,32,32

    使用 mmap_mode='r'，避免一次性加载 700MB+ 数据。
    """
    def __init__(
        self,
        root,
        split="train",
        train_ratio=0.8,
        seed=2025,
        norm="sym_p99",
        gray_mode="mean"
    ):
        super().__init__()

        self.root = root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.norm = norm
        self.gray_mode = gray_mode

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")

        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".npy")
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {root}")

        self.arrays = []
        self.lengths = []

        for path in self.files:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)

            if arr.ndim != 4:
                raise ValueError(f"{path} should be 4D N,C,H,W, got {arr.shape}")

            if arr.shape[1] not in [1, 3]:
                raise ValueError(f"{path} channel should be 1 or 3, got {arr.shape}")

            if arr.shape[2] != 32 or arr.shape[3] != 32:
                raise ValueError(f"{path} spatial size should be 32x32, got {arr.shape}")

            self.arrays.append(arr)
            self.lengths.append(arr.shape[0])

        self.cum_lengths = np.cumsum(self.lengths)
        total = int(self.cum_lengths[-1])

        indices = np.arange(total)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        n_train = int(total * train_ratio)

        if split == "train":
            self.indices = indices[:n_train]
        elif split in ["test", "val"]:
            self.indices = indices[n_train:]
        elif split == "all":
            self.indices = indices
        else:
            raise ValueError(f"Unsupported split: {split}")

        print(
            f"[ToolWearRGBDataset] split={split}, "
            f"samples={len(self.indices)}, "
            f"files={len(self.files)}, "
            f"norm={norm}, gray_mode={gray_mode}"
        )

    def __len__(self):
        return len(self.indices)

    def _locate(self, global_index):
        file_id = int(np.searchsorted(self.cum_lengths, global_index, side="right"))
        prev = 0 if file_id == 0 else self.cum_lengths[file_id - 1]
        local_id = int(global_index - prev)
        return file_id, local_id

    def __getitem__(self, index):
        global_index = int(self.indices[index])
        file_id, local_id = self._locate(global_index)

        x = np.asarray(self.arrays[file_id][local_id], dtype=np.float32)
        x = normalize_toolwear_sample(x, norm=self.norm)
        x = rgb_to_gray_tensor_np(x, mode=self.gray_mode)

        return torch.from_numpy(x).float()


def build_phi(block_size, cs_ratio, device, seed=2025):
    """
    保持原 IDM 的块压缩感知形式：
        Phi: N x q
        N = block_size * block_size
        q = ceil(cs_ratio * N)
    """
    N = block_size * block_size
    q = int(np.ceil(cs_ratio * N))

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    mat = torch.randn(N, N, device=device, generator=generator)
    U, _, Vh = torch.linalg.svd(mat, full_matrices=True)
    Phi = (U @ Vh)[:, :q].contiguous()
    Phi.requires_grad_(False)

    return Phi


def save_phi(Phi, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(Phi.detach().cpu(), path)


def load_phi_or_create(path, block_size, cs_ratio, device, seed=2025):
    if os.path.exists(path):
        Phi = torch.load(path, map_location=device).float()
        print(f"[INFO] Loaded Phi from {path}")
        return Phi.to(device)

    Phi = build_phi(block_size, cs_ratio, device, seed=seed)
    print(f"[INFO] Created new Phi, shape={tuple(Phi.shape)}")
    return Phi


def make_cs_operators(x_shape, Phi, block_size, device, permute=True):
    """
    为当前 batch 构造 A 和 AT。

    x_shape:
        B,C,H,W

    对 ToolWear 默认 C=1,H=W=32。
    如果以后模型支持 RGB，也可以 C=3。
    """
    b, c, h, w = x_shape
    N = block_size * block_size

    total = b * c * h * w

    if total % N != 0:
        raise ValueError(
            f"Total elements {total} must be divisible by block N={N}. "
            f"x_shape={x_shape}, block_size={block_size}"
        )

    if permute:
        perm = torch.randperm(total, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(total, device=device)
    else:
        perm = None
        perm_inv = None

    def A(z):
        flat = z.reshape(-1)

        if permute:
            flat = flat[perm]

        y = flat.reshape(-1, N) @ Phi
        return y

    def AT(y):
        flat = (y @ Phi.t()).reshape(-1)

        if permute:
            flat = flat[perm_inv]

        return flat.reshape(b, c, h, w)

    return A, AT


def frequency_loss(pred, gt):
    """
    轻量频域约束，适合磨损/振动信号二维化后的频谱一致性。
    """
    pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
    gt_fft = torch.fft.rfft2(gt.float(), norm="ortho")

    pred_mag = torch.log1p(torch.abs(pred_fft))
    gt_mag = torch.log1p(torch.abs(gt_fft))

    return torch.mean(torch.abs(pred_mag - gt_mag))


def save_tensor_as_image(x, path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    x = x.detach().float().cpu().clamp(0, 1)

    if x.ndim == 3:
        if x.shape[0] == 1:
            img = x[0].numpy()
            cmap = "gray"
        else:
            img = x.permute(1, 2, 0).numpy()
            cmap = None
    else:
        img = x.numpy()
        cmap = "gray"

    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()