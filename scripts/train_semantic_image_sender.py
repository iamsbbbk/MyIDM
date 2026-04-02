import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------
# bootstrap: 保证从 scripts/ 直接运行时可导入项目根目录模块
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from industrial_semantic.config import load_config
from industrial_semantic.dataset import _load_any_array, _infer_loaded_modality, _ensure_image_ch_first
from industrial_semantic.utils import (
    set_seed,
    get_logger,
    compute_class_weights_from_hist,
    macro_classification_metrics,
    false_idle_metrics,
)


# =========================================================
# 基础工具
# =========================================================
SUPPORTED_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".csv", ".txt", ".mat"}


def infer_class_from_filename(path: Path) -> Optional[int]:
    """
    从文件名推断类标，例如：
    rgb_x_0.npy -> 0
    rgb_x_1.npy -> 1
    ...
    """
    stem = path.stem.lower()

    # 优先匹配最后一个数字
    m = re.search(r"(\d+)(?!.*\d)", stem)
    if m:
        return int(m.group(1))

    return None


def is_probably_label_file(path: Path) -> bool:
    stem = path.stem.lower()
    if "label" in stem:
        return True
    if "_y_" in stem or stem.startswith("y_") or stem.endswith("_y"):
        return True
    if "target" in stem:
        return True
    return False


def mapped_to_contact_wear(mapped_class: int):
    """
    mapped_class:
        0 = idle
        1 = initial
        2 = steady
        3 = accelerating

    contact:
        0 = idle
        1 = non-idle

    wear:
        -1 = ignore (idle)
         0 = initial
         1 = steady
         2 = accelerating
    """
    if mapped_class == 0:
        return 0, -1
    return 1, mapped_class - 1


def save_index_csv(entries: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "sample_idx",
        "mapped_class",
        "contact_label",
        "wear_label",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in entries:
            writer.writerow({k: row.get(k, "") for k in fields})


# =========================================================
# 数据扫描与索引构建
# =========================================================
def build_toolwear_rgb_entries(
    root_dir: Path,
    num_classes: int = 4,
    max_files: int = 0,
    max_samples_per_file: int = 0,
    max_samples_per_class: int = 0,
    logger=None,
) -> List[Dict]:
    """
    扫描 ToolWear_RGB 目录，构建逐样本索引。
    支持 batched image file:
        (N, 3, 32, 32)
        (N, 32, 32, 3)
    """
    logger = logger or get_logger("build_toolwear_rgb_entries")

    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)
    files = sorted(files)

    if len(files) == 0:
        raise RuntimeError(f"在 {root_dir} 下未找到支持的数据文件")

    entries: List[Dict] = []
    class_counter = {i: 0 for i in range(num_classes)}
    used_files = 0

    for p in files:
        if max_files > 0 and used_files >= max_files:
            break

        if is_probably_label_file(p):
            logger.info(f"跳过疑似 label 文件: {p}")
            continue

        try:
            arr = _load_any_array(p)
        except Exception as e:
            logger.warning(f"读取失败，跳过文件: {p}, err={e}")
            continue

        kind = _infer_loaded_modality(arr)
        class_id = infer_class_from_filename(p)

        logger.info(f"[scan] file={p} shape={tuple(arr.shape)} kind={kind} class_id={class_id}")

        if class_id is None:
            logger.warning(f"无法从文件名推断类标，跳过: {p}")
            continue
        if class_id < 0 or class_id >= num_classes:
            logger.warning(f"类标越界，跳过: {p}, class_id={class_id}")
            continue

        if kind == "label_vector":
            logger.info(f"检测到 label 向量文件，跳过: {p}")
            continue

        if kind == "image_single":
            if max_samples_per_class > 0 and class_counter[class_id] >= max_samples_per_class:
                continue

            contact_label, wear_label = mapped_to_contact_wear(class_id)
            entries.append({
                "path": str(p),
                "sample_idx": -1,
                "mapped_class": class_id,
                "contact_label": contact_label,
                "wear_label": wear_label,
            })
            class_counter[class_id] += 1
            used_files += 1

        elif kind == "image_batch":
            n = arr.shape[0]
            if max_samples_per_file > 0:
                n = min(n, max_samples_per_file)

            added = 0
            for i in range(n):
                if max_samples_per_class > 0 and class_counter[class_id] >= max_samples_per_class:
                    break

                contact_label, wear_label = mapped_to_contact_wear(class_id)
                entries.append({
                    "path": str(p),
                    "sample_idx": i,
                    "mapped_class": class_id,
                    "contact_label": contact_label,
                    "wear_label": wear_label,
                })
                class_counter[class_id] += 1
                added += 1

            logger.info(f"[scan] image_batch accepted: file={p}, added={added}")
            used_files += 1

        else:
            logger.info(f"非图像模态，跳过: {p}, kind={kind}")

    if len(entries) == 0:
        raise RuntimeError("没有构建出任何有效样本索引，请检查 ToolWear_RGB 目录结构和文件名")

    logger.info(f"总样本数: {len(entries)}")
    logger.info(f"类别分布: {class_counter}")
    return entries


def stratified_split(
    entries: List[Dict],
    valid_ratio: float = 0.2,
    seed: int = 42,
):
    """
    按 mapped_class 分层随机划分 train / valid
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[Dict]] = {}

    for e in entries:
        cls = int(e["mapped_class"])
        by_class.setdefault(cls, []).append(e)

    train_entries, valid_entries = [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_valid = max(1, int(len(items) * valid_ratio)) if len(items) > 1 else 0
        valid_entries.extend(items[:n_valid])
        train_entries.extend(items[n_valid:])

    rng.shuffle(train_entries)
    rng.shuffle(valid_entries)
    return train_entries, valid_entries


# =========================================================
# Dataset
# =========================================================
class ToolWearRGBIndexDataset(Dataset):
    def __init__(
        self,
        entries: List[Dict],
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        cache_files: bool = True,
    ):
        self.entries = entries
        self.cache_files = cache_files
        self.file_cache: Dict[str, np.ndarray] = {}

        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.clip(self.image_std, 1e-8, None)

    def __len__(self):
        return len(self.entries)

    def _normalize_range(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if image.max() > 1.5:
            image = image / 255.0
        return np.clip(image, 0.0, 1.0)

    def _load_file(self, path: str) -> np.ndarray:
        if self.cache_files and path in self.file_cache:
            return self.file_cache[path]

        arr = _load_any_array(Path(path))
        kind = _infer_loaded_modality(arr)

        if kind == "image_single":
            img = _ensure_image_ch_first(arr).astype(np.float32)
            img = self._normalize_range(img)
            data = img[None, ...]  # [1,C,H,W]

        elif kind == "image_batch":
            arr = np.asarray(arr)
            # [N,C,H,W]
            if arr.ndim == 4 and arr.shape[1] in [1, 3]:
                data = arr.astype(np.float32)
            # [N,H,W,C]
            elif arr.ndim == 4 and arr.shape[-1] in [1, 3]:
                data = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)
            else:
                # fallback: 一张张处理
                imgs = [_ensure_image_ch_first(arr[i]).astype(np.float32) for i in range(arr.shape[0])]
                data = np.stack(imgs, axis=0)

            # 单通道扩成 3 通道
            if data.shape[1] == 1:
                data = np.repeat(data, 3, axis=1)

            data = self._normalize_range(data)

        else:
            raise ValueError(f"文件不是有效图像模态: path={path}, kind={kind}")

        if self.cache_files:
            self.file_cache[path] = data

        return data

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        path = entry["path"]
        sample_idx = int(entry["sample_idx"])

        data = self._load_file(path)

        if sample_idx < 0:
            image = data[0]
        else:
            image = data[sample_idx]

        image_raw = image.astype(np.float32)  # [C,H,W], 0~1
        image_norm = ((image_raw - self.image_mean) / self.image_std).astype(np.float32)

        return {
            "image": torch.from_numpy(image_norm).float(),
            "image_raw": torch.from_numpy(image_raw).float(),
            "mapped_class": torch.tensor(int(entry["mapped_class"]), dtype=torch.long),
            "contact_label": torch.tensor(int(entry["contact_label"]), dtype=torch.long),
            "wear_label": torch.tensor(int(entry["wear_label"]), dtype=torch.long),
            "meta": {
                "path": path,
                "sample_idx": sample_idx,
            },
        }


# =========================================================
# Model
# =========================================================
class ConvBNAct2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct2d(in_ch, out_ch, 3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch or stride != 1:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.short = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        identity = self.short(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)


class SemanticImageSenderModel(nn.Module):
    """
    图像发送端模型：
    - mapped_head: 4 类
    - contact_head: 2 类 (idle / non-idle)
    - wear_head: 3 类 (ignore idle)
    - decoder: 图像重构辅助头
    """
    def __init__(
        self,
        num_mapped_classes: int = 4,
        num_contact_classes: int = 2,
        num_wear_classes: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
        use_decoder: bool = True,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            ConvBNAct2d(3, c1, kernel_size=3, stride=1),
            ResBlock2d(c1, c1, stride=1, dropout=dropout),
        )
        self.stage1 = nn.Sequential(
            ResBlock2d(c1, c2, stride=2, dropout=dropout),  # 32 -> 16
            ResBlock2d(c2, c2, stride=1, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ResBlock2d(c2, c3, stride=2, dropout=dropout),  # 16 -> 8
            ResBlock2d(c3, c3, stride=1, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            ResBlock2d(c3, c4, stride=2, dropout=dropout),  # 8 -> 4
            ResBlock2d(c4, c4, stride=1, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        self.mapped_head = nn.Sequential(
            nn.Linear(c4, c4 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c4 // 2, num_mapped_classes),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(c4, c4 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c4 // 2, num_contact_classes),
        )
        self.wear_head = nn.Sequential(
            nn.Linear(c4, c4 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c4 // 2, num_wear_classes),
        )

        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder = nn.Sequential(
                nn.Conv2d(c4, c3, 3, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 4 -> 8
                nn.Conv2d(c3, c2, 3, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8 -> 16
                nn.Conv2d(c2, c1, 3, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16 -> 32
                nn.Conv2d(c1, 3, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.decoder = None

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        feat_map = self.stage3(x)

        feat = self.pool(feat_map).flatten(1)
        feat = self.dropout(feat)

        mapped_logits = self.mapped_head(feat)
        contact_logits = self.contact_head(feat)
        wear_logits = self.wear_head(feat)

        reconstruction = self.decoder(feat_map) if self.decoder is not None else None

        mapped_probs = torch.softmax(mapped_logits, dim=1)
        contact_probs = torch.softmax(contact_logits, dim=1)
        wear_probs = torch.softmax(wear_logits, dim=1)

        return {
            "mapped_logits": mapped_logits,
            "contact_logits": contact_logits,
            "wear_logits": wear_logits,
            "mapped_probs": mapped_probs,
            "contact_probs": contact_probs,
            "wear_probs": wear_probs,
            "reconstruction": reconstruction,
        }


# =========================================================
# Train / Eval
# =========================================================
def build_hist(entries: List[Dict], field: str, num_classes: int, ignore_value: Optional[int] = None):
    hist = np.zeros((num_classes,), dtype=np.int64)
    for e in entries:
        v = int(e[field])
        if ignore_value is not None and v == ignore_value:
            continue
        if 0 <= v < num_classes:
            hist[v] += 1
    return hist


def evaluate(model, loader, device, recon_weight: float = 0.2):
    model.eval()

    all_mapped_true, all_mapped_pred = [], []
    all_contact_true, all_contact_pred = [], []
    all_wear_true, all_wear_pred = [], []

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            x_raw = batch["image_raw"].to(device)
            mapped_y = batch["mapped_class"].to(device)
            contact_y = batch["contact_label"].to(device)
            wear_y = batch["wear_label"].to(device)

            out = model(x)

            loss_mapped = F.cross_entropy(out["mapped_logits"], mapped_y)
            loss_contact = F.cross_entropy(out["contact_logits"], contact_y)
            loss_wear = F.cross_entropy(out["wear_logits"], wear_y, ignore_index=-1)
            if out["reconstruction"] is not None:
                loss_recon = F.l1_loss(out["reconstruction"], x_raw)
            else:
                loss_recon = torch.tensor(0.0, device=device)

            loss = loss_mapped + 0.5 * loss_contact + 0.5 * loss_wear + recon_weight * loss_recon

            total_loss += float(loss.item())
            n_batches += 1

            mapped_pred = torch.argmax(out["mapped_probs"], dim=1)
            contact_pred = torch.argmax(out["contact_probs"], dim=1)
            wear_pred = torch.argmax(out["wear_probs"], dim=1)

            all_mapped_true.extend(mapped_y.detach().cpu().numpy().tolist())
            all_mapped_pred.extend(mapped_pred.detach().cpu().numpy().tolist())

            all_contact_true.extend(contact_y.detach().cpu().numpy().tolist())
            all_contact_pred.extend(contact_pred.detach().cpu().numpy().tolist())

            valid_mask = wear_y != -1
            if valid_mask.any():
                all_wear_true.extend(wear_y[valid_mask].detach().cpu().numpy().tolist())
                all_wear_pred.extend(wear_pred[valid_mask].detach().cpu().numpy().tolist())

    mapped_metrics = macro_classification_metrics(all_mapped_true, all_mapped_pred, num_classes=4)
    contact_metrics = macro_classification_metrics(all_contact_true, all_contact_pred, num_classes=2)
    wear_metrics = macro_classification_metrics(all_wear_true, all_wear_pred, num_classes=3) if len(all_wear_true) > 0 else None
    idle_metrics = false_idle_metrics(all_mapped_true, all_mapped_pred, idle_class=0)

    return {
        "loss": total_loss / max(n_batches, 1),
        "mapped": mapped_metrics,
        "contact": contact_metrics,
        "wear": wear_metrics,
        "idle_guard": idle_metrics,
    }


def save_history_json(history: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(history), f, ensure_ascii=False, indent=2)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/industrial_semantic.yaml")
    parser.add_argument("--data_root", type=str, default="./data/ToolWear_RGB")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/industrial_semantic_image")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--recon_weight", type=float, default=0.2)
    parser.add_argument("--disable_decoder", action="store_true")

    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_samples_per_file", type=int, default=0)
    parser.add_argument("--max_samples_per_class", type=int, default=0)
    parser.add_argument("--cache_files", action="store_true")

    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    logger = get_logger("train_semantic_image_sender")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------
    # 构建索引
    # ------------------------------------------
    entries = build_toolwear_rgb_entries(
        root_dir=data_root,
        num_classes=4,
        max_files=args.max_files,
        max_samples_per_file=args.max_samples_per_file,
        max_samples_per_class=args.max_samples_per_class,
        logger=logger,
    )

    train_entries, valid_entries = stratified_split(
        entries,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    save_index_csv(train_entries, save_dir / "train_index.csv")
    save_index_csv(valid_entries, save_dir / "valid_index.csv")

    logger.info(f"train samples: {len(train_entries)}")
    logger.info(f"valid samples: {len(valid_entries)}")

    train_hist = build_hist(train_entries, "mapped_class", 4)
    valid_hist = build_hist(valid_entries, "mapped_class", 4)
    logger.info(f"train mapped hist: {train_hist.tolist()}")
    logger.info(f"valid mapped hist: {valid_hist.tolist()}")

    train_ds = ToolWearRGBIndexDataset(
        train_entries,
        cache_files=args.cache_files,
    )
    valid_ds = ToolWearRGBIndexDataset(
        valid_entries,
        cache_files=args.cache_files,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.startswith("cuda")),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.startswith("cuda")),
        drop_last=False,
    )

    # ------------------------------------------
    # 模型
    # ------------------------------------------
    model = SemanticImageSenderModel(
        num_mapped_classes=4,
        num_contact_classes=2,
        num_wear_classes=3,
        base_channels=args.base_channels,
        dropout=args.dropout,
        use_decoder=(not args.disable_decoder),
    ).to(device)

    logger.info(f"Image sender model created. base_channels={args.base_channels}")

    mapped_w = compute_class_weights_from_hist(train_hist).to(device)
    contact_hist = build_hist(train_entries, "contact_label", 2)
    wear_hist = build_hist(train_entries, "wear_label", 3, ignore_value=-1)

    contact_w = compute_class_weights_from_hist(contact_hist).to(device)
    wear_w = compute_class_weights_from_hist(wear_hist).to(device)

    logger.info(f"mapped weights: {mapped_w.detach().cpu().numpy().tolist()}")
    logger.info(f"contact weights: {contact_w.detach().cpu().numpy().tolist()}")
    logger.info(f"wear weights: {wear_w.detach().cpu().numpy().tolist()}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    history = []
    best_score = -1.0

    # ------------------------------------------
    # 训练
    # ------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            x_raw = batch["image_raw"].to(device)
            mapped_y = batch["mapped_class"].to(device)
            contact_y = batch["contact_label"].to(device)
            wear_y = batch["wear_label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                out = model(x)

                loss_mapped = F.cross_entropy(out["mapped_logits"], mapped_y, weight=mapped_w)
                loss_contact = F.cross_entropy(out["contact_logits"], contact_y, weight=contact_w)
                loss_wear = F.cross_entropy(out["wear_logits"], wear_y, weight=wear_w, ignore_index=-1)

                if out["reconstruction"] is not None:
                    loss_recon = F.l1_loss(out["reconstruction"], x_raw)
                else:
                    loss_recon = torch.tensor(0.0, device=device)

                loss = (
                    loss_mapped
                    + 0.5 * loss_contact
                    + 0.5 * loss_wear
                    + args.recon_weight * loss_recon
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        val_metrics = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            recon_weight=args.recon_weight,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_metrics": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['mapped']['accuracy']:.4f} "
            f"val_f1={val_metrics['mapped']['f1_macro']:.4f} "
            f"false_idle={val_metrics['idle_guard']['false_idle_rate']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "config": cfg,
            "script_args": vars(args),
            "model_type": "image_sender",
            "model_state_dict": model.state_dict(),
            "valid_metrics": val_metrics,
            "class_hist_train": train_hist,
            "class_hist_valid": valid_hist,
            "label_mapping": {
                "mapped_class": {
                    "0": "idle",
                    "1": "initial",
                    "2": "steady",
                    "3": "accelerating",
                },
                "contact_label": {
                    "0": "idle",
                    "1": "non-idle",
                },
                "wear_label": {
                    "-1": "NA",
                    "0": "initial",
                    "1": "steady",
                    "2": "accelerating",
                },
            },
            "model_hparams": {
                "base_channels": args.base_channels,
                "dropout": args.dropout,
                "use_decoder": not args.disable_decoder,
            },
        }

        torch.save(ckpt, save_dir / "last.pt")
        save_history_json(history, save_dir / "history.json")

        score = val_metrics["mapped"]["f1_macro"] - 0.2 * val_metrics["idle_guard"]["false_idle_rate"]
        if score > best_score:
            best_score = score
            torch.save(ckpt, save_dir / "best.pt")
            logger.info(f"更新 best.pt, score={best_score:.6f}")

    logger.info("图像发送端训练完成。")


if __name__ == "__main__":
    main()