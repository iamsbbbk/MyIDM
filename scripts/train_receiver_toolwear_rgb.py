import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from industrial_semantic.dataset import _load_any_array, _infer_loaded_modality, _ensure_image_ch_first
from industrial_semantic.cs_operator import OperatorFactory
from industrial_semantic.utils import set_seed, get_logger
from model import build_myidm_net


SUPPORTED_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".csv", ".txt", ".mat"}


# =========================================================
# Utils
# =========================================================
def infer_class_from_filename(path: Path) -> Optional[int]:
    stem = path.stem.lower()
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


def save_index_csv(entries: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "sample_idx", "channel_idx", "mapped_class"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)


def psnr_from_mse(mse: float) -> float:
    mse = max(float(mse), 1e-12)
    return 10.0 * math.log10(1.0 / mse)


# =========================================================
# Build entries
# =========================================================
def build_receiver_entries(
    root_dir: Path,
    max_files: int = 0,
    max_samples_per_file: int = 0,
    max_samples_per_class: int = 0,
    logger=None,
):
    logger = logger or get_logger("build_receiver_entries")

    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)
    files = sorted(files)

    if len(files) == 0:
        raise RuntimeError(f"在 {root_dir} 下未找到有效文件")

    entries: List[Dict] = []
    class_counter = {}
    used_files = 0

    for p in files:
        if max_files > 0 and used_files >= max_files:
            break

        if is_probably_label_file(p):
            logger.info(f"跳过 label 文件: {p}")
            continue

        try:
            arr = _load_any_array(p)
        except Exception as e:
            logger.warning(f"读取失败，跳过: {p}, err={e}")
            continue

        kind = _infer_loaded_modality(arr)
        class_id = infer_class_from_filename(p)

        logger.info(f"[scan] file={p} shape={tuple(arr.shape)} kind={kind} class_id={class_id}")

        if class_id is None:
            logger.warning(f"无法推断类标，跳过: {p}")
            continue

        class_counter.setdefault(class_id, 0)

        if kind == "image_single":
            img = _ensure_image_ch_first(arr)
            c = img.shape[0]

            added = 0
            for ch in range(c):
                if max_samples_per_class > 0 and class_counter[class_id] >= max_samples_per_class:
                    break
                entries.append({
                    "path": str(p),
                    "sample_idx": -1,
                    "channel_idx": ch,
                    "mapped_class": class_id,
                })
                class_counter[class_id] += 1
                added += 1

            logger.info(f"[scan] image_single accepted: added={added}")
            used_files += 1

        elif kind == "image_batch":
            n = arr.shape[0]
            if max_samples_per_file > 0:
                n = min(n, max_samples_per_file)

            added = 0
            for i in range(n):
                img = _ensure_image_ch_first(arr[i])
                c = img.shape[0]
                for ch in range(c):
                    if max_samples_per_class > 0 and class_counter[class_id] >= max_samples_per_class:
                        break
                    entries.append({
                        "path": str(p),
                        "sample_idx": i,
                        "channel_idx": ch,
                        "mapped_class": class_id,
                    })
                    class_counter[class_id] += 1
                    added += 1

            logger.info(f"[scan] image_batch accepted: added={added}")
            used_files += 1

        else:
            logger.info(f"非图像模态，跳过: {p}")

    if len(entries) == 0:
        raise RuntimeError("没有构建出任何接收端训练样本，请检查数据目录")

    logger.info(f"接收端训练样本总数: {len(entries)}")
    logger.info(f"类别分布: {class_counter}")
    return entries


def stratified_split(entries: List[Dict], valid_ratio: float = 0.2, seed: int = 42):
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
class ReceiverRGBChannelDataset(Dataset):
    def __init__(self, entries: List[Dict], cache_files: bool = True):
        self.entries = entries
        self.cache_files = cache_files
        self.file_cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.entries)

    def _normalize_range(self, x: np.ndarray):
        x = x.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        return np.clip(x, 0.0, 1.0)

    def _load_file(self, path: str) -> np.ndarray:
        if self.cache_files and path in self.file_cache:
            return self.file_cache[path]

        arr = _load_any_array(Path(path))
        kind = _infer_loaded_modality(arr)

        if kind == "image_single":
            img = _ensure_image_ch_first(arr).astype(np.float32)
            data = img[None, ...]  # [1,C,H,W]

        elif kind == "image_batch":
            arr = np.asarray(arr)
            if arr.ndim == 4 and arr.shape[1] in [1, 3]:
                data = arr.astype(np.float32)
            elif arr.ndim == 4 and arr.shape[-1] in [1, 3]:
                data = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)
            else:
                imgs = [_ensure_image_ch_first(arr[i]).astype(np.float32) for i in range(arr.shape[0])]
                data = np.stack(imgs, axis=0)
        else:
            raise ValueError(f"无效图像文件: path={path}, kind={kind}")

        data = self._normalize_range(data)

        if data.shape[1] == 1:
            data = np.repeat(data, 3, axis=1)

        if self.cache_files:
            self.file_cache[path] = data

        return data

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        path = e["path"]
        sample_idx = int(e["sample_idx"])
        ch = int(e["channel_idx"])

        data = self._load_file(path)

        if sample_idx < 0:
            img = data[0]
        else:
            img = data[sample_idx]

        x = img[ch:ch+1].astype(np.float32)   # [1,H,W]

        return {
            "image": torch.from_numpy(x).float(),
            "mapped_class": torch.tensor(int(e["mapped_class"]), dtype=torch.long),
            "meta": {
                "path": path,
                "sample_idx": sample_idx,
                "channel_idx": ch,
            }
        }


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def evaluate_receiver(net, loader, operator, device, use_amp=True):
    net.eval()

    total_l1 = 0.0
    total_mse = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["image"].to(device)  # [B,1,H,W]
        y = operator.A(x)
        x_hat = net(y, operator.A, operator.AT, use_amp_=use_amp)

        l1 = F.l1_loss(x_hat, x)
        mse = F.mse_loss(x_hat, x)

        total_l1 += float(l1.item())
        total_mse += float(mse.item())
        n_batches += 1

    avg_l1 = total_l1 / max(n_batches, 1)
    avg_mse = total_mse / max(n_batches, 1)
    avg_psnr = psnr_from_mse(avg_mse)

    return {
        "l1": avg_l1,
        "mse": avg_mse,
        "psnr": avg_psnr,
    }


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/ToolWear_RGB")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/myidm_receiver_rgb")
    parser.add_argument("--sd15_path", type=str, default="./sd15")
    parser.add_argument("--checkpoint", type=str, default="", help="可选，用于继续训练")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--img_h", type=int, default=32)
    parser.add_argument("--img_w", type=int, default=32)
    parser.add_argument("--cs_ratio", type=float, default=0.25)
    parser.add_argument("--measurement_dim", type=int, default=0)
    parser.add_argument("--operator_seed", type=int, default=2026)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_samples_per_file", type=int, default=0)
    parser.add_argument("--max_samples_per_class", type=int, default=0)
    parser.add_argument("--cache_files", action="store_true")

    args = parser.parse_args()

    logger = get_logger("train_receiver_toolwear_rgb")
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    entries = build_receiver_entries(
        root_dir=data_root,
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

    train_ds = ReceiverRGBChannelDataset(train_entries, cache_files=args.cache_files)
    valid_ds = ReceiverRGBChannelDataset(valid_entries, cache_files=args.cache_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    operator_factory = OperatorFactory(device=device, dtype=torch.float32)
    operator = operator_factory.get(
        img_h=args.img_h,
        img_w=args.img_w,
        ratio=args.cs_ratio,
        m=args.measurement_dim,
        seed=args.operator_seed,
        op_type="gaussian_flatten",
    )

    net = build_myidm_net(
        T=args.T,
        sd15_path=args.sd15_path,
        checkpoint=args.checkpoint,
        device=device,
        strict=False,
        train_mode=True,
    )

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and args.use_amp))

    best_score = -1e9
    history = []

    for epoch in range(1, args.epochs + 1):
        net.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["image"].to(device)  # [B,1,H,W]

            optimizer.zero_grad(set_to_none=True)

            y = operator.A(x)

            if device.startswith("cuda") and args.use_amp:
                with torch.cuda.amp.autocast():
                    x_hat = net(y, operator.A, operator.AT, use_amp_=True)
                    loss_l1 = F.l1_loss(x_hat, x)
                    loss_mse = F.mse_loss(x_hat, x)
                    loss = loss_l1 + 0.5 * loss_mse

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                x_hat = net(y, operator.A, operator.AT, use_amp_=False)
                loss_l1 = F.l1_loss(x_hat, x)
                loss_mse = F.mse_loss(x_hat, x)
                loss = loss_l1 + 0.5 * loss_mse

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.item())
            total_l1 += float(loss_l1.item())
            total_mse += float(loss_mse.item())
            n_batches += 1

        scheduler.step()

        train_loss = total_loss / max(n_batches, 1)
        train_l1 = total_l1 / max(n_batches, 1)
        train_mse = total_mse / max(n_batches, 1)
        train_psnr = psnr_from_mse(train_mse)

        valid_metrics = evaluate_receiver(
            net=net,
            loader=valid_loader,
            operator=operator,
            device=device,
            use_amp=args.use_amp,
        )

        logger.info(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"train_l1={train_l1:.6f} "
            f"train_psnr={train_psnr:.4f} "
            f"val_l1={valid_metrics['l1']:.6f} "
            f"val_psnr={valid_metrics['psnr']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "valid_metrics": valid_metrics,
            "script_args": vars(args),
            "operator_meta": operator.meta(),
            "model_type": "myidm_receiver",
        }

        torch.save(ckpt, save_dir / "last.pt")

        record = {
            "epoch": epoch,
            "train": {
                "loss": train_loss,
                "l1": train_l1,
                "mse": train_mse,
                "psnr": train_psnr,
            },
            "valid": valid_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # score 越大越好：优先 PSNR，兼顾 L1
        score = valid_metrics["psnr"] - 5.0 * valid_metrics["l1"]
        if score > best_score:
            best_score = score
            torch.save(ckpt, save_dir / "best.pt")
            logger.info(f"更新 best.pt, score={best_score:.6f}")

    logger.info("接收端训练完成。")


if __name__ == "__main__":
    main()