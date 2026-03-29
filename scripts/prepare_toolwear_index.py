import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import yaml

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_semantic.dataset import _load_signal


SUPPORTED_SUFFIXES = {".npy", ".npz", ".pt", ".pth"}


def infer_mapped_class(path: Path):
    s = str(path).lower()

    if "idle" in s or "idling" in s:
        return 0
    if "initial" in s or "initwear" in s or "initialwear" in s:
        return 1
    if "steady" in s or "stable" in s or "steadywear" in s:
        return 2
    if "accelerating" in s or "accel" in s or "accelerate" in s:
        return 3

    return None


def mapped_to_contact_wear(mapped_class: int):
    if mapped_class == 0:
        return 0, -1
    return 2, mapped_class - 1


def infer_tool_id(path: Path):
    s = str(path)
    m = re.search(r"tool[_\-]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"T{m.group(1)}"
    return path.parent.name or "T001"


def infer_run_id(path: Path):
    s = str(path)
    m = re.search(r"run[_\-]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"R{m.group(1)}"
    return path.parent.parent.name if path.parent.parent != path.parent else "R001"


def scan_files(root_dir: Path):
    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)
    files = sorted(files)
    return files


def build_rows(root_dir: Path):
    rows = []
    files = scan_files(root_dir)
    if len(files) == 0:
        raise RuntimeError(f"在 {root_dir} 下没有扫描到支持的数据文件: {SUPPORTED_SUFFIXES}")

    for idx, p in enumerate(files):
        mapped_class = infer_mapped_class(p)
        if mapped_class is None:
            continue

        contact_label, wear_label = mapped_to_contact_wear(mapped_class)

        rows.append({
            "path": str(p.relative_to(root_dir)),
            "mapped_class": mapped_class,
            "contact_label": contact_label,
            "wear_label": wear_label,
            "tool_id": infer_tool_id(p),
            "run_id": infer_run_id(p),
            "window_id": idx,
            "timestamp_start": float(idx),
            "timestamp_end": float(idx + 1),
        })

    if len(rows) == 0:
        raise RuntimeError("没有成功推断出任何标签，请检查目录命名是否含 idle/initial/steady/accelerating 等关键词")
    return rows


def split_rows(rows, train_ratio=0.7, valid_ratio=0.2, seed=42, group_by="tool"):
    rng = random.Random(seed)

    groups = {}
    for row in rows:
        if group_by == "tool":
            key = row["tool_id"]
        elif group_by == "run":
            key = row["run_id"]
        else:
            key = row["path"]
        groups.setdefault(key, []).append(row)

    keys = list(groups.keys())
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_keys = set(keys[:n_train])
    valid_keys = set(keys[n_train:n_train + n_valid])
    test_keys = set(keys[n_train + n_valid:])

    train_rows, valid_rows, test_rows = [], [], []
    for k, items in groups.items():
        if k in train_keys:
            train_rows.extend(items)
        elif k in valid_keys:
            valid_rows.extend(items)
        else:
            test_rows.extend(items)

    return train_rows, valid_rows, test_rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path", "mapped_class", "contact_label", "wear_label",
        "tool_id", "run_id", "window_id", "timestamp_start", "timestamp_end"
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_global_stats(root_dir: Path, rows):
    sum_c = None
    sumsq_c = None
    count = 0

    for row in rows:
        signal = _load_signal(root_dir / row["path"]).astype(np.float32)  # [C,T]
        if sum_c is None:
            sum_c = signal.sum(axis=1)
            sumsq_c = (signal ** 2).sum(axis=1)
        else:
            sum_c += signal.sum(axis=1)
            sumsq_c += (signal ** 2).sum(axis=1)
        count += signal.shape[1]

    mean = sum_c / max(count, 1)
    var = sumsq_c / max(count, 1) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))

    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="刀具磨损数据根目录")
    parser.add_argument("--out_dir", type=str, default="./data/toolwear")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_by", type=str, default="tool", choices=["tool", "run", "none"])
    parser.add_argument("--compute_norm", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(root_dir)
    train_rows, valid_rows, test_rows = split_rows(
        rows,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        group_by=args.group_by,
    )

    write_csv(train_rows, out_dir / "train.csv")
    write_csv(valid_rows, out_dir / "valid.csv")
    write_csv(test_rows, out_dir / "test.csv")

    print(f"train: {len(train_rows)}, valid: {len(valid_rows)}, test: {len(test_rows)}")

    if args.compute_norm:
        mean, std = compute_global_stats(root_dir, train_rows)
        stats = {
            "normalization": {
                "enabled": True,
                "id": "global-zscore-v1",
                "mean": mean,
                "std": std,
            }
        }
        with open(out_dir / "normalization_stats.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(stats, f, sort_keys=False, allow_unicode=True)
        print(f"已保存归一化统计到: {out_dir / 'normalization_stats.yaml'}")


if __name__ == "__main__":
    main()