import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import scipy.io as sio
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


__all__ = [
    "_safe_int",
    "_safe_float",
    "_ensure_ch_first",
    "_ensure_image_ch_first",
    "_load_signal",
    "_load_any_array",
    "_infer_loaded_modality",
    "_split_loaded_samples",
    "_slice_signal",
    "_pad_or_trim_signal",
    "ToolWearDataset",
    "SlidingWindowBuffer",
]


def _safe_int(v, default=0):
    try:
        if v is None or v == "":
            return default
        return int(v)
    except Exception:
        return default


def _safe_float(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _ensure_ch_first(signal: np.ndarray, expected_channels: int = 3) -> np.ndarray:
    """
    将单个时序信号整理为 [C, T]
    支持:
    - [C, T]
    - [T, C]
    - [T*C] 1D
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim == 1:
        if signal.size % expected_channels != 0:
            raise ValueError(
                f"1D 信号长度无法按 {expected_channels} 通道整除，shape={signal.shape}"
            )
        signal = signal.reshape(expected_channels, -1)

    if signal.ndim != 2:
        raise ValueError(f"_ensure_ch_first 只支持 2D 时序信号，当前 shape={signal.shape}")

    if signal.shape[0] == expected_channels:
        return signal.astype(np.float32)

    if signal.shape[1] == expected_channels:
        return signal.T.astype(np.float32)

    raise ValueError(
        f"无法判断时序信号通道维度，期望某一维等于 {expected_channels}，当前 shape={signal.shape}"
    )


def _ensure_image_ch_first(image: np.ndarray, expected_channels: int = 3, allow_gray_to_rgb: bool = True) -> np.ndarray:
    """
    将单张图像整理为 [C, H, W]
    支持:
    - [C, H, W]
    - [H, W, C]
    - [H, W]      -> 若 allow_gray_to_rgb=True 则扩为 3 通道
    """
    image = np.asarray(image)

    if image.ndim == 2:
        image = image[None, :, :]

    if image.ndim != 3:
        raise ValueError(f"_ensure_image_ch_first 只支持 2D/3D 图像，当前 shape={image.shape}")

    # CHW
    if image.shape[0] in [1, expected_channels] and image.shape[1] >= 4 and image.shape[2] >= 4:
        out = image
    # HWC
    elif image.shape[-1] in [1, expected_channels] and image.shape[0] >= 4 and image.shape[1] >= 4:
        out = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"无法判断图像通道维度，当前 shape={image.shape}")

    if out.shape[0] == 1 and expected_channels == 3 and allow_gray_to_rgb:
        out = np.repeat(out, 3, axis=0)

    return out.astype(np.float32)


def _extract_array_from_dict(obj: Dict[str, Any], preferred_keys: Optional[List[str]] = None):
    preferred_keys = preferred_keys or [
        "signal", "data", "x", "vibration", "xyz", "waveform",
        "image", "images", "rgb", "arr", "array"
    ]

    for k in preferred_keys:
        if k in obj:
            v = obj[k]
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray):
                return v

    for _, v in obj.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            return v

    raise ValueError("无法从 dict 中提取 ndarray")


def _load_any_array_from_npy(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.dtype != object:
        return np.asarray(obj)

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        # 常见：object ndarray 里包了 dict
        try:
            item = obj.item()
            if isinstance(item, dict):
                return np.asarray(_extract_array_from_dict(item))
        except Exception:
            pass

        # 再尝试枚举其中元素
        for v in obj.reshape(-1):
            if isinstance(v, np.ndarray):
                return np.asarray(v)

    raise ValueError(f"无法从 npy 文件解析有效数组: {path}")


def _load_any_array_from_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)

    candidate_keys = [
        "signal", "data", "x", "vibration", "xyz", "waveform",
        "image", "images", "rgb", "arr", "array"
    ]
    for k in candidate_keys:
        if k in data:
            return np.asarray(data[k])

    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray):
            return np.asarray(arr)

    raise ValueError(f"无法从 npz 文件解析有效数组: {path}")


def _load_any_array_from_pt(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")

    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()

    if isinstance(obj, dict):
        return np.asarray(_extract_array_from_dict(obj))

    if isinstance(obj, np.ndarray):
        return np.asarray(obj)

    raise ValueError(f"无法从 pt/pth 文件解析有效数组: {path}")


def _load_any_array_from_text(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.loadtxt(path)
    return np.asarray(arr)


def _load_any_array_from_mat(path: Path) -> np.ndarray:
    if not _HAS_SCIPY:
        raise ImportError("当前环境未安装 scipy，无法读取 .mat 文件")

    obj = sio.loadmat(path)
    candidate_keys = [
        "signal", "data", "x", "vibration", "xyz", "waveform",
        "image", "images", "rgb", "arr", "array"
    ]

    for k in candidate_keys:
        if k in obj and isinstance(obj[k], np.ndarray):
            return np.asarray(obj[k])

    for k, v in obj.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            return np.asarray(v)

    raise ValueError(f"无法从 mat 文件解析有效数组: {path}")


def _load_any_array(path: Path) -> np.ndarray:
    """
    通用数组加载器：
    不强行假设它是 [C,T]，而是原样读出 ndarray
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return _load_any_array_from_npy(path)

    if suffix == ".npz":
        return _load_any_array_from_npz(path)

    if suffix in [".pt", ".pth"]:
        return _load_any_array_from_pt(path)

    if suffix in [".csv", ".txt"]:
        return _load_any_array_from_text(path)

    if suffix == ".mat":
        return _load_any_array_from_mat(path)

    raise ValueError(f"暂不支持的文件格式: {suffix}, path={path}")


def _infer_loaded_modality(arr: np.ndarray) -> str:
    """
    粗略判断加载出来的数据是：
    - signal_single
    - signal_batch
    - image_single
    - image_batch
    - label_vector
    - unsupported
    """
    arr = np.asarray(arr)

    if arr.ndim == 0:
        return "unsupported"

    if arr.ndim == 1:
        return "label_vector"

    if arr.ndim == 2:
        # [C,T] or [T,C]
        if 3 in arr.shape:
            return "signal_single"
        # 灰度图 [H,W]
        if arr.shape[0] >= 8 and arr.shape[1] >= 8:
            return "image_single"
        return "unsupported"

    if arr.ndim == 3:
        # 单张图像 [C,H,W] or [H,W,C]
        if ((arr.shape[0] in [1, 3]) and arr.shape[1] >= 8 and arr.shape[2] >= 8) or \
           ((arr.shape[-1] in [1, 3]) and arr.shape[0] >= 8 and arr.shape[1] >= 8):
            return "image_single"

        # 批量时序 [N,C,T] or [N,T,C]
        if arr.shape[1] in [1, 3] and arr.shape[2] >= 16:
            return "signal_batch"
        if arr.shape[2] in [1, 3] and arr.shape[1] >= 16:
            return "signal_batch"

        return "unsupported"

    if arr.ndim == 4:
        # 批量图像 [N,C,H,W] or [N,H,W,C]
        if arr.shape[1] in [1, 3] and arr.shape[2] >= 8 and arr.shape[3] >= 8:
            return "image_batch"
        if arr.shape[-1] in [1, 3] and arr.shape[1] >= 8 and arr.shape[2] >= 8:
            return "image_batch"

        return "unsupported"

    return "unsupported"


def _split_loaded_samples(arr: np.ndarray, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    将一个文件中加载出的 ndarray 拆成若干样本。

    返回:
    {
        "kind": "...",
        "modality": "signal1d" / "image2d" / None,
        "samples": [np.ndarray, ...]
    }
    """
    arr = np.asarray(arr)
    kind = _infer_loaded_modality(arr)

    samples = []
    modality = None

    if kind == "signal_single":
        samples = [_ensure_ch_first(arr)]
        modality = "signal1d"

    elif kind == "signal_batch":
        n = arr.shape[0]
        if max_samples is not None:
            n = min(n, int(max_samples))
        for i in range(n):
            samples.append(_ensure_ch_first(arr[i]))
        modality = "signal1d"

    elif kind == "image_single":
        samples = [_ensure_image_ch_first(arr)]
        modality = "image2d"

    elif kind == "image_batch":
        n = arr.shape[0]
        if max_samples is not None:
            n = min(n, int(max_samples))
        for i in range(n):
            samples.append(_ensure_image_ch_first(arr[i]))
        modality = "image2d"

    return {
        "kind": kind,
        "modality": modality,
        "samples": samples,
    }


def _load_signal(path: Path, signal_key: str = "signal") -> np.ndarray:
    """
    兼容旧代码的“单个时序信号”加载器。
    如果文件实际是图像 batch，会给出更清晰的报错。
    """
    arr = _load_any_array(path)
    kind = _infer_loaded_modality(arr)

    if kind == "signal_single":
        return _ensure_ch_first(arr)

    if kind == "signal_batch":
        raise ValueError(
            f"_load_signal 只适用于单个时序样本，但当前文件更像批量时序数据，shape={arr.shape}。"
            f"请改用 _load_any_array + _split_loaded_samples。"
        )

    if kind in ["image_single", "image_batch"]:
        raise ValueError(
            f"_load_signal 只适用于单个 [C,T] 时序信号，但当前文件更像图像数据，shape={arr.shape}。"
            f"例如你的 ToolWear_RGB 数据通常是 [N,3,32,32]。"
            f"请改用 _load_any_array + _split_loaded_samples。"
        )

    raise ValueError(f"无法将文件解析为单个时序信号，shape={arr.shape}")


def _slice_signal(
    signal: np.ndarray,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    window_length: Optional[int] = None,
) -> np.ndarray:
    signal = _ensure_ch_first(signal)
    total_len = signal.shape[1]

    if start_idx is None and end_idx is None:
        return signal

    if start_idx is not None:
        start_idx = max(0, int(start_idx))
    if end_idx is not None:
        end_idx = min(total_len, int(end_idx))

    if start_idx is not None and end_idx is None:
        if window_length is None:
            raise ValueError("只给 start_idx 时必须同时提供 window_length")
        end_idx = min(total_len, start_idx + int(window_length))

    if start_idx is None and end_idx is not None:
        if window_length is None:
            raise ValueError("只给 end_idx 时必须同时提供 window_length")
        start_idx = max(0, end_idx - int(window_length))

    if start_idx is None or end_idx is None:
        raise ValueError("切片参数不完整")

    if end_idx <= start_idx:
        raise ValueError(f"非法切片范围: start_idx={start_idx}, end_idx={end_idx}")

    return signal[:, start_idx:end_idx].astype(np.float32)


def _pad_or_trim_signal(
    signal: np.ndarray,
    target_length: Optional[int] = None,
    pad_mode: str = "constant",
) -> np.ndarray:
    signal = _ensure_ch_first(signal)
    if target_length is None:
        return signal.astype(np.float32)

    target_length = int(target_length)
    cur_len = signal.shape[1]

    if cur_len == target_length:
        return signal.astype(np.float32)

    if cur_len > target_length:
        return signal[:, :target_length].astype(np.float32)

    pad_len = target_length - cur_len
    if pad_mode == "constant":
        pad_width = ((0, 0), (0, pad_len))
        return np.pad(signal, pad_width=pad_width, mode="constant").astype(np.float32)

    if pad_mode in ["edge", "reflect"]:
        pad_width = ((0, 0), (0, pad_len))
        return np.pad(signal, pad_width=pad_width, mode=pad_mode).astype(np.float32)

    raise ValueError(f"不支持的 pad_mode: {pad_mode}")


class ToolWearDataset(Dataset):
    """
    当前这个 Dataset 仍然主要服务于论文方案中的时序振动数据：
    - path 指向单窗信号文件，或长时序 + start/end 索引
    - 输出 [C,T]

    对于 ToolWear_RGB 这种 [N,3,32,32] 的 batched image 文件，
    推荐走 run_semantic_demo.py 中的通用数组加载流程，而不是这个类。
    """

    def __init__(
        self,
        index_csv: str,
        root_dir: Optional[str] = None,
        signal_key: str = "signal",
        normalization: Optional[Dict[str, Any]] = None,
        window_length: Optional[int] = None,
        pad_mode: str = "constant",
        cache_in_memory: bool = False,
        return_raw: bool = True,
    ):
        self.index_csv = Path(index_csv)
        self.root_dir = Path(root_dir) if root_dir else None
        self.signal_key = signal_key
        self.window_length = window_length
        self.pad_mode = pad_mode
        self.cache_in_memory = cache_in_memory
        self.return_raw = return_raw

        self.rows: List[Dict[str, Any]] = self._read_index(self.index_csv)
        self._cache: Dict[int, Dict[str, Any]] = {}

        self.normalization = normalization or {}
        self.norm_enabled = bool(self.normalization.get("enabled", True))
        self.norm_id = self.normalization.get("id", "global-zscore-v1")

        mean = self.normalization.get("mean", [0.0, 0.0, 0.0])
        std = self.normalization.get("std", [1.0, 1.0, 1.0])

        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1)
        self.std = np.clip(self.std, 1e-8, None)

    @staticmethod
    def _read_index(index_csv: Path) -> List[Dict[str, Any]]:
        if not index_csv.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_csv}")

        rows = []
        with open(index_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"索引文件为空: {index_csv}")
        return rows

    def __len__(self):
        return len(self.rows)

    def _resolve_path(self, relative_or_absolute: str) -> Path:
        p = Path(relative_or_absolute)
        if p.is_absolute():
            return p
        if self.root_dir is not None:
            return self.root_dir / p
        return p

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        if not self.norm_enabled:
            return signal.astype(np.float32)

        if self.mean.shape[0] not in [1, signal.shape[0]]:
            raise ValueError(
                f"mean 通道数与信号不匹配: mean={self.mean.shape}, signal={signal.shape}"
            )
        if self.std.shape[0] not in [1, signal.shape[0]]:
            raise ValueError(
                f"std 通道数与信号不匹配: std={self.std.shape}, signal={signal.shape}"
            )

        return ((signal - self.mean) / self.std).astype(np.float32)

    def label_histogram(
        self,
        field: str,
        num_classes: Optional[int] = None,
        ignore_value: Optional[int] = None,
    ) -> np.ndarray:
        vals = []
        for row in self.rows:
            if field not in row:
                continue
            v_raw = row.get(field, "")
            if v_raw == "":
                continue
            v = _safe_int(v_raw, default=None)
            if v is None:
                continue
            if ignore_value is not None and v == ignore_value:
                continue
            vals.append(v)

        if len(vals) == 0:
            if num_classes is None:
                return np.zeros((0,), dtype=np.int64)
            return np.zeros((num_classes,), dtype=np.int64)

        if num_classes is None:
            num_classes = max(vals) + 1

        hist = np.zeros((num_classes,), dtype=np.int64)
        for v in vals:
            if 0 <= v < num_classes:
                hist[v] += 1
        return hist

    def _parse_labels(self, row: Dict[str, Any]) -> Tuple[int, int, int]:
        mapped_class = _safe_int(row.get("mapped_class", row.get("label", 0)), default=0)

        if "contact_label" in row and row["contact_label"] != "":
            contact_label = _safe_int(row["contact_label"], default=0)
        else:
            contact_label = 0 if mapped_class == 0 else 2

        if "wear_label" in row and row["wear_label"] != "":
            wear_label = _safe_int(row["wear_label"], default=-1)
        else:
            wear_label = -1 if mapped_class == 0 else mapped_class - 1

        return mapped_class, contact_label, wear_label

    def _load_one_sample(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        path = self._resolve_path(row["path"])
        signal = _load_signal(path, signal_key=self.signal_key).astype(np.float32)

        has_start = "start_idx" in row and row["start_idx"] != ""
        has_end = "end_idx" in row and row["end_idx"] != ""

        start_idx = _safe_int(row["start_idx"], default=None) if has_start else None
        end_idx = _safe_int(row["end_idx"], default=None) if has_end else None

        if has_start or has_end:
            signal = _slice_signal(
                signal,
                start_idx=start_idx,
                end_idx=end_idx,
                window_length=self.window_length,
            )

        signal = _pad_or_trim_signal(
            signal,
            target_length=self.window_length,
            pad_mode=self.pad_mode,
        )

        mapped_class, contact_label, wear_label = self._parse_labels(row)
        signal_norm = self._normalize(signal)

        meta = {
            "path": str(path),
            "tool_id": row.get("tool_id", "T001"),
            "run_id": row.get("run_id", "R001"),
            "window_id": _safe_int(row.get("window_id", idx), default=idx),
            "timestamp_start": _safe_float(row.get("timestamp_start", 0.0), default=0.0),
            "timestamp_end": _safe_float(row.get("timestamp_end", 0.0), default=0.0),
            "normalization_id": self.norm_id,
        }

        item = {
            "signal": torch.from_numpy(signal_norm).float(),
            "contact_label": torch.tensor(contact_label, dtype=torch.long),
            "wear_label": torch.tensor(wear_label, dtype=torch.long),
            "mapped_class": torch.tensor(mapped_class, dtype=torch.long),
            "meta": meta,
        }

        if self.return_raw:
            item["signal_raw"] = torch.from_numpy(signal).float()

        return item

    def __getitem__(self, idx: int):
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        item = self._load_one_sample(idx)

        if self.cache_in_memory:
            self._cache[idx] = item

        return item


class SlidingWindowBuffer:
    """
    实时流滑窗缓冲器
    输入 chunk shape: [C,T] 或 [T,C]
    输出 windows: List[[C, window_size]]
    """

    def __init__(self, window_size: int = 1024, hop_size: int = 1024, channels: int = 3):
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.channels = int(channels)
        self.buffer = np.zeros((self.channels, 0), dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros((self.channels, 0), dtype=np.float32)

    def __len__(self):
        return self.buffer.shape[1]

    def push(self, chunk: np.ndarray) -> List[np.ndarray]:
        chunk = _ensure_ch_first(chunk, expected_channels=self.channels).astype(np.float32)

        if chunk.shape[0] != self.channels:
            raise ValueError(
                f"输入 chunk 通道数错误，期望 {self.channels}，实际 {chunk.shape[0]}"
            )

        self.buffer = np.concatenate([self.buffer, chunk], axis=1)

        windows = []
        while self.buffer.shape[1] >= self.window_size:
            win = self.buffer[:, :self.window_size].copy()
            windows.append(win)
            self.buffer = self.buffer[:, self.hop_size:]

        return windows

    def flush_tail(self, pad: bool = False) -> List[np.ndarray]:
        if self.buffer.shape[1] == 0:
            return []

        if self.buffer.shape[1] < self.window_size:
            if not pad:
                return []
            win = _pad_or_trim_signal(self.buffer, target_length=self.window_size, pad_mode="constant")
            self.buffer = np.zeros((self.channels, 0), dtype=np.float32)
            return [win]

        win = self.buffer[:, :self.window_size].copy()
        self.buffer = self.buffer[:, self.hop_size:]
        return [win]