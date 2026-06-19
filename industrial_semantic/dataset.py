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
    "_load_signal",
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
    将输入信号统一整理为 [C, T] 形式。
    允许输入:
    - [C, T]
    - [T, C]
    - [T*3] 一维情况
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim == 1:
        if signal.size % expected_channels != 0:
            raise ValueError(
                f"1D 信号长度无法按 {expected_channels} 通道整除，shape={signal.shape}"
            )
        signal = signal.reshape(expected_channels, -1)

    if signal.ndim != 2:
        raise ValueError(f"信号必须为 2D，当前 shape={signal.shape}")

    # 已经是 [C, T]
    if signal.shape[0] == expected_channels:
        return signal.astype(np.float32)

    # [T, C] -> [C, T]
    if signal.shape[1] == expected_channels:
        return signal.T.astype(np.float32)

    raise ValueError(
        f"无法判断通道维度，期望某一维等于 {expected_channels}，当前 shape={signal.shape}"
    )


def _try_parse_from_dict(obj: Dict[str, Any], signal_key: str = "signal") -> np.ndarray:
    """
    尝试从 dict 中提取信号。
    支持常见键名:
    signal / data / x / vibration / xyz / waveform
    """
    candidate_keys = [
        signal_key,
        "signal",
        "data",
        "x",
        "vibration",
        "xyz",
        "waveform",
        "sensor",
    ]

    for k in candidate_keys:
        if k in obj:
            val = obj[k]
            if torch.is_tensor(val):
                val = val.detach().cpu().numpy()
            if isinstance(val, np.ndarray):
                return _ensure_ch_first(val)

    # 如果 dict 中没有显式 signal key，则找第一个 1D/2D ndarray
    for _, v in obj.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray) and v.ndim in [1, 2]:
            return _ensure_ch_first(v)

    raise ValueError("无法从 dict 中解析出有效信号")


def _load_from_npy(path: Path, signal_key: str = "signal") -> np.ndarray:
    obj = np.load(path, allow_pickle=True)

    # 普通 ndarray
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        return _ensure_ch_first(obj)

    # object ndarray，通常是一个 dict
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        try:
            item = obj.item()
            if isinstance(item, dict):
                return _try_parse_from_dict(item, signal_key=signal_key)
        except Exception:
            pass

        # 万一是 object 数组中嵌套 ndarray
        for v in obj.reshape(-1):
            if isinstance(v, np.ndarray) and v.ndim in [1, 2]:
                return _ensure_ch_first(v)

    raise ValueError(f"无法从 npy 文件中解析有效信号: {path}")


def _load_from_npz(path: Path, signal_key: str = "signal") -> np.ndarray:
    data = np.load(path, allow_pickle=True)

    if signal_key in data:
        return _ensure_ch_first(data[signal_key])

    candidate_keys = ["signal", "data", "x", "vibration", "xyz", "waveform"]
    for k in candidate_keys:
        if k in data:
            return _ensure_ch_first(data[k])

    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim in [1, 2]:
            return _ensure_ch_first(arr)

    raise ValueError(f"无法从 npz 文件中解析有效信号: {path}")


def _load_from_pt(path: Path, signal_key: str = "signal") -> np.ndarray:
    obj = torch.load(path, map_location="cpu")

    if torch.is_tensor(obj):
        return _ensure_ch_first(obj.detach().cpu().numpy())

    if isinstance(obj, dict):
        return _try_parse_from_dict(obj, signal_key=signal_key)

    if isinstance(obj, np.ndarray):
        return _ensure_ch_first(obj)

    raise ValueError(f"无法从 pt/pth 文件中解析有效信号: {path}")


def _load_from_text(path: Path) -> np.ndarray:
    """
    支持 csv / txt:
    - [T, 3]
    - [3, T]
    """
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        arr = np.loadtxt(path, dtype=np.float32)
    return _ensure_ch_first(arr)


def _load_from_mat(path: Path, signal_key: str = "signal") -> np.ndarray:
    if not _HAS_SCIPY:
        raise ImportError("当前环境未安装 scipy，无法读取 .mat 文件")

    obj = sio.loadmat(path)
    candidate_keys = [signal_key, "signal", "data", "x", "vibration", "xyz", "waveform"]

    for k in candidate_keys:
        if k in obj:
            arr = obj[k]
            if isinstance(arr, np.ndarray) and arr.ndim in [1, 2]:
                return _ensure_ch_first(arr)

    for k, v in obj.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim in [1, 2]:
            return _ensure_ch_first(v)

    raise ValueError(f"无法从 mat 文件中解析有效信号: {path}")


def _load_signal(path: Path, signal_key: str = "signal") -> np.ndarray:
    """
    读取原始信号，并统一返回 [C, T] float32
    支持：
    - .npy
    - .npz
    - .pt / .pth
    - .csv / .txt
    - .mat (可选，需要 scipy)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"信号文件不存在: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return _load_from_npy(path, signal_key=signal_key)

    if suffix == ".npz":
        return _load_from_npz(path, signal_key=signal_key)

    if suffix in [".pt", ".pth"]:
        return _load_from_pt(path, signal_key=signal_key)

    if suffix in [".csv", ".txt"]:
        return _load_from_text(path)

    if suffix == ".mat":
        return _load_from_mat(path, signal_key=signal_key)

    raise ValueError(f"暂不支持的数据格式: {suffix}, path={path}")


def _slice_signal(
    signal: np.ndarray,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    window_length: Optional[int] = None,
) -> np.ndarray:
    """
    从长时序信号中切取窗口。
    规则：
    - 若 start/end 都给出，则直接切片
    - 若只给 start + window_length，则 end = start + window_length
    - 若只给 end + window_length，则 start = end - window_length
    - 若都不给，则返回原信号
    """
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
    """
    将信号对齐到 target_length:
    - 长于 target_length -> 截断
    - 短于 target_length -> 右侧补零
    """
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
    工业刀具磨损数据集加载器

    支持两种样本组织方式：

    方式 A：每个样本文件本身就是一个 1024 窗口
    CSV:
        path,mapped_class,contact_label,wear_label,tool_id,run_id,window_id,...

    方式 B：path 指向整段长时序，CSV 中额外给出 start_idx/end_idx
    CSV:
        path,start_idx,end_idx,mapped_class,contact_label,wear_label,...

    标签定义：
    - mapped_class:
        0 = idle
        1 = initial
        2 = steady
        3 = accelerating

    - contact_label:
        0 = idle
        1 = mixed
        2 = cutting

    - wear_label:
        -1 = NA（idle）
         0 = initial
         1 = steady
         2 = accelerating
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

        # contact_label 默认根据 mapped_class 推断
        if "contact_label" in row and row["contact_label"] != "":
            contact_label = _safe_int(row["contact_label"], default=0)
        else:
            contact_label = 0 if mapped_class == 0 else 2

        # wear_label 默认根据 mapped_class 推断
        if "wear_label" in row and row["wear_label"] != "":
            wear_label = _safe_int(row["wear_label"], default=-1)
        else:
            wear_label = -1 if mapped_class == 0 else mapped_class - 1

        return mapped_class, contact_label, wear_label

    def _load_one_sample(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        path = self._resolve_path(row["path"])
        signal = _load_signal(path, signal_key=self.signal_key).astype(np.float32)

        # 支持从长时序文件中根据索引切片
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

        # 最后统一 pad / trim 到目标长度
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

    用法示例:
        buf = SlidingWindowBuffer(window_size=1024, hop_size=1024, channels=3)
        windows = buf.push(chunk)  # chunk shape: [C, T] or [T, C]

    返回:
        List[np.ndarray], 每个窗口 shape = [C, window_size]
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
        """
        将缓冲区剩余内容输出：
        - pad=False: 不输出不足 window_size 的尾部
        - pad=True: 右侧补零输出一个尾窗口
        """
        if self.buffer.shape[1] == 0:
            return []

        if self.buffer.shape[1] < self.window_size:
            if not pad:
                return []
            win = _pad_or_trim_signal(self.buffer, target_length=self.window_size, pad_mode="constant")
            self.buffer = np.zeros((self.channels, 0), dtype=np.float32)
            return [win]

        # 理论上不会到这里，因为 >=window_size 的内容应该已被 push 取出
        win = self.buffer[:, :self.window_size].copy()
        self.buffer = self.buffer[:, self.hop_size:]
        return [win]