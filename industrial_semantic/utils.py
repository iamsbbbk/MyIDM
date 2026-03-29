import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


__all__ = [
    "set_seed",
    "get_logger",
    "softmax_np",
    "sigmoid_np",
    "safe_div",
    "ensure_signal_ch_first",
    "ensure_batch_bct",
    "normalized_entropy",
    "compute_signal_anchors",
    "compute_subwindow_rms",
    "compute_class_weights_from_hist",
    "confusion_matrix_np",
    "macro_classification_metrics",
    "false_idle_metrics",
    "recursive_to_numpy",
    "detach_to_cpu",
    "move_to_device",
    "tensor_item",
    "count_parameters",
    "save_json",
    "load_json",
    "estimate_tensor_nbytes",
    "estimate_object_nbytes",
    "format_metrics",
    "AverageMeter",
    "LatencyMeter",
    "Timer",
]


# =========================================================
# 基础随机种子与日志
# =========================================================
def set_seed(seed: int = 42, deterministic: bool = True):
    """
    固定随机种子，便于实验复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(
    name: str = "industrial_semantic",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    获取统一 logger，避免重复添加 handler。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger


# =========================================================
# 数值与张量基础工具
# =========================================================
def safe_div(numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], eps: float = 1e-12):
    return numerator / (denominator + eps)


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(logits)
    return exps / (np.sum(exps, axis=axis, keepdims=True) + 1e-12)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def tensor_item(x: Any):
    """
    安全地把 0-dim tensor / numpy scalar 转为 Python 标量。
    """
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().numpy()
    if isinstance(x, np.generic):
        return x.item()
    return x


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# =========================================================
# 信号形状整理工具
# =========================================================
def ensure_signal_ch_first(signal: Union[np.ndarray, torch.Tensor], expected_channels: int = 3) -> np.ndarray:
    """
    将单个信号整理成 [C, T] 格式。

    支持输入：
    - [C, T]
    - [T, C]
    - [T * C] (1D)
    """
    if torch.is_tensor(signal):
        signal = signal.detach().cpu().numpy()

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim == 1:
        if signal.size % expected_channels != 0:
            raise ValueError(
                f"1D 信号长度不能被通道数整除: len={signal.size}, channels={expected_channels}"
            )
        signal = signal.reshape(expected_channels, -1)

    if signal.ndim != 2:
        raise ValueError(f"单个信号必须为 2D，当前 shape={signal.shape}")

    if signal.shape[0] == expected_channels:
        return signal.astype(np.float32)

    if signal.shape[1] == expected_channels:
        return signal.T.astype(np.float32)

    raise ValueError(
        f"无法推断通道维度，期待某一维等于 {expected_channels}，当前 shape={signal.shape}"
    )


def ensure_batch_bct(x: Union[np.ndarray, torch.Tensor], channels: int = 3) -> torch.Tensor:
    """
    将输入整理成 [B, C, T] torch.Tensor

    支持：
    - [C, T]
    - [T, C]
    - [B, C, T]
    - [B, T, C]
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if not torch.is_tensor(x):
        raise TypeError(f"输入必须是 np.ndarray 或 torch.Tensor，当前为 {type(x)}")

    x = x.float()

    if x.ndim == 2:
        arr = ensure_signal_ch_first(x.detach().cpu().numpy(), expected_channels=channels)
        return torch.from_numpy(arr).unsqueeze(0).float()

    if x.ndim != 3:
        raise ValueError(f"批信号必须为 3D，当前 shape={tuple(x.shape)}")

    if x.shape[1] == channels:
        return x.float()

    if x.shape[2] == channels:
        return x.transpose(1, 2).contiguous().float()

    raise ValueError(f"无法推断 batch 通道维度，当前 shape={tuple(x.shape)}")


# =========================================================
# 信息熵与不确定性
# =========================================================
def normalized_entropy(probs: np.ndarray) -> float:
    """
    归一化熵，范围 [0, 1]。
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.size <= 1:
        return 0.0
    probs = np.clip(probs, 1e-12, None)
    probs = probs / np.sum(probs)
    ent = -np.sum(probs * np.log(probs))
    return float(ent / math.log(probs.size))


# =========================================================
# 工业信号锚点特征
# =========================================================
def compute_subwindow_rms(signal: np.ndarray, splits: int = 4, expected_channels: int = 3) -> List[float]:
    """
    计算若干子窗口的 RMS，用于 occupancy / mixed 状态估计。
    """
    signal = ensure_signal_ch_first(signal, expected_channels=expected_channels).astype(np.float32)
    total_len = signal.shape[1]
    splits = max(1, int(splits))
    split_len = max(1, total_len // splits)

    rms_list = []
    for i in range(splits):
        start = i * split_len
        end = total_len if i == splits - 1 else min(total_len, (i + 1) * split_len)
        seg = signal[:, start:end]
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        rms_list.append(rms)
    return rms_list


def compute_signal_anchors(signal: np.ndarray, sampling_rate: int = 10240, expected_channels: int = 3) -> Dict[str, float]:
    """
    计算一组轻量级工业语义锚点特征。
    这些特征适合：
    - M1 anchor-only 模式
    - 云端展示
    - MyIDM 辅助解释
    - 通信压缩时的 side-channel

    返回全局聚合后的标量特征。
    """
    signal = ensure_signal_ch_first(signal, expected_channels=expected_channels).astype(np.float32)
    c, t = signal.shape

    # 时域统计
    mean_axis = np.mean(signal, axis=1)
    mean_abs_axis = np.mean(np.abs(signal), axis=1)
    rms_axis = np.sqrt(np.mean(signal ** 2, axis=1) + 1e-12)
    var_axis = np.var(signal, axis=1)
    std_axis = np.sqrt(var_axis + 1e-12)
    peak_axis = np.max(np.abs(signal), axis=1)
    ptp_axis = np.ptp(signal, axis=1)

    centered = signal - mean_axis[:, None]
    skew_axis = np.mean((centered / (std_axis[:, None] + 1e-12)) ** 3, axis=1)
    kurt_axis = np.mean((centered / (std_axis[:, None] + 1e-12)) ** 4, axis=1)
    crest_axis = peak_axis / (rms_axis + 1e-12)
    impulse_axis = peak_axis / (mean_abs_axis + 1e-12)
    shape_axis = rms_axis / (mean_abs_axis + 1e-12)

    # 频域统计
    if t >= 2:
        spec = np.abs(np.fft.rfft(signal, axis=1)) ** 2  # 功率谱
        freqs = np.fft.rfftfreq(t, d=1.0 / max(sampling_rate, 1))
    else:
        spec = np.zeros((c, 1), dtype=np.float32)
        freqs = np.zeros((1,), dtype=np.float32)

    total_energy_axis = np.sum(spec, axis=1) + 1e-12

    low_mask = (freqs >= 0) & (freqs < 1000)
    mid_mask = (freqs >= 1000) & (freqs < 3000)
    high_mask = freqs >= 3000

    low_energy_axis = np.sum(spec[:, low_mask], axis=1) if np.any(low_mask) else np.zeros((c,), dtype=np.float32)
    mid_energy_axis = np.sum(spec[:, mid_mask], axis=1) if np.any(mid_mask) else np.zeros((c,), dtype=np.float32)
    high_energy_axis = np.sum(spec[:, high_mask], axis=1) if np.any(high_mask) else np.zeros((c,), dtype=np.float32)

    dominant_idx = np.argmax(spec, axis=1)
    dominant_freq_axis = freqs[dominant_idx] if len(freqs) > 0 else np.zeros((c,), dtype=np.float32)
    spectral_centroid_axis = np.sum(spec * freqs[None, :], axis=1) / total_energy_axis

    low_ratio_axis = low_energy_axis / total_energy_axis
    mid_ratio_axis = mid_energy_axis / total_energy_axis
    high_ratio_axis = high_energy_axis / total_energy_axis

    # 子窗波动
    sub_rms = compute_subwindow_rms(signal, splits=4, expected_channels=expected_channels)
    sub_rms = np.asarray(sub_rms, dtype=np.float32)

    anchors = {
        # 全局时域
        "global_mean": float(np.mean(mean_axis)),
        "global_mean_abs": float(np.mean(mean_abs_axis)),
        "global_rms": float(np.mean(rms_axis)),
        "global_std": float(np.mean(std_axis)),
        "variance": float(np.mean(var_axis)),
        "global_peak": float(np.mean(peak_axis)),
        "peak_to_peak": float(np.mean(ptp_axis)),
        "skewness": float(np.mean(skew_axis)),
        "kurtosis": float(np.mean(kurt_axis)),
        "crest_factor": float(np.mean(crest_axis)),
        "impulse_factor": float(np.mean(impulse_axis)),
        "shape_factor": float(np.mean(shape_axis)),

        # 全局频域
        "dominant_freq": float(np.mean(dominant_freq_axis)),
        "spectral_centroid": float(np.mean(spectral_centroid_axis)),
        "low_band_energy": float(np.mean(low_energy_axis)),
        "mid_band_energy": float(np.mean(mid_energy_axis)),
        "high_band_energy": float(np.mean(high_energy_axis)),
        "low_band_ratio": float(np.mean(low_ratio_axis)),
        "mid_band_ratio": float(np.mean(mid_ratio_axis)),
        "high_band_ratio": float(np.mean(high_ratio_axis)),

        # 子窗稳定性
        "subwindow_rms_mean": float(np.mean(sub_rms)),
        "subwindow_rms_std": float(np.std(sub_rms)),
        "subwindow_rms_max": float(np.max(sub_rms)),
        "subwindow_rms_min": float(np.min(sub_rms)),
    }

    # 为后续兼容性保留轴均值摘要
    for i in range(min(c, 3)):
        axis_name = ["x", "y", "z"][i] if i < 3 else f"ch{i}"
        anchors[f"rms_{axis_name}"] = float(rms_axis[i])
        anchors[f"peak_{axis_name}"] = float(peak_axis[i])
        anchors[f"std_{axis_name}"] = float(std_axis[i])

    return anchors


# =========================================================
# 分类指标与工业关键指标
# =========================================================
def compute_class_weights_from_hist(hist: np.ndarray, min_count: float = 1.0, normalize: bool = True) -> torch.Tensor:
    """
    根据类别计数直方图生成类别权重。
    """
    hist = np.asarray(hist, dtype=np.float32).reshape(-1)
    if hist.size == 0:
        return torch.tensor([], dtype=torch.float32)

    hist = np.maximum(hist, float(min_count))
    total = float(hist.sum())
    weights = total / (len(hist) * hist)

    if normalize and np.mean(weights) > 0:
        weights = weights / np.mean(weights)

    return torch.tensor(weights, dtype=torch.float32)


def confusion_matrix_np(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def macro_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict[str, Any]:
    """
    计算宏平均分类指标，并返回 confusion matrix 与 per-class 指标。
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=np.int64),
            "per_class": [],
        }

    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    acc = float(np.mean(y_true == y_pred))

    precisions, recalls, f1s = [], [], []
    per_class = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        tn = float(cm.sum() - tp - fp - fn)

        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        support = int(cm[c, :].sum())

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        per_class.append({
            "class_id": c,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "specificity": float(specificity),
            "support": support,
        })

    return {
        "accuracy": acc,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def false_idle_metrics(
    mapped_true: Sequence[int],
    mapped_pred: Sequence[int],
    idle_class: int = 0,
) -> Dict[str, float]:
    """
    工业场景里非常关键的指标：
    真实非 idle，但被错判为 idle 的比例。

    这类错误最危险，因为它会导致发送端直接不传关键切削数据。
    """
    y_true = np.asarray(mapped_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(mapped_pred, dtype=np.int64).reshape(-1)

    non_idle_mask = y_true != idle_class
    true_non_idle = max(int(np.sum(non_idle_mask)), 1)

    false_idle = int(np.sum((y_true != idle_class) & (y_pred == idle_class)))
    false_idle_rate = false_idle / true_non_idle

    true_idle = max(int(np.sum(y_true == idle_class)), 1)
    idle_as_non_idle = int(np.sum((y_true == idle_class) & (y_pred != idle_class)))
    idle_false_alarm_rate = idle_as_non_idle / true_idle

    return {
        "false_idle_count": float(false_idle),
        "false_idle_rate": float(false_idle_rate),
        "idle_false_alarm_count": float(idle_as_non_idle),
        "idle_false_alarm_rate": float(idle_false_alarm_rate),
    }


# =========================================================
# 递归数据结构处理
# =========================================================
def recursive_to_numpy(obj: Any) -> Any:
    """
    递归把嵌套结构中的 tensor 转成 numpy / Python 标量。
    """
    if torch.is_tensor(obj):
        obj = obj.detach().cpu()
        if obj.ndim == 0:
            return obj.item()
        return obj.numpy()

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: recursive_to_numpy(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [recursive_to_numpy(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(recursive_to_numpy(v) for v in obj)

    return obj


def detach_to_cpu(obj: Any) -> Any:
    """
    递归把 tensor 从计算图分离并移动到 CPU。
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {k: detach_to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [detach_to_cpu(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(detach_to_cpu(v) for v in obj)

    return obj


def move_to_device(obj: Any, device: Union[str, torch.device]) -> Any:
    """
    递归把嵌套结构中的 tensor 移动到指定 device。
    """
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)

    return obj


# =========================================================
# 文件读写与估算工具
# =========================================================
def save_json(obj: Dict[str, Any], path: Union[str, Path], indent: int = 2, ensure_ascii: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_obj = recursive_to_numpy(obj)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64, np.int32, np.int64)):
            return o.item()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_obj, f, indent=indent, ensure_ascii=ensure_ascii, default=_default)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_tensor_nbytes(x: Union[np.ndarray, torch.Tensor]) -> int:
    if torch.is_tensor(x):
        return int(x.numel() * x.element_size())
    x = np.asarray(x)
    return int(x.nbytes)


def estimate_object_nbytes(obj: Any) -> int:
    """
    粗略估算嵌套对象占用字节数。
    对 packet 大小评估很有用。
    """
    if obj is None:
        return 0

    if torch.is_tensor(obj):
        return estimate_tensor_nbytes(obj)

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)

    if isinstance(obj, (bytes, bytearray)):
        return len(obj)

    if isinstance(obj, str):
        return len(obj.encode("utf-8"))

    if isinstance(obj, (int, float, bool, np.generic)):
        return 8

    if isinstance(obj, dict):
        return sum(estimate_object_nbytes(k) + estimate_object_nbytes(v) for k, v in obj.items())

    if isinstance(obj, (list, tuple, set)):
        return sum(estimate_object_nbytes(v) for v in obj)

    return 0


# =========================================================
# 指标格式化
# =========================================================
def format_metrics(metrics: Dict[str, Any], prefix: str = "") -> str:
    """
    将嵌套 metrics dict 格式化为一行易读文本。
    """
    parts = []

    def _walk(d: Dict[str, Any], parent_key: str = ""):
        for k, v in d.items():
            key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                _walk(v, key)
            elif isinstance(v, np.ndarray):
                continue
            elif isinstance(v, list):
                continue
            elif isinstance(v, (float, int, np.floating, np.integer)):
                if isinstance(v, float):
                    parts.append(f"{key}={v:.6f}")
                else:
                    parts.append(f"{key}={v}")
            else:
                parts.append(f"{key}={v}")

    _walk(metrics)
    line = " | ".join(parts)
    return f"{prefix}{line}" if prefix else line


# =========================================================
# 统计器与计时器
# =========================================================
class AverageMeter:
    def __init__(self, name: str = "meter"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        val = float(val)
        n = int(n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __repr__(self):
        return f"{self.name}(val={self.val:.6f}, avg={self.avg:.6f}, count={self.count})"


class LatencyMeter:
    """
    用于统计端到端传输或推理延迟。
    """
    def __init__(self, name: str = "latency_ms"):
        self.name = name
        self.values: List[float] = []

    def reset(self):
        self.values = []

    def update(self, latency_ms: float):
        self.values.append(float(latency_ms))

    @property
    def count(self) -> int:
        return len(self.values)

    def summary(self) -> Dict[str, float]:
        if len(self.values) == 0:
            return {
                "count": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        arr = np.asarray(self.values, dtype=np.float32)
        return {
            "count": float(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def __repr__(self):
        s = self.summary()
        return (
            f"{self.name}(count={int(s['count'])}, mean={s['mean']:.4f}, std={s['std']:.4f}, "
            f"min={s['min']:.4f}, max={s['max']:.4f}, p95={s['p95']:.4f})"
        )


class Timer:
    def __init__(self, name: str = "timer"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()
        self.end_time = None
        return self

    def stop(self):
        self.end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return float(end - self.start_time)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()