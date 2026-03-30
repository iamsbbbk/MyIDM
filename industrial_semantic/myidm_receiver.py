import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_semantic.utils import recursive_to_numpy, ensure_signal_ch_first


def symmetric_dequantize_ndarray(q: np.ndarray, scale: Union[float, np.ndarray]) -> np.ndarray:
    return np.asarray(q, dtype=np.float32) * np.asarray(scale, dtype=np.float32)


def _to_tensor(x: Any, device: str = "cpu") -> Optional[torch.Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.float().to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    return None


def _maybe_squeeze_batch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _extract_signal_from_any(obj: Any) -> Optional[np.ndarray]:
    """
    从模型输出中尽量提取 [C,T] 或 [B,C,T] 形式的重构信号。
    """
    if obj is None:
        return None

    if torch.is_tensor(obj):
        arr = obj.detach().cpu().numpy()
        arr = _maybe_squeeze_batch(arr)
        if arr.ndim == 2:
            if 3 in arr.shape:
                return ensure_signal_ch_first(arr)
        if arr.ndim == 3:
            return arr.astype(np.float32)
        return None

    if isinstance(obj, np.ndarray):
        arr = _maybe_squeeze_batch(obj)
        if arr.ndim == 2 and 3 in arr.shape:
            return ensure_signal_ch_first(arr)
        if arr.ndim == 3:
            return arr.astype(np.float32)
        return None

    if isinstance(obj, dict):
        preferred_keys = [
            "reconstructed_signal",
            "signal",
            "x_hat",
            "recon",
            "reconstruction",
            "output",
            "pred",
        ]
        for k in preferred_keys:
            if k in obj:
                sig = _extract_signal_from_any(obj[k])
                if sig is not None:
                    return sig

        for _, v in obj.items():
            sig = _extract_signal_from_any(v)
            if sig is not None:
                return sig
        return None

    if isinstance(obj, (list, tuple)):
        for v in obj:
            sig = _extract_signal_from_any(v)
            if sig is not None:
                return sig
        return None

    return None


class MyIDMReceiver:
    def __init__(
        self,
        device: str = "cpu",
        checkpoint: str = "",
        myidm_module: str = "model",
        myidm_forward_module: str = "forward",
        model_class_name: str = "",
        model_class_candidates=None,
        builder_function_candidates=None,
        function_candidates=None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        return_raw_when_available: bool = True,
        verbose: bool = False,
    ):
        self.device = device
        self.checkpoint = checkpoint
        self.myidm_module_name = myidm_module
        self.myidm_forward_module_name = myidm_forward_module
        self.model_class_name = model_class_name
        self.model_class_candidates = model_class_candidates or [
            "MyIDM",
            "IDM",
            "Model",
            "Net",
            "Receiver",
            "Decoder",
        ]
        self.builder_function_candidates = builder_function_candidates or [
            "build_model",
            "create_model",
            "get_model",
            "load_model",
        ]
        self.function_candidates = function_candidates or [
            "decode_packet",
            "decode",
            "reconstruct",
            "inference",
            "forward",
            "sample",
        ]
        self.init_kwargs = init_kwargs or {}
        self.return_raw_when_available = bool(return_raw_when_available)
        self.verbose = bool(verbose)

        self.model_module = self._safe_import(self.myidm_module_name)
        self.forward_module = self._safe_import(self.myidm_forward_module_name)

        self.model = self._build_model()
        self.forward_fn = self._find_forward_fn()

    def _log(self, *msg):
        if self.verbose:
            print("[MyIDMReceiver]", *msg)

    def _safe_import(self, module_name: str):
        if not module_name:
            return None
        try:
            mod = importlib.import_module(module_name)
            self._log(f"import success: {module_name}")
            return mod
        except Exception as e:
            self._log(f"import failed: {module_name}, err={e}")
            return None

    def _try_instantiate(self, obj, kwargs: Dict[str, Any]):
        # 尽量兼容不同构造签名
        tries = [
            lambda: obj(**kwargs),
            lambda: obj(kwargs),
            lambda: obj(),
        ]
        last_err = None
        for fn in tries:
            try:
                return fn()
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise last_err

    def _load_checkpoint_if_needed(self, model):
        if model is None or not self.checkpoint:
            return model
        if not hasattr(model, "load_state_dict"):
            return model

        ckpt_path = Path(self.checkpoint)
        if not ckpt_path.exists():
            self._log(f"checkpoint not found: {ckpt_path}")
            return model

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state = ckpt
            if isinstance(ckpt, dict):
                for key in ["model_state_dict", "state_dict", "model", "ema_state_dict"]:
                    if key in ckpt and isinstance(ckpt[key], dict):
                        state = ckpt[key]
                        break
            model.load_state_dict(state, strict=False)
            self._log(f"checkpoint loaded: {ckpt_path}")
        except Exception as e:
            self._log(f"checkpoint load failed: {e}")

        return model

    def _build_model(self):
        mod = self.model_module
        if mod is None:
            return None

        # 1. 显式 class name
        if self.model_class_name and hasattr(mod, self.model_class_name):
            cls = getattr(mod, self.model_class_name)
            try:
                model = self._try_instantiate(cls, self.init_kwargs)
                model = self._load_checkpoint_if_needed(model)
                if isinstance(model, nn.Module):
                    model = model.to(self.device).eval()
                self._log(f"model created from explicit class: {self.model_class_name}")
                return model
            except Exception as e:
                self._log(f"explicit class instantiate failed: {e}")

        # 2. builder 函数
        for fn_name in self.builder_function_candidates:
            if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
                fn = getattr(mod, fn_name)
                for attempt in [
                    lambda: fn(**self.init_kwargs),
                    lambda: fn(self.init_kwargs),
                    lambda: fn(),
                ]:
                    try:
                        model = attempt()
                        model = self._load_checkpoint_if_needed(model)
                        if isinstance(model, nn.Module):
                            model = model.to(self.device).eval()
                        self._log(f"model created from builder: {fn_name}")
                        return model
                    except Exception:
                        pass

        # 3. 常见 class candidates
        for cls_name in self.model_class_candidates:
            if hasattr(mod, cls_name):
                cls = getattr(mod, cls_name)
                if inspect.isclass(cls):
                    try:
                        model = self._try_instantiate(cls, self.init_kwargs)
                        model = self._load_checkpoint_if_needed(model)
                        if isinstance(model, nn.Module):
                            model = model.to(self.device).eval()
                        self._log(f"model created from candidate class: {cls_name}")
                        return model
                    except Exception:
                        pass

        # 4. 找模块中任意 nn.Module 子类
        for name, obj in mod.__dict__.items():
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj.__module__ == mod.__name__:
                try:
                    model = self._try_instantiate(obj, self.init_kwargs)
                    model = self._load_checkpoint_if_needed(model)
                    if isinstance(model, nn.Module):
                        model = model.to(self.device).eval()
                    self._log(f"model created from generic nn.Module class: {name}")
                    return model
                except Exception:
                    continue

        self._log("no usable model instance found, fallback mode enabled")
        return None

    def _find_forward_fn(self):
        mod = self.forward_module
        if mod is None:
            return None

        for fn_name in self.function_candidates:
            if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
                self._log(f"forward function found: {fn_name}")
                return getattr(mod, fn_name)

        # 兜底：如果模块里有任何同名函数
        for name, obj in mod.__dict__.items():
            if callable(obj) and obj.__module__ == mod.__name__:
                if name in ["forward", "inference", "sample", "decode", "reconstruct"]:
                    self._log(f"forward fallback function found: {name}")
                    return obj

        return None

    def _normalize_packet_like(self, packet_like: Any) -> Dict[str, Any]:
        """
        标准化成：
        {
            "header": ...,
            "semantic": ...,
            "anchors": ...,
            "payload": ...
        }
        """
        if packet_like is None:
            return {"header": {}, "semantic": {}, "anchors": {}, "payload": {}}

        if hasattr(packet_like, "to_dict") and callable(packet_like.to_dict):
            d = packet_like.to_dict()
            if isinstance(d, dict):
                return {
                    "header": d.get("header", {}),
                    "semantic": d.get("semantic", {}),
                    "anchors": d.get("anchors", {}),
                    "payload": d.get("payload", {}),
                }

        if isinstance(packet_like, dict):
            # 已经是 packet dict
            if any(k in packet_like for k in ["header", "semantic", "anchors", "payload"]):
                return {
                    "header": packet_like.get("header", {}),
                    "semantic": packet_like.get("semantic", {}),
                    "anchors": packet_like.get("anchors", {}),
                    "payload": packet_like.get("payload", {}),
                }

            # 如果直接是 payload dict
            return {
                "header": {},
                "semantic": {},
                "anchors": {},
                "payload": packet_like,
            }

        return {"header": {}, "semantic": {}, "anchors": {}, "payload": {}}

    def _extract_inputs(self, packet_like: Any) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
        packet_dict = self._normalize_packet_like(packet_like)
        payload = packet_dict.get("payload", {})

        raw_signal = None
        latent = None

        # 1. raw signal
        if "raw_signal" in payload and payload["raw_signal"] is not None:
            raw_signal = np.asarray(payload["raw_signal"], dtype=np.float32)
            if raw_signal.ndim == 2 and 3 in raw_signal.shape:
                raw_signal = ensure_signal_ch_first(raw_signal)

        # 2. latent
        if "latent" in payload and payload["latent"] is not None:
            latent = np.asarray(payload["latent"], dtype=np.float32)

        if latent is None and "latent_q" in payload and "latent_scale" in payload:
            latent = symmetric_dequantize_ndarray(payload["latent_q"], payload["latent_scale"])

        # 顶层兼容
        if raw_signal is None and isinstance(packet_like, dict) and "raw_signal" in packet_like:
            raw_signal = np.asarray(packet_like["raw_signal"], dtype=np.float32)
            if raw_signal.ndim == 2 and 3 in raw_signal.shape:
                raw_signal = ensure_signal_ch_first(raw_signal)

        if latent is None and isinstance(packet_like, dict):
            if "latent" in packet_like and packet_like["latent"] is not None:
                latent = np.asarray(packet_like["latent"], dtype=np.float32)
            elif "latent_q" in packet_like and "latent_scale" in packet_like:
                latent = symmetric_dequantize_ndarray(packet_like["latent_q"], packet_like["latent_scale"])

        return packet_dict, latent, raw_signal

    def _call_with_candidates(self, fn, candidate_args):
        last_err = None
        for desc, arg in candidate_args:
            try:
                out = fn(arg)
                return out, desc, None
            except Exception as e:
                last_err = e
        return None, None, last_err

    def _run_model(self, packet_dict, latent, raw_signal):
        payload = packet_dict.get("payload", {})

        latent_tensor = _to_tensor(latent, device=self.device)
        if latent_tensor is not None and latent_tensor.ndim == 2:
            latent_tensor = latent_tensor.unsqueeze(0)

        raw_tensor = _to_tensor(raw_signal, device=self.device)
        if raw_tensor is not None and raw_tensor.ndim == 2:
            raw_tensor = raw_tensor.unsqueeze(0)

        unified_dict = {
            "packet": packet_dict,
            "header": packet_dict.get("header", {}),
            "semantic": packet_dict.get("semantic", {}),
            "anchors": packet_dict.get("anchors", {}),
            "payload": payload,
            "latent": latent_tensor,
            "raw_signal": raw_tensor,
        }

        # A. 优先尝试 model 对象
        if self.model is not None:
            method_candidates = [
                "decode_packet",
                "decode",
                "reconstruct",
                "inference",
                "forward",
            ]

            for method_name in method_candidates:
                if hasattr(self.model, method_name):
                    fn = getattr(self.model, method_name)
                    out, desc, err = self._call_with_candidates(
                        fn,
                        [
                            ("packet_dict", packet_dict),
                            ("unified_dict", unified_dict),
                            ("payload", payload),
                            ("latent_tensor", latent_tensor),
                            ("raw_tensor", raw_tensor),
                        ],
                    )
                    if out is not None:
                        return out, f"model.{method_name}:{desc}"

            # model callable
            if callable(self.model):
                out, desc, err = self._call_with_candidates(
                    self.model,
                    [
                        ("latent_tensor", latent_tensor),
                        ("raw_tensor", raw_tensor),
                        ("unified_dict", unified_dict),
                        ("packet_dict", packet_dict),
                    ],
                )
                if out is not None:
                    return out, f"model.__call__:{desc}"

        # B. 尝试 forward.py 中的函数
        if self.forward_fn is not None:
            # 尝试 forward_fn(arg) 或 forward_fn(model, arg)
            for desc, arg in [
                ("packet_dict", packet_dict),
                ("unified_dict", unified_dict),
                ("payload", payload),
                ("latent_tensor", latent_tensor),
                ("raw_tensor", raw_tensor),
            ]:
                try:
                    out = self.forward_fn(arg)
                    return out, f"forward_fn:{desc}"
                except Exception:
                    pass

                if self.model is not None:
                    try:
                        out = self.forward_fn(self.model, arg)
                        return out, f"forward_fn(model,{desc})"
                    except Exception:
                        pass

        return None, "no_backend"

    def _normalize_output(self, out, used_path: str, packet_dict: Dict[str, Any], latent, raw_signal):
        recon = _extract_signal_from_any(out)

        result = {
            "used_path": used_path,
            "packet_mode": packet_dict.get("header", {}).get("payload_mode", "NA"),
            "semantic": packet_dict.get("semantic", {}),
            "anchors": packet_dict.get("anchors", {}),
            "reconstructed_signal": recon,
            "receiver_raw_output": recursive_to_numpy(out),
        }

        # 如果模型没显式输出可识别信号，但 packet 本身带 raw_signal，可兜底
        if result["reconstructed_signal"] is None and raw_signal is not None and self.return_raw_when_available:
            result["reconstructed_signal"] = np.asarray(raw_signal, dtype=np.float32)
            result["used_path"] = used_path + "+raw_fallback"

        return result

    def decode_packet(self, packet_like: Any) -> Dict[str, Any]:
        packet_dict, latent, raw_signal = self._extract_inputs(packet_like)
        out, used_path = self._run_model(packet_dict, latent=latent, raw_signal=raw_signal)

        if out is not None:
            return self._normalize_output(out, used_path, packet_dict, latent, raw_signal)

        # 后端都不可用时，仍然保证系统可运行
        if raw_signal is not None and self.return_raw_when_available:
            return {
                "used_path": "packet_raw_passthrough",
                "packet_mode": packet_dict.get("header", {}).get("payload_mode", "NA"),
                "semantic": packet_dict.get("semantic", {}),
                "anchors": packet_dict.get("anchors", {}),
                "reconstructed_signal": np.asarray(raw_signal, dtype=np.float32),
                "receiver_raw_output": None,
            }

        return {
            "used_path": "semantic_only",
            "packet_mode": packet_dict.get("header", {}).get("payload_mode", "NA"),
            "semantic": packet_dict.get("semantic", {}),
            "anchors": packet_dict.get("anchors", {}),
            "reconstructed_signal": None,
            "receiver_raw_output": None,
        }

    def decode(self, x: Any) -> Dict[str, Any]:
        # 如果传进来本身就是 packet-like，就按 packet 处理
        if isinstance(x, dict) and any(k in x for k in ["header", "semantic", "anchors", "payload", "latent", "raw_signal"]):
            return self.decode_packet(x)

        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            x_np = x
        else:
            return {
                "used_path": "decode_unsupported",
                "packet_mode": "NA",
                "semantic": {},
                "anchors": {},
                "reconstructed_signal": None,
                "receiver_raw_output": recursive_to_numpy(x),
            }

        x_np = np.asarray(x_np)

        # 若像原始三轴信号，则直接回传
        if x_np.ndim in [2, 3]:
            arr = _maybe_squeeze_batch(x_np)
            if arr.ndim == 2 and 3 in arr.shape:
                arr = ensure_signal_ch_first(arr)
                return {
                    "used_path": "decode_raw_direct",
                    "packet_mode": "NA",
                    "semantic": {},
                    "anchors": {},
                    "reconstructed_signal": arr.astype(np.float32),
                    "receiver_raw_output": None,
                }

        # 其余情况当作 latent
        pkt = {
            "header": {"payload_mode": "M2"},
            "semantic": {},
            "anchors": {},
            "payload": {"latent": x_np},
        }
        return self.decode_packet(pkt)

    def reconstruct(self, x: Any) -> Dict[str, Any]:
        return self.decode(x)

    def inference(self, x: Any) -> Dict[str, Any]:
        return self.decode(x)

    def forward(self, x: Any) -> Dict[str, Any]:
        return self.decode(x)
