import importlib
import inspect
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .model import load_sender_checkpoint
from .protocol import SemanticPacket, bytes_to_packet
from .utils import (
    ensure_batch_bct,
    ensure_signal_ch_first,
    estimate_object_nbytes,
    get_logger,
    recursive_to_numpy,
)


def symmetric_dequantize_ndarray(q: np.ndarray, scale: Union[float, np.ndarray]) -> np.ndarray:
    return np.asarray(q, dtype=np.float32) * np.asarray(scale, dtype=np.float32)


def _import_object(module_name: str, class_name: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _safe_load_checkpoint_into_obj(obj: Any, checkpoint_path: str, device: str = "cpu"):
    if not checkpoint_path:
        return obj

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"接收端 checkpoint 不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # 1) 优先兼容 nn.Module.load_state_dict
    if hasattr(obj, "load_state_dict"):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        obj.load_state_dict(state_dict, strict=False)
        return obj

    # 2) 兼容自定义 load_model / load_ckpt
    for method_name in ["load_model", "load_ckpt", "load_checkpoint"]:
        if hasattr(obj, method_name):
            getattr(obj, method_name)(str(ckpt_path))
            return obj

    # 3) 没有加载入口则直接返回
    return obj


class IdentityPreprocessHook:
    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context


class IdentityPostprocessHook:
    def __call__(self, result: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return result


class PacketReplayGuard:
    """
    防止重复 packet 被反复处理。
    """

    def __init__(self, capacity: int = 4096):
        self.capacity = int(capacity)
        self.cache = OrderedDict()

    def _make_key(self, packet: SemanticPacket) -> str:
        h = packet.header
        return "|".join(
            [
                str(h.get("session_id", "")),
                str(h.get("machine_id", "")),
                str(h.get("spindle_id", "")),
                str(h.get("tool_id", "")),
                str(h.get("run_id", "")),
                str(h.get("window_id", "")),
                str(h.get("checksum", "")),
            ]
        )

    def seen(self, packet: SemanticPacket) -> bool:
        key = self._make_key(packet)
        if key in self.cache:
            self.cache.move_to_end(key)
            return True

        self.cache[key] = True
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return False


class ReceiverSessionTracker:
    """
    跟踪每个 session/tool/run 的接收状态。
    """

    def __init__(self, history_size: int = 256):
        self.history_size = int(history_size)
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _session_key(self, header: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(header.get("session_id", "")),
                str(header.get("machine_id", "")),
                str(header.get("spindle_id", "")),
                str(header.get("tool_id", "")),
                str(header.get("run_id", "")),
            ]
        )

    def _ensure_session(self, header: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        key = self._session_key(header)
        if key not in self.sessions:
            self.sessions[key] = {
                "session_key": key,
                "packet_count": 0,
                "active_packet_count": 0,
                "idle_segment_count": 0,
                "idle_window_count": 0,
                "idle_duration_sec": 0.0,
                "duplicate_count": 0,
                "alarm_count": 0,
                "last_window_id": None,
                "last_packet_mode": None,
                "last_mapped_class": None,
                "last_confidence": None,
                "last_timestamp_end": None,
                "wear_history": deque(maxlen=self.history_size),
                "confidence_history": deque(maxlen=self.history_size),
                "mode_history": deque(maxlen=self.history_size),
                "used_path_history": deque(maxlen=self.history_size),
                "updated_at": None,
            }
        return key, self.sessions[key]

    def mark_duplicate(self, header: Dict[str, Any]):
        _, sess = self._ensure_session(header)
        sess["duplicate_count"] += 1
        sess["updated_at"] = time.time()

    def update(self, packet: SemanticPacket, result: Dict[str, Any]) -> Dict[str, Any]:
        header = packet.header
        semantic = packet.semantic
        payload = packet.payload

        _, sess = self._ensure_session(header)
        sess["packet_count"] += 1
        sess["last_window_id"] = header.get("window_id")
        sess["last_packet_mode"] = header.get("payload_mode")
        sess["last_mapped_class"] = semantic.get("mapped_class", None)
        sess["last_confidence"] = semantic.get("confidence", None)
        sess["last_timestamp_end"] = header.get("timestamp_end", None)
        sess["updated_at"] = time.time()

        sess["mode_history"].append(header.get("payload_mode"))
        sess["used_path_history"].append(result.get("used_path", "unknown"))

        confidence = semantic.get("confidence", None)
        if confidence is not None:
            sess["confidence_history"].append(float(confidence))

        mapped_class = semantic.get("mapped_class", None)
        if mapped_class is not None:
            sess["wear_history"].append(int(mapped_class))

        mode = header.get("payload_mode", "M2")
        if mode == "M0":
            sess["idle_segment_count"] += 1
            sess["idle_window_count"] += int(payload.get("idle_window_count", 1))
            duration = float(payload.get("idle_duration_sec", 0.0))
            if duration <= 0:
                try:
                    duration = float(header.get("timestamp_end", 0.0)) - float(header.get("timestamp_start", 0.0))
                except Exception:
                    duration = 0.0
            sess["idle_duration_sec"] += max(duration, 0.0)
        else:
            sess["active_packet_count"] += 1

        if bool(result.get("alarm", False)):
            sess["alarm_count"] += 1

        return self.snapshot_by_header(header)

    def snapshot_by_header(self, header: Dict[str, Any]) -> Dict[str, Any]:
        key, sess = self._ensure_session(header)
        return {
            "session_key": key,
            "packet_count": int(sess["packet_count"]),
            "active_packet_count": int(sess["active_packet_count"]),
            "idle_segment_count": int(sess["idle_segment_count"]),
            "idle_window_count": int(sess["idle_window_count"]),
            "idle_duration_sec": float(sess["idle_duration_sec"]),
            "duplicate_count": int(sess["duplicate_count"]),
            "alarm_count": int(sess["alarm_count"]),
            "last_window_id": sess["last_window_id"],
            "last_packet_mode": sess["last_packet_mode"],
            "last_mapped_class": sess["last_mapped_class"],
            "last_confidence": sess["last_confidence"],
            "last_timestamp_end": sess["last_timestamp_end"],
            "wear_history": list(sess["wear_history"]),
            "confidence_history": list(sess["confidence_history"]),
            "mode_history": list(sess["mode_history"]),
            "used_path_history": list(sess["used_path_history"]),
            "updated_at": sess["updated_at"],
        }


class MyIDMModelBridge:
    """
    动态桥接 MyIDM 原有接收模型。
    支持 4 类输入调用路径：
    1. packet 级
    2. latent 级
    3. signal/raw_signal 级
    4. callable 自动尝试

    这样就不需要大改 MyIDM 原模型。
    """

    def __init__(self, receiver_cfg: Dict[str, Any], device: Optional[str] = None):
        self.cfg = receiver_cfg or {}
        self.logger = get_logger("MyIDMModelBridge")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.accept_mode = self.cfg.get("accept_mode", "auto")
        self.invocation_order = self.cfg.get(
            "invocation_order",
            ["packet", "latent", "signal", "callable"],
        )

        self.packet_method_candidates = self.cfg.get(
            "packet_method_candidates",
            ["decode_packet", "receive_packet", "decode", "inference"],
        )
        self.latent_method_candidates = self.cfg.get(
            "latent_method_candidates",
            ["decode_latent", "decode", "reconstruct_from_latent", "forward", "inference"],
        )
        self.signal_method_candidates = self.cfg.get(
            "signal_method_candidates",
            ["predict_signal", "reconstruct", "forward", "predict", "inference"],
        )

        self.expected_signal_channels = int(self.cfg.get("expected_signal_channels", 3))

        self.preprocess_hook = self._build_hook(
            module_name=self.cfg.get("preprocess_module", ""),
            class_name=self.cfg.get("preprocess_class_name", ""),
            kwargs=self.cfg.get("preprocess_kwargs", {}) or {},
            default=IdentityPreprocessHook(),
        )
        self.postprocess_hook = self._build_hook(
            module_name=self.cfg.get("postprocess_module", ""),
            class_name=self.cfg.get("postprocess_class_name", ""),
            kwargs=self.cfg.get("postprocess_kwargs", {}) or {},
            default=IdentityPostprocessHook(),
        )

        self.model = self._load_model()

    def _build_hook(self, module_name: str, class_name: str, kwargs: Dict[str, Any], default: Any):
        if not module_name or not class_name:
            return default
        try:
            cls = _import_object(module_name, class_name)
            return cls(**kwargs)
        except Exception as e:
            self.logger.warning(f"Hook 加载失败，将使用默认实现: {module_name}.{class_name}, err={e}")
            return default

    def _load_model(self):
        module_name = self.cfg.get("module", "")
        class_name = self.cfg.get("class_name", "")
        checkpoint = self.cfg.get("checkpoint", "")
        init_kwargs = self.cfg.get("init_kwargs", {}) or {}

        if not module_name or not class_name:
            self.logger.warning("未配置 receiver.module/class_name，MyIDM 模型桥将处于未激活状态。")
            return None

        try:
            cls = _import_object(module_name, class_name)
            obj = cls(**init_kwargs)
            obj = _safe_load_checkpoint_into_obj(obj, checkpoint, device=self.device)

            if isinstance(obj, torch.nn.Module):
                obj.to(self.device)
                obj.eval()

            self.logger.info(f"MyIDM 模型已加载: {module_name}.{class_name}")
            return obj
        except Exception as e:
            self.logger.warning(f"MyIDM 模型加载失败: {module_name}.{class_name}, err={e}")
            return None

    def _latent_to_tensor(self, latent: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if latent is None:
            return None
        latent = np.asarray(latent, dtype=np.float32)
        if latent.ndim == 2:
            latent_t = torch.from_numpy(latent).unsqueeze(0).float()
        elif latent.ndim == 3:
            latent_t = torch.from_numpy(latent).float()
        else:
            raise ValueError(f"latent 维度错误，期望 2D/3D，实际 {latent.shape}")
        return latent_t.to(self.device)

    def _signal_to_tensor(self, signal: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if signal is None:
            return None
        signal_t = ensure_batch_bct(signal, channels=self.expected_signal_channels)
        return signal_t.to(self.device)

    def _build_context(
        self,
        packet: SemanticPacket,
        latent: Optional[np.ndarray],
        signal: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        context = {
            "packet": packet,
            "packet_dict": packet.to_dict(),
            "header": packet.header,
            "semantic": packet.semantic,
            "anchors": packet.anchors,
            "payload": packet.payload,
            "mode": packet.payload_mode,
            "latent": latent,
            "latent_tensor": self._latent_to_tensor(latent),
            "signal": signal,
            "raw_signal": signal,
            "signal_tensor": self._signal_to_tensor(signal),
            "raw_signal_tensor": self._signal_to_tensor(signal),
        }
        context = self.preprocess_hook(context)
        return context

    def _call_with_variants(
        self,
        fn,
        context: Dict[str, Any],
        positional_candidates: List[Tuple[str, Any]],
    ) -> Tuple[Any, str]:
        last_err = None

        # 1) 优先按签名尝试 kwargs 调用
        try:
            sig = inspect.signature(fn)
            params = sig.parameters

            if len(params) == 0:
                return fn(), "noarg"

            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if accepts_kwargs:
                return fn(**context), "kwargs-full"

            kwargs = {}
            for name in params.keys():
                if name in context:
                    kwargs[name] = context[name]

            if len(kwargs) > 0:
                return fn(**kwargs), f"kwargs:{list(kwargs.keys())}"
        except Exception as e:
            last_err = e

        # 2) 再尝试若干位置参数形式
        for tag, value in positional_candidates:
            if value is None:
                continue
            try:
                return fn(value), tag
            except Exception as e:
                last_err = e

        # 3) 全部失败
        if last_err is not None:
            raise last_err
        raise RuntimeError("未能成功调用目标函数")

    def _invoke_on_target(
        self,
        target_obj: Any,
        method_names: Sequence[str],
        context: Dict[str, Any],
        method_group: str,
    ) -> Optional[Dict[str, Any]]:
        if target_obj is None:
            return None

        if method_group == "packet":
            positional_candidates = [
                ("packet", context["packet"]),
                ("packet_dict", context["packet_dict"]),
                ("payload", context["payload"]),
            ]
        elif method_group == "latent":
            positional_candidates = [
                ("latent_tensor", context["latent_tensor"]),
                ("latent_np", context["latent"]),
                ("packet", context["packet"]),
            ]
        elif method_group == "signal":
            positional_candidates = [
                ("signal_tensor", context["signal_tensor"]),
                ("signal_np", context["signal"]),
                ("packet", context["packet"]),
            ]
        else:
            positional_candidates = [
                ("packet", context["packet"]),
                ("latent_tensor", context["latent_tensor"]),
                ("signal_tensor", context["signal_tensor"]),
            ]

        for method_name in method_names:
            if not hasattr(target_obj, method_name):
                continue
            fn = getattr(target_obj, method_name)
            try:
                out, call_tag = self._call_with_variants(fn, context, positional_candidates)
                return {
                    "used_path": f"myidm:{method_group}:{method_name}:{call_tag}",
                    "raw_output": out,
                }
            except Exception as e:
                self.logger.debug(f"调用失败 {method_group}.{method_name}, err={e}")
                continue
        return None

    def _extract_signal_like(self, obj: Any) -> Optional[np.ndarray]:
        if obj is None:
            return None

        if torch.is_tensor(obj):
            obj = obj.detach().cpu().numpy()

        if isinstance(obj, np.ndarray):
            arr = obj
        elif isinstance(obj, list):
            try:
                arr = np.asarray(obj, dtype=np.float32)
            except Exception:
                return None
        else:
            return None

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 2:
            if arr.shape[0] == self.expected_signal_channels or arr.shape[1] == self.expected_signal_channels:
                try:
                    return ensure_signal_ch_first(arr, expected_channels=self.expected_signal_channels)
                except Exception:
                    return None

        if arr.ndim == 3:
            # [B, C, T] 或 [B, T, C]
            if arr.shape[0] == 1:
                arr2 = arr[0]
                if arr2.shape[0] == self.expected_signal_channels or arr2.shape[1] == self.expected_signal_channels:
                    try:
                        return ensure_signal_ch_first(arr2, expected_channels=self.expected_signal_channels)
                    except Exception:
                        return None

        return None

    def _standardize_output(self, raw_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        out = recursive_to_numpy(raw_result.get("raw_output", None))
        std = {
            "used_path": raw_result.get("used_path", "myidm:unknown"),
            "receiver_output": out,
            "reconstructed_signal": None,
            "prediction": None,
            "confidence": None,
            "logits": None,
        }

        # 1) 如果直接就是信号
        sig = self._extract_signal_like(out)
        if sig is not None:
            std["reconstructed_signal"] = sig
            std = self.postprocess_hook(std, context=context)
            return std

        # 2) dict 输出
        if isinstance(out, dict):
            signal_keys = [
                "reconstructed_signal",
                "reconstruction",
                "signal",
                "output_signal",
                "x_hat",
                "decoded_signal",
            ]
            pred_keys = [
                "prediction",
                "pred",
                "label",
                "class_id",
                "mapped_class",
            ]
            conf_keys = [
                "confidence",
                "score",
                "prob",
            ]
            logits_keys = [
                "logits",
                "pred_logits",
                "cls_logits",
            ]

            for k in signal_keys:
                if k in out:
                    sig = self._extract_signal_like(out[k])
                    if sig is not None:
                        std["reconstructed_signal"] = sig
                        break

            for k in pred_keys:
                if k in out:
                    std["prediction"] = out[k]
                    break

            for k in conf_keys:
                if k in out:
                    std["confidence"] = out[k]
                    break

            for k in logits_keys:
                if k in out:
                    std["logits"] = out[k]
                    break

            std = self.postprocess_hook(std, context=context)
            return std

        # 3) tuple/list 输出
        if isinstance(out, (list, tuple)):
            if len(out) > 0:
                sig0 = self._extract_signal_like(out[0])
                if sig0 is not None:
                    std["reconstructed_signal"] = sig0
                else:
                    std["prediction"] = out[0]
            if len(out) > 1:
                std["logits"] = out[1]
            std = self.postprocess_hook(std, context=context)
            return std

        # 4) 标量 / 其他输出
        std["prediction"] = out
        std = self.postprocess_hook(std, context=context)
        return std

    def invoke(
        self,
        packet: SemanticPacket,
        latent: Optional[np.ndarray] = None,
        signal: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.model is None:
            return None

        context = self._build_context(packet, latent=latent, signal=signal)

        order = list(self.invocation_order)
        if self.accept_mode == "packet":
            order = ["packet", "callable", "latent", "signal"]
        elif self.accept_mode == "latent":
            order = ["latent", "callable", "signal", "packet"]
        elif self.accept_mode == "signal":
            order = ["signal", "callable", "latent", "packet"]

        for item in order:
            if item == "packet":
                result = self._invoke_on_target(
                    self.model,
                    self.packet_method_candidates,
                    context=context,
                    method_group="packet",
                )
                if result is not None:
                    return self._standardize_output(result, context=context)

            elif item == "latent":
                if context["latent_tensor"] is None and context["latent"] is None:
                    continue
                result = self._invoke_on_target(
                    self.model,
                    self.latent_method_candidates,
                    context=context,
                    method_group="latent",
                )
                if result is not None:
                    return self._standardize_output(result, context=context)

            elif item == "signal":
                if context["signal_tensor"] is None and context["signal"] is None:
                    continue
                result = self._invoke_on_target(
                    self.model,
                    self.signal_method_candidates,
                    context=context,
                    method_group="signal",
                )
                if result is not None:
                    return self._standardize_output(result, context=context)

            elif item == "callable":
                if callable(self.model):
                    try:
                        call_result = None
                        if context["signal_tensor"] is not None:
                            call_result = self.model(context["signal_tensor"])
                            used_path = "myidm:callable:signal_tensor"
                        elif context["latent_tensor"] is not None:
                            call_result = self.model(context["latent_tensor"])
                            used_path = "myidm:callable:latent_tensor"
                        else:
                            call_result = self.model(context["packet_dict"])
                            used_path = "myidm:callable:packet_dict"

                        return self._standardize_output(
                            {"used_path": used_path, "raw_output": call_result},
                            context=context,
                        )
                    except Exception as e:
                        self.logger.debug(f"callable 调用失败, err={e}")
                        continue

        return None


class IndustrialSemanticReceiverEngine:
    """
    工业语义通信接收引擎。
    负责：
    1. packet 校验
    2. 去重
    3. idle summary / anchor-only / latent / raw 四类包处理
    4. MyIDM 动态桥接
    5. fallback reference decoder
    6. 状态跟踪与告警融合
    """

    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("IndustrialSemanticReceiverEngine")

        self.receiver_cfg = self.cfg.get("receiver", {})
        self.model_cfg = self.cfg.get("model", {})
        self.expected_signal_channels = int(self.receiver_cfg.get("expected_signal_channels", 3))

        self.strict_protocol = bool(self.receiver_cfg.get("strict_protocol", False))
        self.strict_normalization = bool(self.receiver_cfg.get("strict_normalization", False))
        self.supported_protocols = self.receiver_cfg.get("supported_protocols", ["ISC-TWM-1.0"])
        self.supported_normalization_ids = self.receiver_cfg.get("supported_normalization_ids", [])

        self.allow_duplicate_window = bool(self.receiver_cfg.get("allow_duplicate_window", False))
        self.replay_guard = PacketReplayGuard(capacity=int(self.receiver_cfg.get("replay_cache_size", 4096)))
        self.session_tracker = ReceiverSessionTracker(history_size=int(self.receiver_cfg.get("history_size", 256)))

        self.bridge = MyIDMModelBridge(self.receiver_cfg, device=self.device)

        self.reference_sender = None
        if bool(self.receiver_cfg.get("use_reference_decoder_fallback", True)):
            ref_ckpt = self.receiver_cfg.get("reference_checkpoint", "")
            if ref_ckpt:
                try:
                    self.reference_sender, _ = load_sender_checkpoint(ref_ckpt, device=self.device)
                    self.logger.info(f"reference decoder 已加载: {ref_ckpt}")
                except Exception as e:
                    self.logger.warning(f"reference decoder 加载失败: {e}")

    def _validate_packet(self, packet: SemanticPacket):
        header = packet.header
        protocol_version = header.get("protocol_version", "")
        normalization_id = header.get("normalization_id", "")

        if self.strict_protocol and self.supported_protocols:
            if protocol_version not in self.supported_protocols:
                raise ValueError(
                    f"协议版本不受支持: {protocol_version}, supported={self.supported_protocols}"
                )

        if self.strict_normalization and self.supported_normalization_ids:
            if normalization_id not in self.supported_normalization_ids:
                raise ValueError(
                    f"normalization_id 不受支持: {normalization_id}, supported={self.supported_normalization_ids}"
                )

    def _alarm_from_packet(self, packet: SemanticPacket) -> bool:
        semantic = packet.semantic
        payload = packet.payload

        if bool(payload.get("alarm", False)):
            return True

        if semantic.get("wear_state", "") == "accelerating":
            return True

        if int(semantic.get("mapped_class", -1)) == 3:
            return True

        return False

    def _extract_latent_from_packet(self, packet: SemanticPacket) -> Optional[np.ndarray]:
        payload = packet.payload
        if "latent_q" in payload and "latent_scale" in payload:
            return symmetric_dequantize_ndarray(payload["latent_q"], payload["latent_scale"]).astype(np.float32)
        return None

    def _extract_signal_from_packet(self, packet: SemanticPacket) -> Optional[np.ndarray]:
        payload = packet.payload
        if "raw_signal" in payload:
            try:
                return ensure_signal_ch_first(payload["raw_signal"], expected_channels=self.expected_signal_channels)
            except Exception:
                arr = np.asarray(payload["raw_signal"], dtype=np.float32)
                return arr
        return None

    @torch.no_grad()
    def _decode_latent_with_reference(self, latent: np.ndarray) -> Optional[np.ndarray]:
        if self.reference_sender is None:
            return None
        try:
            latent = np.asarray(latent, dtype=np.float32)
            if latent.ndim == 2:
                latent_t = torch.from_numpy(latent).unsqueeze(0).float().to(self.device)
            elif latent.ndim == 3:
                latent_t = torch.from_numpy(latent).float().to(self.device)
            else:
                return None

            recon = self.reference_sender.decode_latent(latent_t)
            if recon is None:
                return None
            recon = recon[0].detach().cpu().numpy()
            recon = ensure_signal_ch_first(recon, expected_channels=self.expected_signal_channels)
            return recon.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"reference decoder 解码 latent 失败: {e}")
            return None

    def _make_base_result(
        self,
        packet: SemanticPacket,
        packet_nbytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "ok",
            "duplicate": False,
            "packet_mode": packet.payload_mode,
            "header": packet.header,
            "semantic": packet.semantic,
            "anchors": packet.anchors,
            "payload_summary": {
                "keys": list(packet.payload.keys()),
                "estimated_nbytes": int(packet_nbytes if packet_nbytes is not None else estimate_object_nbytes(packet.to_dict())),
            },
            "used_path": None,
            "receiver_output": None,
            "reconstructed_signal": None,
            "prediction": None,
            "confidence": None,
            "logits": None,
            "alarm": self._alarm_from_packet(packet),
            "session_state": None,
        }

    def receive(self, packet_or_bytes: Union[SemanticPacket, bytes, bytearray]) -> Dict[str, Any]:
        if isinstance(packet_or_bytes, (bytes, bytearray)):
            packet_bytes = bytes(packet_or_bytes)
            packet = bytes_to_packet(packet_bytes, verify_checksum=True)
            packet_nbytes = len(packet_bytes)
        else:
            packet = packet_or_bytes
            packet_nbytes = estimate_object_nbytes(packet.to_dict())

        self._validate_packet(packet)

        base_result = self._make_base_result(packet, packet_nbytes=packet_nbytes)

        # 去重
        if self.replay_guard.seen(packet):
            self.session_tracker.mark_duplicate(packet.header)
            if not self.allow_duplicate_window:
                base_result["status"] = "duplicate_skipped"
                base_result["duplicate"] = True
                base_result["used_path"] = "deduplicate:skip"
                base_result["session_state"] = self.session_tracker.snapshot_by_header(packet.header)
                return base_result

        mode = packet.payload_mode

        # M0: idle 段摘要
        if mode == "M0":
            base_result["used_path"] = "summary:idle"
            base_result["session_state"] = self.session_tracker.update(packet, base_result)
            return base_result

        # M1: only anchors
        if mode == "M1":
            base_result["used_path"] = "summary:anchor_only"
            base_result["session_state"] = self.session_tracker.update(packet, base_result)
            return base_result

        # M2/M3: 尝试 latent/raw/MyIDM
        latent = self._extract_latent_from_packet(packet)
        signal = self._extract_signal_from_packet(packet)

        # 如果没 raw_signal，但有 latent，可先通过 reference decoder 恢复 signal
        if signal is None and latent is not None:
            signal = self._decode_latent_with_reference(latent)

        # 优先调用 MyIDM
        bridge_result = self.bridge.invoke(packet, latent=latent, signal=signal)

        if bridge_result is not None:
            base_result["used_path"] = bridge_result.get("used_path")
            base_result["receiver_output"] = bridge_result.get("receiver_output")
            base_result["reconstructed_signal"] = bridge_result.get("reconstructed_signal")
            base_result["prediction"] = bridge_result.get("prediction")
            base_result["confidence"] = bridge_result.get("confidence")
            base_result["logits"] = bridge_result.get("logits")
            base_result["session_state"] = self.session_tracker.update(packet, base_result)
            return base_result

        # 回退 1：只有 reference decoder
        if signal is not None:
            base_result["used_path"] = "fallback:signal_only"
            base_result["reconstructed_signal"] = signal
            base_result["session_state"] = self.session_tracker.update(packet, base_result)
            return base_result

        # 回退 2：只有语义
        base_result["used_path"] = "fallback:semantic_only"
        base_result["session_state"] = self.session_tracker.update(packet, base_result)
        return base_result

    def receive_file(self, packet_file: Union[str, Path]) -> Dict[str, Any]:
        packet_file = Path(packet_file)
        with open(packet_file, "rb") as f:
            packet_bytes = f.read()
        return self.receive(packet_bytes)

    def batch_receive_dir(self, packet_dir: Union[str, Path], suffix: str = ".bin") -> List[Dict[str, Any]]:
        packet_dir = Path(packet_dir)
        files = sorted([p for p in packet_dir.rglob(f"*{suffix}") if p.is_file()])
        results = []
        for p in files:
            results.append(self.receive_file(p))
        return results


class MyIDMReceiverAdapter(IndustrialSemanticReceiverEngine):
    """
    为兼容之前脚本命名，保留这个类名。
    """
    def __call__(self, packet_or_bytes: Union[SemanticPacket, bytes, bytearray]) -> Dict[str, Any]:
        return self.receive(packet_or_bytes)