import importlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .model import SemanticSenderModel, load_sender_checkpoint
from .protocol import SemanticPacket, packet_to_bytes, bytes_to_packet
from .utils import (
    compute_signal_anchors,
    ensure_signal_ch_first,
    get_logger,
    normalized_entropy,
    recursive_to_numpy,
)


def symmetric_quantize_ndarray(arr: np.ndarray, num_bits: int = 8) -> Tuple[np.ndarray, float]:
    if num_bits < 2:
        raise ValueError("num_bits 必须 >= 2")
    arr = np.asarray(arr, dtype=np.float32)
    qmax = (1 << (num_bits - 1)) - 1
    max_abs = float(np.max(np.abs(arr)))
    max_abs = max(max_abs, 1e-8)
    scale = max_abs / qmax
    q = np.clip(np.round(arr / scale), -qmax, qmax).astype(np.int16)
    return q, float(scale)


def symmetric_dequantize_ndarray(q: np.ndarray, scale: Union[float, np.ndarray]) -> np.ndarray:
    return np.asarray(q, dtype=np.float32) * np.asarray(scale, dtype=np.float32)


class WearStateManager:
    """
    磨损状态时间平滑器
    """

    def __init__(self, ema_alpha: float = 0.60, promotion_threshold: float = 0.55):
        self.ema_alpha = float(ema_alpha)
        self.promotion_threshold = float(promotion_threshold)
        self.reset()

    def reset(self):
        self.wear_ema = None
        self.stage = None

    def update(self, wear_probs: np.ndarray) -> Tuple[int, np.ndarray]:
        wear_probs = np.asarray(wear_probs, dtype=np.float32)

        if self.wear_ema is None:
            self.wear_ema = wear_probs.copy()
        else:
            self.wear_ema = self.ema_alpha * self.wear_ema + (1.0 - self.ema_alpha) * wear_probs

        pred = int(np.argmax(self.wear_ema))

        if self.stage is None:
            self.stage = pred
            return self.stage, self.wear_ema.copy()

        if pred == self.stage:
            return self.stage, self.wear_ema.copy()

        if pred > self.stage:
            if self.wear_ema[pred] >= self.promotion_threshold:
                self.stage = pred
            return self.stage, self.wear_ema.copy()

        if wear_probs[pred] >= 0.95 and self.wear_ema[self.stage] < 0.25:
            self.stage = pred

        return self.stage, self.wear_ema.copy()


class IndustrialSemanticSenderEngine:
    """
    发送端：
    - 语义识别
    - 占空比估计
    - 传输模式选择
    - latent 量化与 packet 封装
    - idle 段聚合
    """

    def __init__(
        self,
        config: Dict[str, Any],
        sender_model: Optional[SemanticSenderModel] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config
        self.logger = get_logger("IndustrialSemanticSenderEngine")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if sender_model is not None:
            self.model = sender_model.to(self.device).eval()
            self.sender_ckpt = None
        elif checkpoint_path:
            self.model, self.sender_ckpt = load_sender_checkpoint(checkpoint_path, device=self.device)
        else:
            self.model = SemanticSenderModel(self.cfg["model"]).to(self.device).eval()
            self.sender_ckpt = None

        norm_cfg = self.cfg["dataset"].get("normalization", {})
        self.norm_enabled = bool(norm_cfg.get("enabled", True))
        self.norm_id = norm_cfg.get("id", "global-zscore-v1")
        self.norm_mean = np.asarray(norm_cfg.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1, 1)
        self.norm_std = np.asarray(norm_cfg.get("std", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1, 1)
        self.norm_std = np.clip(self.norm_std, 1e-8, None)

        self.runtime_cfg = self.cfg.get("runtime", {})
        self.policy_cfg = self.cfg.get("policy", {})
        self.dataset_cfg = self.cfg.get("dataset", {})

        self.window_length = int(self.dataset_cfg.get("window_length", 1024))
        self.sampling_rate = int(self.dataset_cfg.get("sampling_rate", 10240))

        self.state_manager = WearStateManager(
            ema_alpha=self.policy_cfg.get("ema_alpha", 0.60),
            promotion_threshold=self.policy_cfg.get("promotion_threshold", 0.55),
        )

        self.window_counter = 0
        self.pending_idle_segment = None

    def reset_sequence_state(self):
        """
        在切换文件 / 刀具 / run 时调用，避免状态串台。
        """
        self.state_manager.reset()
        self.pending_idle_segment = None

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        if not self.norm_enabled:
            return signal.astype(np.float32)
        return ((signal - self.norm_mean) / self.norm_std).astype(np.float32)

    def _estimate_occupancy_ratio(self, signal_raw: np.ndarray) -> Tuple[float, List[int], List[float]]:
        """
        基于微窗 RMS 粗估切削占空比，用来辅助 contact 判定。
        """
        splits = int(self.policy_cfg.get("occupancy_splits", 4))
        energy_ratio = float(self.policy_cfg.get("occupancy_energy_ratio", 0.35))

        signal_raw = ensure_signal_ch_first(signal_raw, expected_channels=3).astype(np.float32)
        total_len = signal_raw.shape[1]
        split_len = total_len // splits

        if split_len <= 0:
            return 0.0, [0] * splits, [0.0] * splits

        energies = []
        for i in range(splits):
            start = i * split_len
            end = total_len if i == splits - 1 else (i + 1) * split_len
            seg = signal_raw[:, start:end]
            rms = np.sqrt(np.mean(seg ** 2) + 1e-12)
            energies.append(float(rms))

        max_e = max(energies) if energies else 0.0
        mean_e = float(np.mean(energies)) if energies else 0.0

        if max_e < 1e-8:
            mask = [0 for _ in energies]
            return 0.0, mask, energies

        thr = max(max_e * energy_ratio, mean_e * 0.95)
        mask = [1 if e >= thr else 0 for e in energies]
        occ = float(np.mean(mask)) if mask else 0.0

        return occ, mask, energies

    def _decide_contact_state(self, contact_probs: np.ndarray, occupancy_ratio: float) -> Tuple[str, int]:
        idle_conf_thr = float(self.policy_cfg.get("idle_conf_threshold", 0.90))
        idle_occ_thr = float(self.policy_cfg.get("idle_occupancy_threshold", 0.20))
        cut_occ_thr = float(self.policy_cfg.get("cutting_occupancy_threshold", 0.80))

        idle_p = float(contact_probs[0])

        if idle_p >= idle_conf_thr and occupancy_ratio <= idle_occ_thr:
            return "idle", 0

        if occupancy_ratio >= cut_occ_thr:
            return "cutting", 2

        return "mixed", 1

    def _decide_transmission_mode(
        self,
        contact_state: str,
        wear_state: Optional[str],
        wear_probs: np.ndarray,
        confidence: float,
        uncertainty: float,
        ood_score: float,
    ) -> str:
        uncertainty_thr = float(self.policy_cfg.get("uncertainty_threshold", 0.55))
        ood_thr = float(self.policy_cfg.get("ood_threshold", 0.60))
        accel_alarm_thr = float(self.policy_cfg.get("acceleration_alarm_threshold", 0.75))
        allow_anchor_only = bool(self.policy_cfg.get("allow_anchor_only_for_steady", False))
        anchor_only_conf_thr = float(self.policy_cfg.get("anchor_only_conf_threshold", 0.98))

        if contact_state == "idle":
            return "M0"

        if uncertainty >= uncertainty_thr or ood_score >= ood_thr:
            return "M3"

        if wear_state == "accelerating" and float(wear_probs[2]) >= accel_alarm_thr:
            return "M3"

        if allow_anchor_only and wear_state == "steady" and confidence >= anchor_only_conf_thr:
            return "M1"

        return "M2"

    def _build_header(self, metadata: Optional[Dict[str, Any]], payload_mode: str) -> Dict[str, Any]:
        metadata = metadata or {}
        self.window_counter += 1

        return {
            "protocol_version": "ISC-TWM-1.0",
            "session_id": metadata.get("session_id", self.runtime_cfg.get("session_id", "SESSION-001")),
            "machine_id": metadata.get("machine_id", self.runtime_cfg.get("machine_id", "M001")),
            "spindle_id": metadata.get("spindle_id", self.runtime_cfg.get("spindle_id", "S001")),
            "tool_id": metadata.get("tool_id", self.runtime_cfg.get("tool_id", "T001")),
            "run_id": metadata.get("run_id", "RUN-001"),
            "window_id": metadata.get("window_id", self.window_counter),
            "timestamp_start": float(metadata.get("timestamp_start", time.time())),
            "timestamp_end": float(metadata.get("timestamp_end", time.time())),
            "sampling_rate": int(metadata.get("sampling_rate", self.sampling_rate)),
            "window_length": int(metadata.get("window_length", self.window_length)),
            "axis_order": "XYZ",
            "encoder_version": self.runtime_cfg.get("encoder_version", "semtx-1.0"),
            "normalization_id": self.norm_id,
            "payload_mode": payload_mode,
            "checksum": "",
        }

    def _update_idle_segment(self, header: Dict[str, Any], anchors: Dict[str, float], semantic: Dict[str, Any]):
        if self.pending_idle_segment is None:
            self.pending_idle_segment = {
                "start_header": header,
                "end_header": header,
                "count": 1,
                "anchors_sum": {k: float(v) for k, v in anchors.items()},
                "semantic": semantic.copy(),
            }
            return

        self.pending_idle_segment["end_header"] = header
        self.pending_idle_segment["count"] += 1
        for k, v in anchors.items():
            self.pending_idle_segment["anchors_sum"][k] = self.pending_idle_segment["anchors_sum"].get(k, 0.0) + float(v)

    def _flush_idle_segment(self, return_bytes: bool = False):
        if self.pending_idle_segment is None:
            return None

        seg = self.pending_idle_segment
        start_header = seg["start_header"]
        end_header = seg["end_header"]
        count = seg["count"]
        anchors_avg = {k: v / max(count, 1) for k, v in seg["anchors_sum"].items()}

        header = start_header.copy()
        header["payload_mode"] = "M0"
        header["window_id"] = f"{start_header['window_id']}-{end_header['window_id']}"
        header["timestamp_end"] = end_header["timestamp_end"]

        semantic = seg["semantic"].copy()
        semantic["contact_state"] = "idle"
        semantic["mapped_class"] = 0

        payload = {
            "idle_window_count": count,
            "idle_duration_sec": float(header["timestamp_end"] - header["timestamp_start"]),
            "summary_type": "idle_segment_summary",
        }

        packet = SemanticPacket(
            header=header,
            semantic=semantic,
            anchors=anchors_avg,
            payload=payload,
        )

        self.pending_idle_segment = None
        return packet_to_bytes(packet) if return_bytes else packet

    @torch.no_grad()
    def analyze_window(self, signal: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        signal_raw = ensure_signal_ch_first(signal, expected_channels=3).astype(np.float32)
        if signal_raw.shape[1] != self.window_length:
            raise ValueError(f"输入窗口长度必须为 {self.window_length}，当前为 {signal_raw.shape[1]}")

        signal_norm = self._normalize(signal_raw)
        x = torch.from_numpy(signal_norm).unsqueeze(0).float().to(self.device)

        outputs = self.model(x)
        contact_probs = outputs["contact_probs"][0].detach().cpu().numpy()
        wear_probs = outputs["wear_probs"][0].detach().cpu().numpy()
        mapped_probs = outputs["mapped_probs"][0].detach().cpu().numpy()
        latent = outputs["latent"][0].detach().cpu().numpy()

        occupancy_ratio, occupancy_mask, occupancy_energies = self._estimate_occupancy_ratio(signal_raw)
        contact_state, contact_label = self._decide_contact_state(contact_probs, occupancy_ratio)

        if contact_state == "idle":
            wear_stage = None
            wear_stage_name = None
            confidence = float(contact_probs[0])
            mapped_class = 0
        else:
            wear_stage, wear_ema = self.state_manager.update(wear_probs)
            wear_stage_name = ["initial", "steady", "accelerating"][wear_stage]
            confidence = float(wear_probs[wear_stage])
            mapped_class = wear_stage + 1

        contact_ent = normalized_entropy(contact_probs)
        wear_ent = normalized_entropy(wear_probs)
        mapped_ent = normalized_entropy(mapped_probs)
        uncertainty = float(max(contact_ent, wear_ent if contact_state != "idle" else 0.0, mapped_ent))
        ood_score = float(max(uncertainty, 1.0 - float(np.max(mapped_probs))))

        mode = self._decide_transmission_mode(
            contact_state=contact_state,
            wear_state=wear_stage_name,
            wear_probs=wear_probs,
            confidence=confidence,
            uncertainty=uncertainty,
            ood_score=ood_score,
        )

        anchors = compute_signal_anchors(signal_raw, sampling_rate=self.sampling_rate)

        semantic = {
            "contact_state": contact_state,
            "contact_probs": contact_probs.astype(np.float32),
            "wear_state": wear_stage_name if wear_stage_name is not None else "NA",
            "wear_probs": wear_probs.astype(np.float32),
            "mapped_class": int(mapped_class),
            "mapped_probs": mapped_probs.astype(np.float32),
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "ood_score": float(ood_score),
            "occupancy_ratio": float(occupancy_ratio),
            "occupancy_mask": np.asarray(occupancy_mask, dtype=np.int8),
            "occupancy_energies": np.asarray(occupancy_energies, dtype=np.float32),
        }

        return {
            "signal_raw": signal_raw,
            "signal_norm": signal_norm,
            "latent": latent,
            "anchors": anchors,
            "semantic": semantic,
            "mode": mode,
            "metadata": metadata or {},
        }

    def _build_packet_from_analysis(self, analysis: Dict[str, Any]) -> SemanticPacket:
        mode = analysis["mode"]
        signal_raw = analysis["signal_raw"]
        latent = analysis["latent"]
        anchors = analysis["anchors"]
        semantic = analysis["semantic"]
        metadata = analysis["metadata"]

        header = self._build_header(metadata, payload_mode=mode)

        payload = {}
        if mode == "M0":
            payload = {
                "summary_type": "single_idle_window",
            }

        elif mode == "M1":
            payload = {
                "summary_type": "anchor_only",
            }

        elif mode == "M2":
            bits = int(self.policy_cfg.get("normal_bits", 8))
            q, scale = symmetric_quantize_ndarray(latent, num_bits=bits)
            payload = {
                "latent_q": q,
                "latent_scale": np.asarray(scale, dtype=np.float32),
                "latent_bits": bits,
                "latent_shape": np.asarray(q.shape, dtype=np.int32),
            }

        elif mode == "M3":
            send_raw = bool(self.policy_cfg.get("send_raw_on_alarm", True))
            bits = int(self.policy_cfg.get("alarm_bits", 12))
            q, scale = symmetric_quantize_ndarray(latent, num_bits=bits)
            payload = {
                "latent_q": q,
                "latent_scale": np.asarray(scale, dtype=np.float32),
                "latent_bits": bits,
                "latent_shape": np.asarray(q.shape, dtype=np.int32),
                "alarm": True,
            }
            if send_raw:
                payload["raw_signal"] = signal_raw.astype(np.float32)

        packet = SemanticPacket(
            header=header,
            semantic=semantic,
            anchors=anchors,
            payload=payload,
        )
        return packet

    def process_window(
        self,
        signal: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        return_bytes: bool = False,
    ) -> Dict[str, Any]:
        analysis = self.analyze_window(signal, metadata=metadata)
        packets = []

        # 遇到非 idle 先 flush 之前累积的 idle segment
        if analysis["semantic"]["contact_state"] != "idle":
            flushed = self._flush_idle_segment(return_bytes=return_bytes)
            if flushed is not None:
                packets.append(flushed)

        packet = self._build_packet_from_analysis(analysis)

        if analysis["mode"] == "M0":
            self._update_idle_segment(packet.header, packet.anchors, packet.semantic)
        else:
            packets.append(packet_to_bytes(packet) if return_bytes else packet)

        return {
            "packets": packets,
            "analysis": analysis,
        }

    def finalize(self, return_bytes: bool = False) -> List[Union[SemanticPacket, bytes]]:
        flushed = self._flush_idle_segment(return_bytes=return_bytes)
        return [flushed] if flushed is not None else []


class MyIDMReceiverAdapter:

    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        self.cfg = config
        self.logger = get_logger("MyIDMReceiverAdapter")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.receiver_cfg = self.cfg.get("receiver", {})
        self.method_candidates = self.receiver_cfg.get(
            "method_candidates",
            ["decode_packet", "decode", "reconstruct", "forward", "inference"],
        )

        self.receiver = self._load_receiver_instance()

        self.reference_sender = None
        if self.receiver_cfg.get("use_reference_decoder_fallback", True):
            ref_ckpt = self.receiver_cfg.get("reference_checkpoint", "")
            if ref_ckpt:
                try:
                    self.reference_sender, _ = load_sender_checkpoint(ref_ckpt, device=self.device)
                    self.logger.info(f"已加载 reference decoder checkpoint: {ref_ckpt}")
                except Exception as e:
                    self.logger.warning(f"reference decoder 加载失败: {e}")

    def _load_receiver_instance(self):
        # 默认直接用我们新增的桥接类
        module_name = self.receiver_cfg.get("module", "") or "industrial_semantic.myidm_receiver"
        class_name = self.receiver_cfg.get("class_name", "") or "MyIDMReceiver"
        checkpoint = self.receiver_cfg.get("checkpoint", "")

        init_kwargs = dict(self.receiver_cfg.get("init_kwargs", {}) or {})
        init_kwargs.setdefault("device", self.device)
        if checkpoint and "checkpoint" not in init_kwargs:
            init_kwargs["checkpoint"] = checkpoint

        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            obj = cls(**init_kwargs)

            if isinstance(obj, torch.nn.Module):
                obj.to(self.device)
                obj.eval()

            self.logger.info(f"已加载接收端: {module_name}.{class_name}")
            return obj
        except Exception as e:
            self.logger.warning(f"接收端加载失败: {e}")
            return None

    def _normalize_receiver_output(self, out, method_name: str):
        out = recursive_to_numpy(out)
        if isinstance(out, dict):
            out.setdefault("used_path", f"myidm:{method_name}")
            return out
        return {
            "used_path": f"myidm:{method_name}",
            "receiver_output": out,
        }

    def _try_call_receiver(self, packet: SemanticPacket, latent: Optional[np.ndarray], raw_signal: Optional[np.ndarray]):
        if self.receiver is None:
            return None

        latent_tensor = None
        if latent is not None:
            latent_tensor = torch.from_numpy(latent).unsqueeze(0).float().to(self.device)

        raw_tensor = None
        if raw_signal is not None:
            raw_tensor = torch.from_numpy(raw_signal).unsqueeze(0).float().to(self.device)

        attempts = []
        for method_name in self.method_candidates:
            if hasattr(self.receiver, method_name):
                method = getattr(self.receiver, method_name)
                attempts.extend([
                    (method_name, method, packet),
                    (method_name, method, packet.to_dict()),
                    (method_name, method, packet.payload),
                    (method_name, method, {"packet": packet.to_dict(), "latent": latent_tensor, "raw_signal": raw_tensor}),
                ])
                if latent_tensor is not None:
                    attempts.append((method_name, method, latent_tensor))
                if raw_tensor is not None:
                    attempts.append((method_name, method, raw_tensor))

        if callable(self.receiver):
            if latent_tensor is not None:
                attempts.append(("__call__", self.receiver, latent_tensor))
            if raw_tensor is not None:
                attempts.append(("__call__", self.receiver, raw_tensor))

        last_err = None
        for name, fn, arg in attempts:
            try:
                out = fn(arg)
                return self._normalize_receiver_output(out, name)
            except Exception as e:
                last_err = e

        if last_err is not None:
            self.logger.warning(f"receiver 调用失败，最后一次错误: {last_err}")
        return None

    def receive(self, packet_or_bytes: Union[SemanticPacket, bytes]) -> Dict[str, Any]:
        packet = bytes_to_packet(packet_or_bytes) if isinstance(packet_or_bytes, (bytes, bytearray)) else packet_or_bytes

        mode = packet.payload_mode
        semantic = packet.semantic
        anchors = packet.anchors
        payload = packet.payload

        if mode == "M0":
            return {
                "used_path": "summary",
                "packet_mode": mode,
                "semantic": semantic,
                "anchors": anchors,
                "payload": payload,
                "reconstructed_signal": None,
            }

        if mode == "M1":
            return {
                "used_path": "anchor_only",
                "packet_mode": mode,
                "semantic": semantic,
                "anchors": anchors,
                "payload": payload,
                "reconstructed_signal": None,
            }

        raw_signal = payload.get("raw_signal", None)
        latent = None
        if "latent_q" in payload and "latent_scale" in payload:
            latent = symmetric_dequantize_ndarray(payload["latent_q"], payload["latent_scale"])
        elif "latent" in payload:
            latent = np.asarray(payload["latent"], dtype=np.float32)

        # 1) 优先尝试 MyIDM 接收端
        myidm_result = self._try_call_receiver(packet, latent=latent, raw_signal=raw_signal)
        if myidm_result is not None:
            myidm_result.setdefault("packet_mode", mode)
            myidm_result.setdefault("semantic", semantic)
            myidm_result.setdefault("anchors", anchors)
            return myidm_result

        # 2) fallback: reference decoder
        if latent is not None and self.reference_sender is not None:
            try:
                latent_tensor = torch.from_numpy(latent).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    recon = self.reference_sender.decode_latent(latent_tensor)
                return {
                    "used_path": "reference_decoder",
                    "packet_mode": mode,
                    "semantic": semantic,
                    "anchors": anchors,
                    "reconstructed_signal": recon[0].detach().cpu().numpy(),
                }
            except Exception as e:
                self.logger.warning(f"reference decoder 解码失败: {e}")

        # 3) fallback: raw passthrough
        if raw_signal is not None:
            return {
                "used_path": "raw_passthrough",
                "packet_mode": mode,
                "semantic": semantic,
                "anchors": anchors,
                "reconstructed_signal": np.asarray(raw_signal, dtype=np.float32),
            }

        # 4) 最后只保留语义
        return {
            "used_path": "semantic_only",
            "packet_mode": mode,
            "semantic": semantic,
            "anchors": anchors,
            "reconstructed_signal": None,
        }
