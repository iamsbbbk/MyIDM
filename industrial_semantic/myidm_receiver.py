import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_semantic.dataset import _ensure_image_ch_first
from industrial_semantic.utils import recursive_to_numpy
from industrial_semantic.cs_operator import OperatorFactory


class MyIDMReceiver:
    """
    针对当前 MyIDM 真实结构的精准接收器。

    关键修正：
    - 不再错误地把 SD1.5 UNet 的输入输出通道从 4 改成 8
    - 因为当前 MyIDM Step.body 中实际送入 unet 的是:
        F.pixel_unshuffle(x, 2)
      对单通道 x 而言，输入应为 4 通道

    支持：
    - measurement / y
    - y_rgb
    - raw_image / raw_input
    - 从 raw_image 模拟测量再走 MyIDM
    """

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str = "",
        sd15_path: str = "./sd15",
        num_steps: int = 8,
        img_h: int = 32,
        img_w: int = 32,
        cs_ratio: float = 0.25,
        measurement_dim: int = 0,
        operator_seed: int = 2026,
        use_amp: bool = True,
        simulate_measurement_from_raw: bool = True,
        reconstruct_rgb_channelwise: bool = True,
        auto_normalize_image: bool = True,
        verbose: bool = False,
    ):
        self.device = device if (torch.cuda.is_available() or str(device) == "cpu") else "cpu"
        self.checkpoint = checkpoint
        self.sd15_path = sd15_path
        self.num_steps = int(num_steps)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.cs_ratio = float(cs_ratio)
        self.measurement_dim = int(measurement_dim)
        self.operator_seed = int(operator_seed)
        self.use_amp = bool(use_amp and str(self.device).startswith("cuda"))
        self.simulate_measurement_from_raw = bool(simulate_measurement_from_raw)
        self.reconstruct_rgb_channelwise = bool(reconstruct_rgb_channelwise)
        self.auto_normalize_image = bool(auto_normalize_image)
        self.verbose = bool(verbose)

        self.operator_factory = OperatorFactory(device=self.device, dtype=torch.float32)
        self.myidm_model_module = self._import_root_module("model")
        self.net = self._build_net()

    def _log(self, *msg):
        if self.verbose:
            print("[MyIDMReceiver]", *msg)

    def _import_root_module(self, module_name: str):
        try:
            mod = importlib.import_module(module_name)
            self._log(f"成功导入根模块: {module_name}")
            return mod
        except Exception as e:
            self._log(f"导入根模块失败: {module_name}, err={e}")
            return None

    def _load_sd15_unet(self):
        try:
            from diffusers import UNet2DConditionModel
        except Exception as e:
            raise ImportError(f"未能导入 diffusers.UNet2DConditionModel: {e}")

        sd15_path = Path(self.sd15_path)
        if not sd15_path.exists():
            raise FileNotFoundError(f"sd15 路径不存在: {sd15_path}")

        last_err = None

        try:
            unet = UNet2DConditionModel.from_pretrained(
                str(sd15_path),
                subfolder="unet",
                local_files_only=True,
            )
            self._log("从 sd15/unet 成功加载 UNet")
            return unet
        except Exception as e:
            last_err = e

        try:
            unet = UNet2DConditionModel.from_pretrained(
                str(sd15_path),
                local_files_only=True,
            )
            self._log("从 sd15 根目录成功加载 UNet")
            return unet
        except Exception as e:
            last_err = e

        raise RuntimeError(f"加载 SD15 UNet 失败: {last_err}")

    def _load_checkpoint_if_exists(self, model: nn.Module):
        if not self.checkpoint:
            self._log("未提供 MyIDM checkpoint，将使用当前初始化权重")
            return model

        ckpt_path = Path(self.checkpoint)
        if not ckpt_path.exists():
            self._log(f"checkpoint 不存在: {ckpt_path}")
            return model

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state = ckpt
            if isinstance(ckpt, dict):
                for k in ["model_state_dict", "state_dict", "model", "net", "ema_state_dict"]:
                    if k in ckpt and isinstance(ckpt[k], dict):
                        state = ckpt[k]
                        break
            missing, unexpected = model.load_state_dict(state, strict=False)
            self._log(f"checkpoint 加载完成: {ckpt_path}")
            self._log(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        except Exception as e:
            self._log(f"checkpoint 加载失败: {e}")

        return model

    def _dry_run_check(self, net: nn.Module):
        """
        构建后做一次干跑，提前暴露 shape / channel 问题。
        """
        try:
            operator = self._get_operator(
                operator_meta={
                    "img_h": self.img_h,
                    "img_w": self.img_w,
                    "ratio": self.cs_ratio,
                    "m": self.measurement_dim,
                    "seed": self.operator_seed,
                    "op_type": "gaussian_flatten",
                }
            )
            y = torch.zeros(1, operator.m, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                _ = net(y, operator.A, operator.AT, use_amp_=self.use_amp)
            self._log("MyIDM Net 干跑检查通过")
            return True
        except Exception as e:
            self._log(f"MyIDM Net 干跑失败: {e}")
            return False

    def _build_net(self):
        if self.myidm_model_module is None:
            self._log("根目录 model.py 未成功导入，无法构建 MyIDM Net")
            return None

        if not hasattr(self.myidm_model_module, "Net"):
            self._log("根目录 model.py 中未找到 Net 类")
            return None

        try:
            # 关键修正：这里不再做错误的 8->8 通道适配
            unet = self._load_sd15_unet()

            NetCls = getattr(self.myidm_model_module, "Net")
            net = NetCls(T=self.num_steps, unet=unet)
            net = self._load_checkpoint_if_exists(net)

            if isinstance(net, nn.Module):
                net = net.to(self.device).eval()

            ok = self._dry_run_check(net)
            if not ok:
                self._log("MyIDM Net 干跑失败，将保留 net 但后续可能 fallback")
            else:
                self._log("MyIDM Net 构建成功")

            return net
        except Exception as e:
            self._log(f"MyIDM Net 构建失败: {e}")
            return None

    def _normalize_packet_like(self, packet_like: Any) -> Dict[str, Any]:
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
            if any(k in packet_like for k in ["header", "semantic", "anchors", "payload"]):
                return {
                    "header": packet_like.get("header", {}),
                    "semantic": packet_like.get("semantic", {}),
                    "anchors": packet_like.get("anchors", {}),
                    "payload": packet_like.get("payload", {}),
                }
            return {
                "header": {},
                "semantic": {},
                "anchors": {},
                "payload": packet_like,
            }

        return {"header": {}, "semantic": {}, "anchors": {}, "payload": {}}

    def _extract_packet_fields(self, packet_like: Any):
        packet = self._normalize_packet_like(packet_like)
        header = packet.get("header", {})
        semantic = packet.get("semantic", {})
        payload = packet.get("payload", {})

        data_modality = (
            semantic.get("data_modality")
            or payload.get("data_modality")
            or header.get("data_modality")
            or "unknown"
        )

        operator_meta = payload.get("operator_meta", {}) or {}

        y = payload.get("y", None)
        if y is None:
            y = payload.get("measurement", None)

        y_rgb = payload.get("y_rgb", None)
        if y_rgb is None:
            y_rgb = payload.get("measurements_rgb", None)

        raw_input = payload.get("raw_input", None)
        if raw_input is None:
            raw_input = payload.get("raw_image", None)
        if raw_input is None:
            raw_input = payload.get("raw_signal", None)

        return packet, data_modality, operator_meta, y, y_rgb, raw_input

    def _get_operator(self, operator_meta: Optional[Dict[str, Any]] = None):
        meta = dict(operator_meta or {})
        meta.setdefault("img_h", self.img_h)
        meta.setdefault("img_w", self.img_w)
        meta.setdefault("ratio", self.cs_ratio)
        meta.setdefault("m", self.measurement_dim)
        meta.setdefault("seed", self.operator_seed)
        meta.setdefault("op_type", "gaussian_flatten")
        return self.operator_factory.get_from_meta(meta)

    def _normalize_image_range(self, image: np.ndarray):
        image = np.asarray(image, dtype=np.float32)
        if not self.auto_normalize_image:
            return image
        if image.max() > 1.5:
            image = image / 255.0
        return image

    @torch.no_grad()
    def _reconstruct_single_channel_from_y(self, y: np.ndarray, operator) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("MyIDM Net 尚未成功构建，无法执行重构")

        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[None, :]

        y_tensor = torch.from_numpy(y).float().to(self.device)
        x_hat = self.net(y_tensor, operator.A, operator.AT, use_amp_=self.use_amp)
        x_hat = x_hat.detach().cpu().numpy()  # [B,1,H,W]
        return x_hat[0, 0]

    @torch.no_grad()
    def _simulate_y_from_single_channel(self, x_1chw: torch.Tensor, operator):
        y = operator.A(x_1chw)
        return y.detach().cpu().numpy()[0]

    def _reconstruct_rgb_from_measurements(self, y_rgb: Any, operator_meta: Optional[Dict[str, Any]] = None):
        if y_rgb is None:
            return None

        if isinstance(y_rgb, list):
            channels = [np.asarray(v, dtype=np.float32).reshape(-1) for v in y_rgb]
        else:
            y_rgb = np.asarray(y_rgb, dtype=np.float32)
            if y_rgb.ndim == 1:
                channels = [y_rgb]
            elif y_rgb.ndim == 2:
                channels = [y_rgb[i] for i in range(y_rgb.shape[0])]
            else:
                raise ValueError(f"y_rgb 形状不支持: {y_rgb.shape}")

        if len(channels) == 1:
            operator = self._get_operator(operator_meta=operator_meta)
            gray = self._reconstruct_single_channel_from_y(channels[0], operator)
            return gray[None, :, :]

        recons = []
        for y_c in channels:
            operator = self._get_operator(operator_meta=operator_meta)
            rec_c = self._reconstruct_single_channel_from_y(y_c, operator)
            recons.append(rec_c)
        return np.stack(recons, axis=0)

    def _simulate_measurement_then_reconstruct(self, raw_image: np.ndarray, operator_meta: Optional[Dict[str, Any]] = None):
        img = _ensure_image_ch_first(raw_image)
        img = self._normalize_image_range(img)

        channels = img.shape[0]
        operator = self._get_operator(operator_meta=operator_meta)

        if channels == 1 or not self.reconstruct_rgb_channelwise:
            x = torch.from_numpy(img[:1][None, ...]).float().to(self.device)  # [1,1,H,W]
            y = self._simulate_y_from_single_channel(x, operator)
            rec = self._reconstruct_single_channel_from_y(y, operator)
            return rec[None, :, :], y[None, :]

        ys = []
        recons = []
        for c in range(channels):
            x = torch.from_numpy(img[c:c+1][None, ...]).float().to(self.device)  # [1,1,H,W]
            y = self._simulate_y_from_single_channel(x, operator)
            rec = self._reconstruct_single_channel_from_y(y, operator)
            ys.append(y)
            recons.append(rec)

        return np.stack(recons, axis=0), np.stack(ys, axis=0)

    def decode_packet(self, packet_like: Any) -> Dict[str, Any]:
        packet, data_modality, operator_meta, y, y_rgb, raw_input = self._extract_packet_fields(packet_like)

        semantic = packet.get("semantic", {})
        anchors = packet.get("anchors", {})
        packet_mode = packet.get("header", {}).get("payload_mode", "NA")

        # 1) 优先：sender 已发送 y_rgb
        if y_rgb is not None:
            try:
                recon = self._reconstruct_rgb_from_measurements(y_rgb, operator_meta=operator_meta)
                return {
                    "used_path": "myidm:measurement_rgb",
                    "packet_mode": packet_mode,
                    "semantic": semantic,
                    "anchors": anchors,
                    "reconstructed_data": recon,
                    "reconstructed_signal": recon,
                    "receiver_raw_output": None,
                    "data_modality": data_modality,
                }
            except Exception as e:
                self._log(f"measurement_rgb 解码失败: {e}")

        # 2) sender 已发送单通道 y
        if y is not None:
            try:
                operator = self._get_operator(operator_meta=operator_meta)
                gray = self._reconstruct_single_channel_from_y(np.asarray(y, dtype=np.float32), operator)
                recon = gray[None, :, :]
                return {
                    "used_path": "myidm:measurement",
                    "packet_mode": packet_mode,
                    "semantic": semantic,
                    "anchors": anchors,
                    "reconstructed_data": recon,
                    "reconstructed_signal": recon,
                    "receiver_raw_output": None,
                    "data_modality": data_modality,
                }
            except Exception as e:
                self._log(f"measurement 解码失败: {e}")

        # 3) sender 还没训练好时，用 raw_image 在接收端本地模拟测量再重构
        if raw_input is not None and self.simulate_measurement_from_raw and self.net is not None:
            try:
                if data_modality == "image2d":
                    recon, sim_y = self._simulate_measurement_then_reconstruct(raw_input, operator_meta=operator_meta)
                    return {
                        "used_path": "myidm:simulate_from_raw_image",
                        "packet_mode": packet_mode,
                        "semantic": semantic,
                        "anchors": anchors,
                        "reconstructed_data": recon,
                        "reconstructed_signal": recon,
                        "simulated_measurements": sim_y,
                        "receiver_raw_output": None,
                        "data_modality": data_modality,
                    }
            except Exception as e:
                self._log(f"simulate_from_raw_image 失败: {e}")

        # 4) fallback
        if raw_input is not None:
            return {
                "used_path": "packet_raw_passthrough",
                "packet_mode": packet_mode,
                "semantic": semantic,
                "anchors": anchors,
                "reconstructed_data": np.asarray(raw_input),
                "reconstructed_signal": np.asarray(raw_input),
                "receiver_raw_output": None,
                "data_modality": data_modality,
            }

        return {
            "used_path": "semantic_only",
            "packet_mode": packet_mode,
            "semantic": semantic,
            "anchors": anchors,
            "reconstructed_data": None,
            "reconstructed_signal": None,
            "receiver_raw_output": None,
            "data_modality": data_modality,
        }

    def decode(self, x: Any) -> Dict[str, Any]:
        return self.decode_packet(x)

    def reconstruct(self, x: Any) -> Dict[str, Any]:
        return self.decode_packet(x)

    def inference(self, x: Any) -> Dict[str, Any]:
        return self.decode_packet(x)

    def forward(self, x: Any) -> Dict[str, Any]:
        return self.decode_packet(x)