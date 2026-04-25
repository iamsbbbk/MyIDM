# deploy/idm_runtime.py
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from model import build_myidm_net
from deploy.runtime_utils import make_A_AT_for_shape, infer_hw_from_blocks


class MyIDMRuntime:
    """
    只包装现有 MyIDM，不改网络本体。
    由于 model.py 使用了全局变量，因此这里必须串行推理。
    """
    def __init__(
        self,
        checkpoint: str,
        sd_path: str = "./sd15",
        step_number: int | None = None,
        block_size: int | None = None,
        device: str | None = None,
        phi_path: str | None = None,
        use_amp: bool | None = None,
    ):
        self.checkpoint = str(checkpoint)
        self.sd_path = str(sd_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = (self.device.type == "cuda") if use_amp is None else bool(use_amp)

        ckpt = torch.load(self.checkpoint, map_location="cpu")
        self.ckpt = ckpt if isinstance(ckpt, dict) else {}
        cfg = self.ckpt.get("config", {}) if isinstance(self.ckpt, dict) else {}

        self.step_number = int(step_number if step_number is not None else cfg.get("step_number", 8))
        self.block_size = int(block_size if block_size is not None else cfg.get("block_size", 8))

        self.model = build_myidm_net(
            T=self.step_number,
            sd15_path=self.sd_path,
            checkpoint=self.checkpoint,
            device=str(self.device),
            strict=False,
            train_mode=False,
        )
        self.model.eval()

        self._phi_ckpt = None
        if isinstance(self.ckpt, dict):
            msd = self.ckpt.get("matrix_state_dict", {})
            if isinstance(msd, dict) and "Phi" in msd:
                self._phi_ckpt = msd["Phi"].detach().float().cpu()

        self._phi_file = None
        if phi_path:
            phi_path = Path(phi_path)
            if not phi_path.exists():
                raise FileNotFoundError(f"phi_path 不存在: {phi_path}")
            self._phi_file = torch.from_numpy(np.load(phi_path)).float()

        if self._phi_file is None and self._phi_ckpt is None:
            raise RuntimeError(
                "没有可用的 Phi。请确保 checkpoint 中包含 matrix_state_dict['Phi']，"
                "或启动时显式提供 --phi-path"
            )

        self._lock = threading.Lock()

    def _resolve_phi(self, meta: dict) -> torch.Tensor:
        phi_source = str(meta.get("phi_source", "checkpoint")).lower()

        if phi_source == "checkpoint":
            if self._phi_ckpt is None:
                raise RuntimeError("当前 checkpoint 不包含 Phi，无法使用 phi_source=checkpoint")
            return self._phi_ckpt.to(self.device)

        if phi_source == "file":
            if self._phi_file is None:
                raise RuntimeError("未提供 phi_path，无法使用 phi_source=file")
            return self._phi_file.to(self.device)

        if phi_source == "inline":
            phi_inline = meta.get("phi", None)
            if phi_inline is None:
                raise RuntimeError("phi_source=inline 但 meta 中没有 phi")
            return torch.tensor(phi_inline, dtype=torch.float32, device=self.device)

        raise ValueError(f"未知 phi_source: {phi_source}")

    def _normalize_measurement(self, measurement: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(measurement, np.ndarray):
            y = torch.from_numpy(measurement).float()
        else:
            y = measurement.detach().float().cpu()

        # 允许:
        # [q, L]      -> [1, q, L]
        # [C, q, L]   -> [C, q, L]
        if y.ndim == 2:
            y = y.unsqueeze(0)
        elif y.ndim != 3:
            raise ValueError(f"measurement 期望 2D/3D，当前 shape={tuple(y.shape)}")

        return y.to(self.device)

    def infer(self, measurement: np.ndarray | torch.Tensor, meta: dict) -> dict:
        """
        输入:
            measurement: [q, L] 或 [Bflat, q, L]
            meta:
                {
                  "height": 32,
                  "width": 32,
                  "block_size": 8,
                  "channels": 1,
                  "phi_source": "checkpoint"
                }
        输出:
            {
              "reconstruction": np.ndarray,
              "status": "ok",
              ...
            }
        """
        t0 = time.time()

        with self._lock:
            y = self._normalize_measurement(measurement)
            block_size = int(meta.get("block_size", self.block_size))
            channels = int(meta.get("channels", y.shape[0]))

            height = meta.get("height", None)
            width = meta.get("width", None)

            if height is None or width is None:
                height, width = infer_hw_from_blocks(y.shape[-1], block_size)

            height = int(height)
            width = int(width)

            Phi = self._resolve_phi(meta)
            if Phi.ndim != 2:
                raise ValueError(f"Phi 必须是 2D 张量，当前 shape={tuple(Phi.shape)}")

            # Phi: [N, q]
            if Phi.shape[1] != y.shape[1]:
                raise ValueError(
                    f"measurement 的 q 与 Phi 不匹配: y.shape[1]={y.shape[1]}, Phi.shape[1]={Phi.shape[1]}"
                )

            A_func, AT_func = make_A_AT_for_shape(height, width, block_size, Phi)

            with torch.inference_mode():
                x_rec = self.model(y, A_func, AT_func, use_amp_=self.use_amp)

            # x_rec: [Bflat, 1, H, W]
            x_np = x_rec.detach().float().cpu().numpy()
            x_np = x_np.squeeze(1)  # [Bflat, H, W]

            # 如果是典型 RGB/多通道情况，可进一步整理成 [C,H,W]
            if channels > 0 and x_np.shape[0] == channels:
                x_np = x_np.reshape(channels, height, width)

            latency_ms = (time.time() - t0) * 1000.0

            return {
                "reconstruction": x_np.astype(np.float32),
                "status": "ok",
                "height": height,
                "width": width,
                "block_size": block_size,
                "channels": channels,
                "q": int(y.shape[1]),
                "num_blocks": int(y.shape[-1]),
                "latency_ms": float(latency_ms),
                "phi_source": meta.get("phi_source", "checkpoint"),
            }