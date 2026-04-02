import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch


@dataclass
class OperatorMeta:
    op_type: str = "gaussian_flatten"
    img_h: int = 32
    img_w: int = 32
    ratio: float = 0.25
    m: int = 0
    seed: int = 2026


class GaussianCSOperator:
    """
    扁平化高斯压缩感知算子：
        x: [B,1,H,W]
        y: [B,M]

    A(x)  = x_flat @ Phi^T
    AT(y) = y @ Phi -> reshape([B,1,H,W])

    这是一个自洽、易于 sender / receiver 统一的算子实现。
    """

    def __init__(
        self,
        img_h: int = 32,
        img_w: int = 32,
        ratio: float = 0.25,
        m: int = 0,
        seed: int = 2026,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.n = self.img_h * self.img_w
        self.ratio = float(ratio)
        self.m = int(m) if int(m) > 0 else max(1, int(self.n * self.ratio))
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        # Phi: [M, N]
        phi = torch.randn(self.m, self.n, generator=gen, dtype=torch.float32) / math.sqrt(self.m)
        self.phi = phi.to(device=self.device, dtype=self.dtype)
        self.phi_t = self.phi.transpose(0, 1).contiguous()

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W]
        returns y: [B,M]
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"A(x) 期望输入 [B,1,H,W]，当前 shape={tuple(x.shape)}")
        if x.shape[2] != self.img_h or x.shape[3] != self.img_w:
            raise ValueError(
                f"A(x) 输入图像尺寸不匹配，期望 {(self.img_h, self.img_w)}，当前 {(x.shape[2], x.shape[3])}"
            )

        b = x.shape[0]
        x_flat = x.reshape(b, -1)  # [B,N]
        y = x_flat @ self.phi_t    # [B,M]
        return y

    def AT(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B,M] 或 [M]
        returns x_bp: [B,1,H,W]
        """
        if y.ndim == 1:
            y = y.unsqueeze(0)

        if y.ndim != 2:
            raise ValueError(f"AT(y) 期望输入 [B,M] 或 [M]，当前 shape={tuple(y.shape)}")

        if y.shape[1] != self.m:
            raise ValueError(f"AT(y) 测量维数不匹配，期望 M={self.m}，当前 {y.shape[1]}")

        b = y.shape[0]
        x_flat = y @ self.phi      # [B,N]
        x = x_flat.reshape(b, 1, self.img_h, self.img_w)
        return x

    def meta(self) -> Dict[str, Any]:
        return {
            "op_type": "gaussian_flatten",
            "img_h": self.img_h,
            "img_w": self.img_w,
            "ratio": self.ratio,
            "m": self.m,
            "seed": self.seed,
        }


class OperatorFactory:
    """
    小缓存工厂，避免同一配置反复生成 Phi。
    """

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self._cache = {}

    def get(
        self,
        img_h: int = 32,
        img_w: int = 32,
        ratio: float = 0.25,
        m: int = 0,
        seed: int = 2026,
        op_type: str = "gaussian_flatten",
    ):
        key = (op_type, int(img_h), int(img_w), float(ratio), int(m), int(seed), str(self.device), str(self.dtype))
        if key not in self._cache:
            if op_type != "gaussian_flatten":
                raise ValueError(f"暂不支持的 op_type: {op_type}")
            self._cache[key] = GaussianCSOperator(
                img_h=img_h,
                img_w=img_w,
                ratio=ratio,
                m=m,
                seed=seed,
                device=self.device,
                dtype=self.dtype,
            )
        return self._cache[key]

    def get_from_meta(self, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        return self.get(
            img_h=int(meta.get("img_h", 32)),
            img_w=int(meta.get("img_w", 32)),
            ratio=float(meta.get("ratio", 0.25)),
            m=int(meta.get("m", 0)),
            seed=int(meta.get("seed", 2026)),
            op_type=meta.get("op_type", "gaussian_flatten"),
        )