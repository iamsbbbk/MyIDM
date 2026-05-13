import torch
from torch import nn
from contextlib import nullcontext


# =========================================================
# AMP helper used during reversible recomputation
# =========================================================
try:
    from torch.amp import autocast as _torch_amp_autocast

    def _autocast_like(x, enabled=True):
        if (
            enabled
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            return _torch_amp_autocast(
                device_type="cuda",
                dtype=x.dtype,
                enabled=True,
                cache_enabled=False,
            )
        return nullcontext()

except Exception:
    try:
        from torch.cuda.amp import autocast as _cuda_amp_autocast

        def _autocast_like(x, enabled=True):
            if (
                enabled
                and x.is_cuda
                and x.dtype in (torch.float16, torch.bfloat16)
            ):
                return _cuda_amp_autocast(
                    enabled=True,
                    cache_enabled=False,
                )
            return nullcontext()

    except Exception:
        def _autocast_like(x, enabled=True):
            return nullcontext()


def _ste_clamp(x, min_value, max_value):
    """
    Straight-through clamp:
    forward 使用 clamp 后的安全值；
    backward 仍然给原参数梯度，避免参数跑出范围后梯度永久为 0。
    """
    x_clamped = x.clamp(min_value, max_value)
    return x + (x_clamped - x).detach()


# =========================================================
# Reversible Module
# =========================================================
class RevModule(nn.Module):
    """
    改进版可逆模块。

    原始形式：
        y1 = (1 - v) * F(x1) + v * x2
        y2 = x1

    反向恢复：
        x1 = y2
        x2 = (y1 - (1 - v) * F(x1)) / v

    关键改进：
    1. 反向重算时使用 torch.autograd.backward，而不是只对 x1/v 做 autograd.grad。
       这样 body 参数、skip 张量、ATy、Phi、alpha 等外部图路径都能拿到梯度。
    2. v 使用 straight-through clamp，避免越界后梯度消失。
    3. 反向重算时根据输入 dtype 自动开启 autocast，避免 AMP 下 half/float dtype 冲突。
    4. retain_graph=True 默认打开，解决嵌套 RevBackProp 时图被过早释放的问题。
    """
    def __init__(
        self,
        body=None,
        v=0.5,
        min_v=1e-3,
        max_v=0.999,
        retain_graph=True,
        recompute_autocast=True,
    ):
        super().__init__()

        if body is not None:
            self.body = body

        self.v = nn.Parameter(torch.tensor([float(v)], dtype=torch.float32))

        self.min_v = float(min_v)
        self.max_v = float(max_v)
        self.retain_graph = bool(retain_graph)
        self.recompute_autocast = bool(recompute_autocast)

    def _v(self):
        return _ste_clamp(self.v, self.min_v, self.max_v)

    def _call_body(self, x):
        body = getattr(self, "body", None)

        if body is None or not callable(body):
            raise RuntimeError(
                f"{self.__class__.__name__} has no callable body. "
                f"Either pass body=module or override body(self, x)."
            )

        return body(x)

    def forward(self, x1, x2):
        v = self._v().to(device=x1.device, dtype=x1.dtype)
        fx = self._call_body(x1)
        return (1.0 - v) * fx + v * x2, x1

    def backward_pass(self, y1, y2, dy1, dy2):
        """
        y1 = (1 - v) * F(x1) + v * x2
        y2 = x1

        输入：
            y1, y2: 正向输出
            dy1, dy2: 来自后续层的梯度

        输出：
            x1, x2, dx1, dx2
        """
        # 1. 恢复 x1
        with torch.no_grad():
            x1_detached = y2.detach()

        # 2. 重算局部前向图
        with torch.enable_grad():
            x1 = x1_detached.requires_grad_(True)

            with _autocast_like(x1, enabled=self.recompute_autocast):
                v = self._v().to(device=x1.device, dtype=x1.dtype)
                fx = self._call_body(x1)
                v = v.to(device=fx.device, dtype=fx.dtype)
                f_part = (1.0 - v) * fx

            # 3. 恢复 x2
            with torch.no_grad():
                v_safe = v.detach().clamp_min(self.min_v)
                y1_cast = y1.detach().to(device=f_part.device, dtype=f_part.dtype)
                x2 = (y1_cast - f_part.detach()) / v_safe

            # 4. 重新构造 y1，用 autograd.backward 让所有外部路径拿到梯度
            #    这一步非常关键：
            #    - body 参数梯度
            #    - skip 梯度
            #    - Injector 中 A/AT/ATy/Phi 的梯度
            #    - Step 中 alpha/dc_step 的梯度
            #    都依赖这里。
            y1_recomputed = f_part + v * x2.detach()

            dy1_cast = dy1.to(
                device=y1_recomputed.device,
                dtype=y1_recomputed.dtype,
            )

            torch.autograd.backward(
                tensors=y1_recomputed,
                grad_tensors=dy1_cast,
                retain_graph=self.retain_graph,
            )

        # 5. 根据可逆结构手动组装 dx1 / dx2
        with torch.no_grad():
            if x1.grad is None:
                grad_x1 = torch.zeros_like(x1)
            else:
                grad_x1 = x1.grad

            dx1 = grad_x1 + dy2.to(device=grad_x1.device, dtype=grad_x1.dtype)

            v_out = self._v().to(device=dy1.device, dtype=dy1.dtype)
            dx2 = v_out * dy1

        return (
            x1.detach(),
            x2.detach().to(device=y1.device, dtype=y1.dtype),
            dx1.detach(),
            dx2.detach(),
        )


# =========================================================
# Vanilla forward for debugging
# =========================================================
class VanillaBackProp:
    @staticmethod
    def apply(x, layers):
        layers = list(layers)

        if x.size(1) % 2 != 0:
            raise ValueError(
                f"VanillaBackProp expects even channel number, got {x.shape}"
            )

        x1, x2 = x.chunk(2, dim=1)

        for layer in layers:
            x1, x2 = layer(x1, x2)

        return torch.cat([x1, x2], dim=1)


# =========================================================
# Custom reversible autograd
# =========================================================
class RevBackProp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layers):
        layers = list(layers)

        if x.size(1) % 2 != 0:
            raise ValueError(
                f"RevBackProp expects even channel number, got {x.shape}"
            )

        with torch.no_grad():
            x1, x2 = x.chunk(2, dim=1)

            for layer in layers:
                x1, x2 = layer(x1, x2)

        ctx.save_for_backward(x1.detach(), x2.detach())
        ctx.layers = layers

        return torch.cat([x1, x2], dim=1)

    @staticmethod
    def backward(ctx, dx):
        if dx is None:
            return None, None

        dx1, dx2 = dx.chunk(2, dim=1)
        x1, x2 = ctx.saved_tensors

        for layer in reversed(ctx.layers):
            x1, x2, dx1, dx2 = layer.backward_pass(x1, x2, dx1, dx2)

        return torch.cat([dx1, dx2], dim=1), None