import torch
from torch import nn


class RevModule(nn.Module):
    """
    稳定版可逆模块：
    - 对 v 做 clamp，避免除零与极端不稳定
    - 使用 torch.autograd.grad 替代 backward()，避免嵌套 backward 图被提前释放
    - 修复 self.v.grad 可能为 None 的问题
    """
    def __init__(self, body=None, v=0.5, min_v=1e-3, max_v=0.999):
        super().__init__()
        if body is not None:
            self.body = body
        self.v = nn.Parameter(torch.tensor([float(v)], dtype=torch.float32))
        self.min_v = float(min_v)
        self.max_v = float(max_v)

    def _v(self):
        return self.v.clamp(self.min_v, self.max_v)

    def forward(self, x1, x2):
        v = self._v()
        return (1.0 - v) * self.body(x1) + v * x2, x1

    def backward_pass(self, y1, y2, dy1, dy2):
        """
        y1 = (1 - v) * body(x1) + v * x2
        y2 = x1
        """
        v = self._v()

        # 1) 从输出恢复 x1
        with torch.no_grad():
            x1 = y2.detach()

        # 2) 重新前向，构建局部图，并计算对 x1 / v 的梯度
        with torch.enable_grad():
            x1 = x1.requires_grad_(True)
            F_part = (1.0 - v) * self.body(x1)

            grad_x1, grad_v_from_f = torch.autograd.grad(
                outputs=F_part,
                inputs=(x1, self.v),
                grad_outputs=dy1,
                retain_graph=True,   # 关键：嵌套 RevBackProp 时不能太早释放
                create_graph=False,
                allow_unused=True,
            )

        with torch.no_grad():
            if grad_x1 is None:
                grad_x1 = torch.zeros_like(x1)

            if grad_v_from_f is None:
                grad_v_from_f = torch.zeros_like(self.v)

            # dx1
            dx1 = grad_x1 + dy2

            # dx2
            dx2 = v * dy1

            # 恢复 x2
            F_det = F_part.detach()
            x2 = (y1 - F_det) / v

            # 对 v 的额外梯度项：来自 y1 = F + v*x2
            extra_v_grad = (x2 * dy1).sum().reshape_as(self.v)
            total_v_grad = grad_v_from_f + extra_v_grad

            if self.v.grad is None:
                self.v.grad = total_v_grad.clone()
            else:
                self.v.grad = self.v.grad + total_v_grad

        return x1.detach(), x2.detach(), dx1.detach(), dx2.detach()


class VanillaBackProp:
    @staticmethod
    def apply(x, layers):
        x1, x2 = x.chunk(2, dim=1)
        for layer in layers:
            x1, x2 = layer(x1, x2)
        return torch.cat([x1, x2], dim=1)


class RevBackProp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layers):
        # 显式转 list，避免 generator 被消费
        layers = list(layers)

        with torch.no_grad():
            x1, x2 = x.chunk(2, dim=1)
            for layer in layers:
                x1, x2 = layer(x1, x2)

        ctx.save_for_backward(x1.detach(), x2.detach())
        ctx.layers = layers
        return torch.cat([x1, x2], dim=1)

    @staticmethod
    def backward(ctx, dx):
        dx1, dx2 = dx.chunk(2, dim=1)
        x1, x2 = ctx.saved_tensors

        for layer in reversed(ctx.layers):
            x1, x2, dx1, dx2 = layer.backward_pass(x1, x2, dx1, dx2)

        return torch.cat([dx1, dx2], dim=1), None