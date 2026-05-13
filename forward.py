import torch
import torch.utils.checkpoint as checkpoint_utils

from backprop import RevBackProp


# =========================================================
# Checkpoint helper
# =========================================================
def MyCheckpoint(module, x):
    """
    非重入 checkpoint。
    在 no_grad 或输入不需要梯度时直接前向，避免无意义 checkpoint。
    """
    if (not torch.is_grad_enabled()) or (not x.requires_grad):
        return module(x)

    def f(z):
        return module(z)

    return checkpoint_utils.checkpoint(
        f,
        x,
        use_reentrant=False,
    )


def _ste_clamp(x, min_value, max_value):
    x_clamped = x.clamp(min_value, max_value)
    return x + (x_clamped - x).detach()


def _clamp_help_scale(x):
    return _ste_clamp(x, 0.0, 2.0)


def _clamp_merge_scale(x):
    return _ste_clamp(x, -1.0, 1.0)


def _set_transient_skip(module, tensor):
    """
    给 ResNet 临时挂 skip。
    使用 persistent=False，避免保存到 state_dict。
    不在 forward 末尾清空，因为可逆反传阶段还需要它。
    下一次 forward 会覆盖。
    """
    if "skip" in module._buffers:
        module._buffers["skip"] = tensor
    else:
        try:
            module.register_buffer("skip", tensor, persistent=False)
        except Exception:
            setattr(module, "skip", tensor)


def _take_tensor(x):
    """
    兼容某些 diffusers 版本 Attention 返回 tuple 的情况。
    """
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


# =========================================================
# UNet forward
# =========================================================
def MyUNet2DConditionModel_SD_v1_5_forward(self, x):
    """
    裁剪后的 SD1.5 UNet forward。

    输入：
        x: [B, 4, H, W]

    输出：
        x: [B, 4, H, W]

    说明：
        原图像域是 [B,1,32,32]。
        Step 中 pixel_unshuffle(x,2) 后变成 [B,4,16,16]，
        因此这里 UNet 仍保持 4 通道输入/输出。
    """
    if x.ndim != 4 or x.shape[1] != 4:
        raise ValueError(
            f"MyIDM UNet expects [B,4,H,W], got shape={tuple(x.shape)}"
        )

    x = self.conv_in(x)
    skip_stack = [x]

    for down in self.down_blocks:
        x, cur_skip = down(x)
        skip_stack.extend(list(cur_skip))

    # 裁剪掉后两层 down block 后，进入 up block 前需要补齐通道。
    x = torch.cat([x, x], dim=1).contiguous()

    for up in self.up_blocks:
        n_skip = len(up.resnets)

        if len(skip_stack) < n_skip:
            raise RuntimeError(
                f"Not enough skip tensors for up block. "
                f"Need {n_skip}, got {len(skip_stack)}."
            )

        cur_skip = skip_stack[-n_skip:]
        x = up(x, cur_skip)
        del skip_stack[-n_skip:]

    x = self.conv_norm_out(x)
    x = self.conv_act(x)
    x = self.conv_out(x)

    return x


# =========================================================
# Down block
# =========================================================
def MyCrossAttnDownBlock2D_SD_v1_5_forward(self, x):
    skip = []

    # 通道改变的第一个 ResNet 不能直接放进 RevModule，先正常执行。
    if self.resnets[0].in_channels != self.resnets[0].out_channels:
        x = MyCheckpoint(self.resnets[0], x)

    help_scale = _clamp_help_scale(self.input_help_scale_factor).to(
        device=x.device,
        dtype=x.dtype,
    )

    x = torch.cat([x, help_scale * x], dim=1).contiguous()

    for i in range(2):
        x = RevBackProp.apply(x, self.rev_module_lists[i])

        x1, x2 = x.chunk(2, dim=1)

        merge_scale = _clamp_merge_scale(self.merge_scale_factors[i]).to(
            device=x.device,
            dtype=x.dtype,
        )

        x_merge = x1 + merge_scale * x2
        skip.append(x_merge)

    x = x_merge

    if self.downsamplers is not None and len(self.downsamplers) > 0:
        x = MyCheckpoint(self.downsamplers[0], x)
        skip.append(x)

    return x, skip


# =========================================================
# Up block
# =========================================================
def MyCrossAttnUpBlock2D_SD_v1_5_forward(self, x, skip):
    if len(skip) < 3:
        raise RuntimeError(
            f"CrossAttnUpBlock expects at least 3 skip tensors, got {len(skip)}."
        )

    x = MyCheckpoint(
        self.resnets[0],
        torch.cat([x, skip[-1]], dim=1).contiguous()
    )

    # resnets[1] 和 resnets[2] 的 skip 通过 transient buffer 传入。
    _set_transient_skip(self.resnets[1], skip[-2])
    _set_transient_skip(self.resnets[2], skip[-3])

    help_scale = _clamp_help_scale(self.input_help_scale_factor).to(
        device=x.device,
        dtype=x.dtype,
    )

    x = torch.cat([x, help_scale * x], dim=1).contiguous()

    x = RevBackProp.apply(x, self.rev_module_list)

    x1, x2 = x.chunk(2, dim=1)

    merge_scale = _clamp_merge_scale(self.merge_scale_factor).to(
        device=x.device,
        dtype=x.dtype,
    )

    x = x1 + merge_scale * x2

    if self.upsamplers is not None and len(self.upsamplers) > 0:
        x = MyCheckpoint(self.upsamplers[0], x)

    return x


# =========================================================
# ResNet block without time embedding
# =========================================================
def MyResnetBlock2D_SD_v1_5_forward(self, x_in):
    skip = getattr(self, "skip", None)

    if skip is not None:
        if skip.shape[-2:] != x_in.shape[-2:]:
            raise RuntimeError(
                f"Skip spatial size mismatch: x={tuple(x_in.shape)}, "
                f"skip={tuple(skip.shape)}"
            )

        skip = skip.to(device=x_in.device, dtype=x_in.dtype)
        x_in = torch.cat([x_in, skip], dim=1).contiguous()

    x = self.norm1(x_in)
    x = self.nonlinearity(x)
    x = self.conv1(x)

    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.conv2(x)

    conv_shortcut = getattr(self, "conv_shortcut", None)

    if conv_shortcut is not None:
        shortcut = conv_shortcut(x_in)
    else:
        shortcut = x_in

    return x + shortcut


# =========================================================
# Transformer block without cross attention
# =========================================================
def MyTransformer2DModel_SD_v1_5_forward(self, x_in):
    b, _, h, w = x_in.shape

    residual = x_in

    x = self.norm(x_in)
    x = self.proj_in(x)

    inner_c = x.shape[1]

    x = x.permute(0, 2, 3, 1).reshape(b, h * w, inner_c).contiguous()

    for block in self.transformer_blocks:
        attn_out = block.attn1(block.norm1(x))
        attn_out = _take_tensor(attn_out)
        x = x + attn_out

        ff_out = block.ff(block.norm3(x))
        ff_out = _take_tensor(ff_out)
        x = x + ff_out

    x = x.reshape(b, h, w, inner_c).permute(0, 3, 1, 2).contiguous()
    x = self.proj_out(x)

    return x + residual