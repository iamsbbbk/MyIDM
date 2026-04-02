import types
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from contextlib import nullcontext

try:
    # 新版 PyTorch
    from torch.amp import autocast as torch_autocast

    def safe_autocast(device_type="cuda", enabled=True, cache_enabled=False):
        return torch_autocast(device_type=device_type, enabled=enabled, cache_enabled=cache_enabled)

except Exception:
    try:
        # PyTorch 2.0.1 / 旧版 CUDA AMP
        from torch.cuda.amp import autocast as cuda_autocast

        def safe_autocast(device_type="cuda", enabled=True, cache_enabled=False):
            if device_type == "cuda":
                return cuda_autocast(enabled=enabled, cache_enabled=cache_enabled)
            return nullcontext()

    except Exception:
        def safe_autocast(device_type="cuda", enabled=True, cache_enabled=False):
            return nullcontext()

from utils import *  # 保持与原工程兼容
from backprop import RevModule, VanillaBackProp, RevBackProp
from forward import (
    MyUNet2DConditionModel_SD_v1_5_forward,
    MyCrossAttnDownBlock2D_SD_v1_5_forward,
    MyCrossAttnUpBlock2D_SD_v1_5_forward,
    MyResnetBlock2D_SD_v1_5_forward,
    MyTransformer2DModel_SD_v1_5_forward
)


# =========================================================
# 全局上下文（与原始工程风格保持兼容）
# =========================================================
y = None
A = None
AT = None
unet = None
ATy = None
alpha_bar = None
alpha_global = None
use_amp = True
t = 1


# =========================================================
# Helper
# =========================================================
def load_sd15_unet(sd15_path="./sd15"):
    """
    加载 SD1.5 UNet。
    注意：这里不再错误地改成 8 通道。
    当前 MyIDM 正确的 UNet 输入输出都是 4 通道。
    """
    try:
        from diffusers import UNet2DConditionModel
    except Exception as e:
        raise ImportError(f"未能导入 diffusers.UNet2DConditionModel: {e}")

    sd15_path = Path(sd15_path)
    if not sd15_path.exists():
        raise FileNotFoundError(f"sd15 路径不存在: {sd15_path}")

    # 优先尝试 root/subfolder=unet
    try:
        unet = UNet2DConditionModel.from_pretrained(
            str(sd15_path),
            subfolder="unet",
            local_files_only=True,
        )
        return unet
    except Exception:
        pass

    # 再尝试直接从路径加载
    try:
        unet = UNet2DConditionModel.from_pretrained(
            str(sd15_path),
            local_files_only=True,
        )
        return unet
    except Exception as e:
        raise RuntimeError(f"加载 SD1.5 UNet 失败: {e}")


def build_myidm_net(
    T=8,
    sd15_path="./sd15",
    checkpoint="",
    device="cuda",
    strict=False,
    train_mode=True,
):
    unet = load_sd15_unet(sd15_path)
    net = Net(T=T, unet=unet)

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt
            if isinstance(ckpt, dict):
                for k in ["model_state_dict", "state_dict", "model", "net", "ema_state_dict"]:
                    if k in ckpt and isinstance(ckpt[k], dict):
                        state = ckpt[k]
                        break
            net.load_state_dict(state, strict=strict)

    if device is not None:
        net = net.to(device)

    if train_mode:
        net.train()
    else:
        net.eval()

    return net


# =========================================================
# Injector
# =========================================================
class Injector(nn.Module):
    def __init__(self, nf, r, T):
        super().__init__()
        self.f2i = nn.ModuleList([
            nn.Sequential(
                nn.PixelShuffle(r),
                nn.Conv2d(nf // (r * r), 1, 1),
            ) for _ in range(T)
        ])
        self.i2f = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, nf // (r * r), 1),
                nn.PixelUnshuffle(r),
            ) for _ in range(T)
        ])

    def forward(self, x_in):
        global t, A, AT, ATy

        x = self.f2i.__getitem__(t - 1)(x_in)
        ax = A(x)
        atax = AT(ax)
        x = torch.cat([x, atax, ATy], dim=1)

        return x_in + self.i2f.__getitem__(t - 1)(x)

# =========================================================
# Step
# =========================================================
class Step(RevModule):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def body(self, x):
        """
        x: [B,1,H,W]   （注意这里是可逆状态的一半，不是 2 通道）
        因此：
            pixel_unshuffle(x,2) -> [B,4,H/2,W/2]
        这就是为什么 UNet 必须保持 4 通道输入输出。
        """
        global t, y, A, AT, unet, ATy, alpha_bar, alpha_global, use_amp

        amp_device = "cuda" if x.is_cuda else "cpu"
        amp_enabled = bool(use_amp and x.is_cuda)

        with safe_autocast(device_type=amp_device, enabled=amp_enabled, cache_enabled=False):
            t = self.t

            cur_alpha_bar = alpha_bar[t].clamp(min=1e-6, max=1.0)
            prev_alpha_bar = alpha_bar[t - 1].clamp(min=1e-6, max=1.0)

            try:
                x_unshuffle = F.pixel_unshuffle(x, 2)   # [B,1,H,W] -> [B,4,H/2,W/2]
                e = F.pixel_shuffle(unet(x_unshuffle), 2)
            except Exception as e_unet:
                raise RuntimeError(
                    f"[MyIDM Step Error] t={t}, x={tuple(x.shape)}, "
                    f"pixel_unshuffle={tuple(x_unshuffle.shape) if 'x_unshuffle' in locals() else 'NA'}"
                ) from e_unet

            # x0 estimation
            x0 = (x - (1.0 - cur_alpha_bar).sqrt() * e) / cur_alpha_bar.sqrt()

            # 数据一致性建议使用 float32 更稳
            x0_fp32 = x0.float()
            y_fp32 = y.float() if torch.is_tensor(y) else y
            dc_grad = AT(A(x0_fp32) - y_fp32).to(x0.dtype)

            # learned step
            lambda_t = alpha_global[t - 1].clamp(min=1e-6, max=1.0)
            x0 = x0 - lambda_t * dc_grad

            # reverse update
            x_prev = prev_alpha_bar.sqrt() * x0 + (1.0 - prev_alpha_bar).sqrt() * e

            return x_prev


# =========================================================
# Net
# =========================================================
class Net(nn.Module):
    def __init__(self, T, unet):
        super().__init__()
        self.T = int(T)

        # -------------------------
        # 裁剪 UNet 结构（保持你原有逻辑）
        # -------------------------
        if hasattr(unet, "time_embedding"):
            del unet.time_embedding
        if hasattr(unet, "mid_block"):
            del unet.mid_block

        unet.down_blocks = unet.down_blocks[:-2]
        unet.down_blocks[-1].downsamplers = None
        unet.up_blocks = unet.up_blocks[2:]

        self.body = nn.ModuleList([Step(T - i) for i in range(T)])

        # -------------------------
        # 稳定参数
        # -------------------------
        self.input_help_scale_factor = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.merge_scale_factor = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        # learned alpha，训练时用 clamp 保证合法
        self.alpha = nn.Parameter(torch.full((T,), 0.5, dtype=torch.float32))

        self.unet = unet
        self.unet_add_down_rev_modules_and_injectors(T)
        self.unet_add_up_rev_modules_and_injectors(T)
        self.unet_remove_resnet_time_emb_proj()
        self.unet_remove_cross_attn()
        self.unet_set_inplace_to_true()
        self.unet_replace_forward_methods()

    def _effective_alpha(self):
        return self.alpha.clamp(0.05, 0.95)

    # -------------------------
    # 结构注入
    # -------------------------
    def unet_add_down_rev_modules_and_injectors(self, T):
        self.unet.down_blocks[0].register_module(
            "injectors", nn.ModuleList([Injector(320, 2, T) for _ in range(4)])
        )
        self.unet.down_blocks[1].register_module(
            "injectors", nn.ModuleList([Injector(640, 4, T) for _ in range(4)])
        )

        for i in range(2):
            self.unet.down_blocks[i].register_module("rev_module_lists", nn.ModuleList([]))
            self.unet.down_blocks[i].register_parameter(
                "input_help_scale_factor", nn.Parameter(torch.ones(1,))
            )
            self.unet.down_blocks[i].register_parameter(
                "merge_scale_factors", nn.Parameter(torch.zeros(2,))
            )

            for j in range(2):
                rev_module_list = nn.ModuleList([])

                if (
                    self.unet.down_blocks[i].resnets[j].in_channels
                    == self.unet.down_blocks[i].resnets[j].out_channels
                ):
                    rev_module_list.append(RevModule(self.unet.down_blocks[i].resnets[j]))

                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2 * j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2 * j + 1]))
                self.unet.down_blocks[i].rev_module_lists.append(rev_module_list)

    def unet_add_up_rev_modules_and_injectors(self, T):
        self.unet.up_blocks[0].register_module(
            "injectors", nn.ModuleList([Injector(640, 4, T) for _ in range(6)])
        )
        self.unet.up_blocks[1].register_module(
            "injectors", nn.ModuleList([Injector(320, 2, T) for _ in range(6)])
        )

        for i in range(2):
            self.unet.up_blocks[i].register_parameter(
                "input_help_scale_factor", nn.Parameter(torch.ones(1,))
            )
            self.unet.up_blocks[i].register_parameter(
                "merge_scale_factor", nn.Parameter(torch.zeros(1,))
            )

            rev_module_list = nn.ModuleList([])
            for j in range(3):
                if j > 0:
                    rev_module_list.append(RevModule(self.unet.up_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2 * j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2 * j + 1]))

            self.unet.up_blocks[i].register_module("rev_module_list", rev_module_list)

    def unet_replace_forward_methods(self):
        from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.transformer_2d import Transformer2DModel

        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(
                    MyCrossAttnDownBlock2D_SD_v1_5_forward, module
                )
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(
                    MyCrossAttnUpBlock2D_SD_v1_5_forward, module
                )
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(
                    MyResnetBlock2D_SD_v1_5_forward, module
                )
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(
                    MyTransformer2DModel_SD_v1_5_forward, module
                )

        self.unet.apply(replace_forward_methods)
        self.unet.forward = types.MethodType(
            MyUNet2DConditionModel_SD_v1_5_forward, self.unet
        )

    def unet_remove_resnet_time_emb_proj(self):
        from diffusers.models.resnet import ResnetBlock2D

        def remove_time_emb(module):
            if isinstance(module, ResnetBlock2D):
                module.time_emb_proj = None

        self.unet.apply(remove_time_emb)

    def unet_remove_cross_attn(self):
        from diffusers.models.attention import BasicTransformerBlock

        def remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                module.attn2 = None
                module.norm2 = None

        self.unet.apply(remove_cross_attn)

    def unet_set_inplace_to_true(self):
        def set_inplace(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True

        self.unet.apply(set_inplace)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, y_, A_, AT_, use_amp_=True):
        global y, A, AT, unet, ATy, alpha_bar, alpha_global, use_amp

        y = y_
        A = A_
        AT = AT_
        unet = self.unet
        use_amp = use_amp_

        alpha_global = self._effective_alpha()
        alpha_bar = torch.cat(
            [torch.ones(1, device=y.device, dtype=alpha_global.dtype), alpha_global.cumprod(dim=0)],
            dim=0
        )

        x = AT(y)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        ATy = x

        help_scale = self.input_help_scale_factor.clamp(0.0, 2.0)
        x = alpha_bar[-1].sqrt() * torch.cat([x, help_scale * x], dim=1)

        x = RevBackProp.apply(x, self.body)

        merge_scale = self.merge_scale_factor.clamp(-1.0, 1.0)
        return x[:, :1] + merge_scale * x[:, 1:]

    def forward_rgb(self, y_rgb, A_, AT_, use_amp_=True):
        """
        可选辅助接口：对 RGB 三通道测量逐通道重构
        y_rgb: [B,3,M] 或 list(len=3) of [B,M]
        """
        if isinstance(y_rgb, (list, tuple)):
            outs = [self.forward(y_c, A_, AT_, use_amp_=use_amp_) for y_c in y_rgb]
            return torch.cat(outs, dim=1)

        if y_rgb.ndim != 3 or y_rgb.shape[1] != 3:
            raise ValueError(f"forward_rgb 期望输入 [B,3,M]，当前 shape={tuple(y_rgb.shape)}")

        outs = []
        for c in range(3):
            outs.append(self.forward(y_rgb[:, c], A_, AT_, use_amp_=use_amp_))
        return torch.cat(outs, dim=1)