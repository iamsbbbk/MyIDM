import types
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch.amp import autocast as torch_autocast

    def safe_autocast(device_type="cuda", enabled=True, cache_enabled=False):
        return torch_autocast(
            device_type=device_type,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )

except Exception:
    try:
        from torch.cuda.amp import autocast as cuda_autocast

        def safe_autocast(device_type="cuda", enabled=True, cache_enabled=False):
            if device_type == "cuda":
                return cuda_autocast(
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )
            return nullcontext()

    except Exception:
        def safe_autocast(device_type="cuda", enabled=True, cache_enabled=False):
            return nullcontext()


from utils import *  # 保持原工程兼容
from backprop import RevModule, RevBackProp
from forward import (
    MyUNet2DConditionModel_SD_v1_5_forward,
    MyCrossAttnDownBlock2D_SD_v1_5_forward,
    MyCrossAttnUpBlock2D_SD_v1_5_forward,
    MyResnetBlock2D_SD_v1_5_forward,
    MyTransformer2DModel_SD_v1_5_forward,
)


# =========================================================
# Global reversible context
# =========================================================
y = None
A = None
AT = None
unet = None
ATy = None

alpha_bar = None
alpha_global = None
dc_global = None

use_amp = True
t = 1


# =========================================================
# Helper
# =========================================================
def _ste_clamp(x, min_value, max_value):
    x_clamped = x.clamp(min_value, max_value)
    return x + (x_clamped - x).detach()


def _strip_module_prefix(state_dict):
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    return new_state


def load_sd15_unet(sd15_path="./sd15"):
    """
    加载本地 SD1.5 UNet。

    注意：
        这里不改 conv_in / conv_out。
        MyIDM 使用 pixel_unshuffle 后的 4 通道 latent-like 输入，
        因此 SD1.5 原生 4 通道结构正好匹配。
    """
    try:
        from diffusers import UNet2DConditionModel
    except Exception as e:
        raise ImportError(f"Failed to import diffusers.UNet2DConditionModel: {e}")

    sd15_path = Path(sd15_path)

    if not sd15_path.exists():
        raise FileNotFoundError(f"sd15 path does not exist: {sd15_path}")

    try:
        return UNet2DConditionModel.from_pretrained(
            str(sd15_path),
            subfolder="unet",
            local_files_only=True,
        )
    except Exception:
        pass

    try:
        return UNet2DConditionModel.from_pretrained(
            str(sd15_path),
            local_files_only=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load SD1.5 UNet from {sd15_path}: {e}")


def build_myidm_net(
    T=8,
    sd15_path="./sd15",
    checkpoint="",
    device="cuda",
    strict=False,
    train_mode=True,
):
    """
    工程级构建接口。
    """
    unet_model = load_sd15_unet(sd15_path)
    net = Net(T=T, unet=unet_model)

    if checkpoint:
        ckpt_path = Path(checkpoint)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=device)

        state = ckpt

        if isinstance(ckpt, dict):
            for key in [
                "model_state_dict",
                "state_dict",
                "model",
                "net",
                "ema_state_dict",
            ]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break

        state = _strip_module_prefix(state)
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
    """
    Measurement-aware feature injector.

    作用：
        从 UNet feature -> 图像域估计 x；
        通过 A / AT 构造数据一致性提示；
        再注入回 feature。

    改进：
        1. A/AT 前强制使用 fp32，避免 AMP 下测量矩阵 matmul 精度不稳定；
        2. 新增轻量 gate，初始时注入较弱，训练过程中自动增强；
        3. 保持原来的 PixelShuffle / PixelUnshuffle 结构。
    """
    def __init__(self, nf, r, T):
        super().__init__()

        self.nf = int(nf)
        self.r = int(r)
        self.T = int(T)

        if nf % (r * r) != 0:
            raise ValueError(f"nf={nf} must be divisible by r*r={r*r}")

        self.f2i = nn.ModuleList([
            nn.Sequential(
                nn.PixelShuffle(r),
                nn.Conv2d(nf // (r * r), 1, kernel_size=1),
            )
            for _ in range(T)
        ])

        self.i2f = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, nf // (r * r), kernel_size=1),
                nn.PixelUnshuffle(r),
            )
            for _ in range(T)
        ])

        # sigmoid(-1.5) ≈ 0.18。
        # 初始注入不过强，避免训练前期震荡。
        self.gates = nn.Parameter(torch.full((T,), -1.5, dtype=torch.float32))

    def forward(self, x_in):
        global t, A, AT, ATy

        idx = int(t) - 1

        if idx < 0 or idx >= self.T:
            raise RuntimeError(
                f"Injector got invalid diffusion step t={t}, T={self.T}"
            )

        x_img = self.f2iidx [<sup>1</sup>](x_in)

        # CS operators are more stable in fp32.
        x_img_fp32 = x_img.float()

        ax = A(x_img_fp32)
        atax = AT(ax).to(device=x_img.device, dtype=x_img.dtype)

        aty = ATy.to(device=x_img.device, dtype=x_img.dtype)

        if atax.shape != x_img.shape:
            raise RuntimeError(
                f"Injector AT(A(x)) shape mismatch: "
                f"x={tuple(x_img.shape)}, atax={tuple(atax.shape)}"
            )

        if aty.shape != x_img.shape:
            raise RuntimeError(
                f"Injector ATy shape mismatch: "
                f"x={tuple(x_img.shape)}, ATy={tuple(aty.shape)}"
            )

        hint = torch.cat([x_img, atax, aty], dim=1).contiguous()

        residual = self.i2fidx [<sup>2</sup>](hint)

        gate = torch.sigmoid(self.gates[idx]).to(
            device=x_in.device,
            dtype=x_in.dtype,
        )

        return x_in + gate * residual


# =========================================================
# One IDM reverse step
# =========================================================
class Step(RevModule):
    def __init__(self, t):
        super().__init__(
            body=None,
            v=0.5,
            min_v=1e-3,
            max_v=0.999,
            retain_graph=True,
            recompute_autocast=True,
        )
        self.t = int(t)

    def body(self, x):
        """
        x: [B,1,H,W]

        pixel_unshuffle(x,2):
            [B,1,H,W] -> [B,4,H/2,W/2]

        因此 SD1.5 UNet 仍然是 4 通道输入/输出。
        """
        global t, y, A, AT, unet, alpha_bar, alpha_global, dc_global, use_amp

        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"MyIDM Step expects [B,1,H,W], got shape={tuple(x.shape)}"
            )

        if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
            raise ValueError(
                f"MyIDM Step requires even H/W for pixel_unshuffle, got {tuple(x.shape)}"
            )

        device_type = "cuda" if x.is_cuda else "cpu"
        amp_enabled = bool(use_amp and x.is_cuda)

        t = self.t

        cur_alpha_bar = alpha_bar[t].clamp(min=1e-6, max=1.0)
        prev_alpha_bar = alpha_bar[t - 1].clamp(min=1e-6, max=1.0)

        cur_alpha_bar = cur_alpha_bar.to(device=x.device, dtype=x.dtype)
        prev_alpha_bar = prev_alpha_bar.to(device=x.device, dtype=x.dtype)

        sqrt_cur = cur_alpha_bar.sqrt()
        sqrt_prev = prev_alpha_bar.sqrt()

        sqrt_one_minus_cur = (1.0 - cur_alpha_bar).clamp(min=1e-6).sqrt()
        sqrt_one_minus_prev = (1.0 - prev_alpha_bar).clamp(min=1e-6).sqrt()

        # 1. UNet predicts residual/noise-like term.
        with safe_autocast(
            device_type=device_type,
            enabled=amp_enabled,
            cache_enabled=False,
        ):
            x_unshuffle = F.pixel_unshuffle(x, 2)

            if x_unshuffle.shape[1] != 4:
                raise RuntimeError(
                    f"pixel_unshuffle should produce 4 channels, "
                    f"got {tuple(x_unshuffle.shape)}"
                )

            e_unshuffle = unet(x_unshuffle)
            e = F.pixel_shuffle(e_unshuffle, 2)

        if e.shape != x.shape:
            raise RuntimeError(
                f"UNet output after pixel_shuffle mismatch: "
                f"x={tuple(x.shape)}, e={tuple(e.shape)}"
            )

        # 2. Estimate x0.
        x0 = (x - sqrt_one_minus_cur * e) / sqrt_cur

        # 3. Data consistency gradient.
        #    A/AT 相关操作强制 fp32，更适合 CS 矩阵乘法。
        with safe_autocast(
            device_type=device_type,
            enabled=False,
            cache_enabled=False,
        ):
            x0_fp32 = x0.float()
            y_fp32 = y.float() if torch.is_tensor(y) else y

            dc_grad = AT(A(x0_fp32) - y_fp32)
            dc_grad = torch.nan_to_num(
                dc_grad,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )

        dc_grad = dc_grad.to(device=x0.device, dtype=x0.dtype)

        # 4. Learned data consistency step.
        lambda_t = dc_global[t - 1].to(device=x0.device, dtype=x0.dtype)

        x0 = x0 - lambda_t * dc_grad

        # 5. Reverse update.
        x_prev = sqrt_prev * x0 + sqrt_one_minus_prev * e

        return x_prev


# =========================================================
# Net
# =========================================================
class Net(nn.Module):
    def __init__(self, T, unet):
        super().__init__()

        self.T = int(T)

        if self.T <= 0:
            raise ValueError(f"T should be positive, got {T}")

        if getattr(unet, "_myidm_patched", False):
            raise RuntimeError(
                "This UNet has already been patched by MyIDM. "
                "Please create a fresh UNet instance."
            )

        self.unet = unet

        self._prepare_unet_structure()

        self.body = nn.ModuleList([
            Step(self.T - i)
            for i in range(self.T)
        ])

        # -------------------------------------------------
        # Learnable schedule
        # -------------------------------------------------
        # alpha 初始化为 0.85，比 0.5 更稳定：
        # T=8 时 alpha_bar ≈ 0.27，不会让初始反投影过弱。
        self.alpha = nn.Parameter(
            torch.full((self.T,), 0.85, dtype=torch.float32)
        )

        # 数据一致性步长和 alpha 解耦。
        # sigmoid(-2.0) ≈ 0.119，初始比较保守。
        self.dc_step = nn.Parameter(
            torch.full((self.T,), -2.0, dtype=torch.float32)
        )

        self.input_help_scale_factor = nn.Parameter(
            torch.tensor([1.0], dtype=torch.float32)
        )

        self.merge_scale_factor = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32)
        )

        self.unet_add_down_rev_modules_and_injectors(self.T)
        self.unet_add_up_rev_modules_and_injectors(self.T)
        self.unet_remove_resnet_time_emb_proj()
        self.unet_remove_cross_attn()
        self.unet_disable_inplace_ops()
        self.unet_replace_forward_methods()

        self.unet._myidm_patched = True

    # -----------------------------------------------------
    # Effective parameters
    # -----------------------------------------------------
    def _effective_alpha(self):
        return _ste_clamp(self.alpha, 0.05, 0.99)

    def _effective_dc_step(self):
        return torch.sigmoid(self.dc_step).clamp(1e-5, 1.0)

    # -----------------------------------------------------
    # UNet preparation
    # -----------------------------------------------------
    def _prepare_unet_structure(self):
        """
        裁剪 SD1.5 UNet。

        原 SD1.5 UNet 大致是：
            down_blocks: 4 个
            up_blocks  : 4 个

        MyIDM 使用浅层结构：
            保留前 2 个 down blocks
            保留后 2 个 up blocks
        """
        if not hasattr(self.unet, "conv_in") or not hasattr(self.unet, "conv_out"):
            raise RuntimeError("Invalid UNet: missing conv_in or conv_out.")

        if self.unet.conv_in.in_channels != 4:
            raise ValueError(
                f"SD1.5 UNet conv_in should have 4 input channels, "
                f"got {self.unet.conv_in.in_channels}"
            )

        if self.unet.conv_out.out_channels != 4:
            raise ValueError(
                f"SD1.5 UNet conv_out should have 4 output channels, "
                f"got {self.unet.conv_out.out_channels}"
            )

        if hasattr(self.unet, "time_embedding"):
            del self.unet.time_embedding

        if hasattr(self.unet, "mid_block"):
            del self.unet.mid_block

        down_blocks = list(self.unet.down_blocks)
        up_blocks = list(self.unet.up_blocks)

        if len(down_blocks) > 2:
            self.unet.down_blocks = nn.ModuleList(down_blocks[:-2])

        if len(up_blocks) > 2:
            self.unet.up_blocks = nn.ModuleList(up_blocks[2:])

        if len(self.unet.down_blocks) != 2 or len(self.unet.up_blocks) != 2:
            raise RuntimeError(
                f"MyIDM expects 2 down blocks and 2 up blocks after pruning, "
                f"got down={len(self.unet.down_blocks)}, up={len(self.unet.up_blocks)}"
            )

        # 第二个 down block 不再继续下采样。
        self.unet.down_blocks[-1].downsamplers = None

    # -----------------------------------------------------
    # Inject reversible modules
    # -----------------------------------------------------
    def unet_add_down_rev_modules_and_injectors(self, T):
        self.unet.down_blocks[0].injectors = nn.ModuleList([
            Injector(320, 2, T)
            for _ in range(4)
        ])

        self.unet.down_blocks[1].injectors = nn.ModuleList([
            Injector(640, 4, T)
            for _ in range(4)
        ])

        for i in range(2):
            block = self.unet.down_blocks[i]

            block.rev_module_lists = nn.ModuleList([])
            block.input_help_scale_factor = nn.Parameter(torch.ones(1))
            block.merge_scale_factors = nn.Parameter(torch.zeros(2))

            for j in range(2):
                rev_module_list = nn.ModuleList([])

                if block.resnets[j].in_channels == block.resnets[j].out_channels:
                    rev_module_list.append(
                        RevModule(block.resnets[j], retain_graph=True)
                    )

                rev_module_list.append(
                    RevModule(block.injectors[2 * j], retain_graph=True)
                )

                rev_module_list.append(
                    RevModule(block.attentions[j], retain_graph=True)
                )

                rev_module_list.append(
                    RevModule(block.injectors[2 * j + 1], retain_graph=True)
                )

                block.rev_module_lists.append(rev_module_list)

    def unet_add_up_rev_modules_and_injectors(self, T):
        self.unet.up_blocks[0].injectors = nn.ModuleList([
            Injector(640, 4, T)
            for _ in range(6)
        ])

        self.unet.up_blocks[1].injectors = nn.ModuleList([
            Injector(320, 2, T)
            for _ in range(6)
        ])

        for i in range(2):
            block = self.unet.up_blocks[i]

            block.input_help_scale_factor = nn.Parameter(torch.ones(1))
            block.merge_scale_factor = nn.Parameter(torch.zeros(1))

            rev_module_list = nn.ModuleList([])

            for j in range(3):
                if j > 0:
                    rev_module_list.append(
                        RevModule(block.resnets[j], retain_graph=True)
                    )

                rev_module_list.append(
                    RevModule(block.injectors[2 * j], retain_graph=True)
                )

                rev_module_list.append(
                    RevModule(block.attentions[j], retain_graph=True)
                )

                rev_module_list.append(
                    RevModule(block.injectors[2 * j + 1], retain_graph=True)
                )

            block.rev_module_list = rev_module_list

    # -----------------------------------------------------
    # Remove unused SD components
    # -----------------------------------------------------
    def unet_remove_resnet_time_emb_proj(self):
        def remove_time_emb(module):
            if module.__class__.__name__ == "ResnetBlock2D":
                if hasattr(module, "time_emb_proj"):
                    module.time_emb_proj = None

        self.unet.apply(remove_time_emb)

    def unet_remove_cross_attn(self):
        def remove_cross_attn(module):
            if module.__class__.__name__ == "BasicTransformerBlock":
                if hasattr(module, "attn2"):
                    module.attn2 = None
                if hasattr(module, "norm2"):
                    module.norm2 = None

        self.unet.apply(remove_cross_attn)

    def unet_disable_inplace_ops(self):
        """
        可逆反传 + checkpoint 下，inplace 激活容易引起版本号错误。
        因此这里不再强行 inplace=True，而是尽量关闭。
        """
        def disable_inplace(module):
            if isinstance(module, (nn.SiLU, nn.Dropout)):
                if hasattr(module, "inplace"):
                    module.inplace = False

        self.unet.apply(disable_inplace)

    # -----------------------------------------------------
    # Replace forward methods
    # -----------------------------------------------------
    def unet_replace_forward_methods(self):
        """
        使用 class name 判断，减少 diffusers 版本路径变化带来的导入问题。
        """
        def replace_forward_methods(module):
            name = module.__class__.__name__

            if name == "CrossAttnDownBlock2D":
                module.forward = types.MethodType(
                    MyCrossAttnDownBlock2D_SD_v1_5_forward,
                    module,
                )

            elif name == "CrossAttnUpBlock2D":
                module.forward = types.MethodType(
                    MyCrossAttnUpBlock2D_SD_v1_5_forward,
                    module,
                )

            elif name == "ResnetBlock2D":
                module.forward = types.MethodType(
                    MyResnetBlock2D_SD_v1_5_forward,
                    module,
                )

            elif name == "Transformer2DModel":
                module.forward = types.MethodType(
                    MyTransformer2DModel_SD_v1_5_forward,
                    module,
                )

        self.unet.apply(replace_forward_methods)

        self.unet.forward = types.MethodType(
            MyUNet2DConditionModel_SD_v1_5_forward,
            self.unet,
        )

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(self, y_, A_, AT_, use_amp_=True):
        """
        y_:
            压缩测量。

        A_:
            forward CS operator.

        AT_:
            transpose CS operator.

        输出：
            [B,1,H,W]

        注意：
            当前模型本质是单通道重构器。
            RGB 训练时建议把 [B,3,H,W] reshape 成 [B*3,1,H,W]，
            这比连续调用 forward_rgb 更安全。
        """
        global y, A, AT, unet, ATy
        global alpha_bar, alpha_global, dc_global, use_amp

        y = y_
        A = A_
        AT = AT_
        unet = self.unet
        use_amp = bool(use_amp_)

        alpha_global = self._effective_alpha()
        dc_global = self._effective_dc_step()

        alpha_bar = torch.cat(
            [
                torch.ones(
                    1,
                    device=alpha_global.device,
                    dtype=alpha_global.dtype,
                ),
                alpha_global.cumprod(dim=0),
            ],
            dim=0,
        )

        x = AT(y)

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"MyIDM Net expects AT(y) to be [B,1,H,W], got {tuple(x.shape)}"
            )

        if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
            raise ValueError(
                f"MyIDM Net requires even H/W, got {tuple(x.shape)}"
            )

        ATy = x

        help_scale = _ste_clamp(
            self.input_help_scale_factor,
            0.0,
            2.0,
        ).to(device=x.device, dtype=x.dtype)

        init_scale = alpha_bar[-1].clamp(min=1e-6, max=1.0).sqrt()
        init_scale = init_scale.to(device=x.device, dtype=x.dtype)

        x = init_scale * torch.cat([x, help_scale * x], dim=1).contiguous()

        x = RevBackProp.apply(x, self.body)

        merge_scale = _ste_clamp(
            self.merge_scale_factor,
            -1.0,
            1.0,
        ).to(device=x.device, dtype=x.dtype)

        out = x[:, :1] + merge_scale * x[:, 1:]

        return out

    @torch.no_grad()
    def forward_rgb(self, y_rgb, A_, AT_, use_amp_=False):
        """
        仅建议验证/推理时使用。

        训练 RGB 时不要连续调用 forward_rgb，
        因为 MyIDM 当前使用全局可逆上下文，多次 forward 后再 backward 会覆盖上下文。

        训练 RGB 推荐写法：
            x: [B,3,H,W]
            x_flat = x.reshape(B*3, 1, H, W)
            y = A(x_flat)
            rec_flat = model(y, A, AT)
            rec = rec_flat.reshape(B,3,H,W)
        """
        if torch.is_grad_enabled():
            raise RuntimeError(
                "forward_rgb is inference-only. "
                "For training, flatten RGB channels into batch dimension."
            )

        if isinstance(y_rgb, (list, tuple)):
            outs = [
                self.forward(y_c, A_, AT_, use_amp_=use_amp_)
                for y_c in y_rgb
            ]
            return torch.cat(outs, dim=1)

        if y_rgb.ndim < 3 or y_rgb.shape[1] != 3:
            raise ValueError(
                f"forward_rgb expects y_rgb with channel dimension 3, "
                f"got shape={tuple(y_rgb.shape)}"
            )

        outs = []

        for c in range(3):
            outs.append(
                self.forward(y_rgb[:, c], A_, AT_, use_amp_=use_amp_)
            )

        return torch.cat(outs, dim=1)

    def extra_repr(self):
        return f"T={self.T}"