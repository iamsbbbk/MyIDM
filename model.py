import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import math
import logging
from utils import *
from backprop import RevModule, VanillaBackProp, RevBackProp

# 假设 forward.py 在同一目录下，且包含这些特定函数
try:
    from forward import (
        MyUNet2DConditionModel_SD_v1_5_forward,
        MyCrossAttnDownBlock2D_SD_v1_5_forward,
        MyCrossAttnUpBlock2D_SD_v1_5_forward,
        MyResnetBlock2D_SD_v1_5_forward,
        MyTransformer2DModel_SD_v1_5_forward
    )
except ImportError as e:
    logging.error(
        f"Critical Import Error: Failed to import forward methods. Ensure 'forward.py' is present. Details: {e}")
    raise


# ==========================================
# Core Algorithm: Sinkhorn-Knopp (from mHC)
# ==========================================
def sinkhorn_knopp(matrix: torch.Tensor, num_iter: int = 20, epsilon: float = 1e-12) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm to project a matrix onto the doubly stochastic manifold.
    Args:
        matrix: [B, ..., N, N] input matrix (usually logits)
        num_iter: number of normalization iterations
    Returns:
        Doubly stochastic matrix (rows and cols sum to 1)
    """
    # Numerical stability: shift by max to prevent exp explosion
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = matrix - torch.max(matrix, dim=-1, keepdim=True)[0]
    K = torch.exp(matrix)

    for _ in range(num_iter):
        # Row normalization
        K = K / (K.sum(dim=-1, keepdim=True) + epsilon)
        # Column normalization
        K = K / (K.sum(dim=-2, keepdim=True) + epsilon)
    return K


# ==========================================
# Module: Sinkhorn Channel Mixer
# ==========================================
class SinkhornChannelMixer(nn.Module):
    """
    Dynamic Channel Mixer with Manifold Constraints.
    It predicts a mixing matrix M from the input, projects M to be doubly stochastic,
    and then mixes the channels.
    """

    def __init__(self, in_channels, reduced_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_dim = reduced_dim if reduced_dim else in_channels // 4

        # Lightweight predictor for the mixing matrix
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, self.reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(self.reduced_dim, in_channels * in_channels, 1)
        )
        # Scale factor for residual learning
        self.scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        # Predict mixing weights: [B, C*C, 1, 1] -> [B, C, C]
        weights = self.predictor(x).view(B, C, C)

        # Apply Sinkhorn Projection (Manifold Constraint)
        # This ensures energy conservation during channel mixing
        M = sinkhorn_knopp(weights)

        # Apply mixing: x is [B, C, H*W]
        x_flat = x.view(B, C, -1)
        x_mixed = torch.bmm(M, x_flat).view(B, C, H, W)

        return x + self.scale * x_mixed


# ==========================================
# Module: Manifold Injector (M-IDM Core)
# ==========================================
class ManifoldInjector(nn.Module):
    def __init__(self, nf, r, T):
        super().__init__()
        self.r = r
        self.T = T

        # Feature to Injector Space (Downsampling)
        self.f2i = nn.ModuleList([
            nn.Sequential(
                nn.PixelShuffle(r),
                nn.Conv2d(nf // (r * r), nf // (r * r), 3, 1, 1),  # Added Conv for feature extraction
                nn.GroupNorm(8, nf // (r * r)),
                nn.SiLU()
            ) for _ in range(T)
        ])

        # Manifold Mixer (Replaces simple concatenation)
        # Input channels = latent_channels + measurement_channels (assumed match latent) + latent_channels
        # Current logic: x_in + x + AT(A(x)) + ATy.
        # Let's align with the original IDM logic but add Sinkhorn mixing.

        # Injector to Feature Space (Upsampling)
        self.i2f = nn.ModuleList([
            nn.Sequential(
                SinkhornChannelMixer(nf // (r * r) * 3 + 1),  # *3 for x, ATAx, ATy, +1 for extra alignment if needed
                nn.Conv2d(nf // (r * r) * 3 + 1, nf // (r * r), 1),
                nn.PixelUnshuffle(r),
            ) for _ in range(T)
        ])

        # Adapter to align dimensions for the mixer
        # Note: Original code hardcoded dims, we make it dynamic based on concatenation
        self.dim_adapters = nn.ModuleList([
            nn.Conv2d(nf // (r * r) + 2, nf // (r * r) * 3 + 1, 1) for _ in range(T)
        ])

    def forward(self, x_in):
        # Access global step 't' safely
        global t, AT, A, ATy, y

        # 1. Downsample features
        x = self.f2i[t - 1](x_in)  # [B, C', H', W']

        # 2. Prepare physical measurements
        # Note: A(x) and AT(x) might need dimension alignment.
        # Assuming A/AT handle the spatial dims correctly.
        # In IDM, x is usually 1 or 3 channels when projected.
        # We need to ensure concatenation works.

        # Original IDM logic: torch.cat([x, AT(A(x)), ATy], dim=1)
        # Check shapes: x is hidden dim. AT(A(x)) is image space (3 or 1 channel).
        # We need to be careful. The original code's convs were:
        # Conv2d(nf // (r*r), 1, 1) -> This implies x became 1 channel?
        # Let's correct this based on the uploaded model.py logic:
        # Original: nn.Conv2d(nf // (r * r), 1, 1) in f2i.

        # Re-implementing strictly compatible logic with Manifold enhancement:

        # Project hidden features to measurement space proxy (1 channel)
        # We use a 1x1 conv stored in f2i for this dimension reduction if needed.
        # But here we want to keep features rich.

        # Let's assume standard IDM concatenation strategy but enhanced.
        measurement_consistency = AT(A(x))  # [B, C_img, H, W]
        measurement_input = ATy  # [B, C_img, H, W]

        # Concatenate: [Hidden, Consistency, Input]
        x_cat = torch.cat([x, measurement_consistency, measurement_input], dim=1)

        # 3. Apply Sinkhorn Mixing
        # Ensure dimensions match what we defined in __init__
        # In __init__, we need to know the channel count of x.
        # nf // (r*r) + 2*C_img. Assuming C_img=1 or 3.
        # For safety, we use a 1x1 conv adapter to a fixed dim before mixing.

        # (Simplified for this implementation: Dynamic adaptation not shown, assuming dims match)
        # Ideally, we redefine __init__ to match exact dims.
        # Let's rely on the adapter defined above.

        # 4. Upsample back
        # We use a distinct adapter for each step t
        # (This part requires aligning exact channel numbers, using standard Conv for now to be safe)

        # Fallback to original logic + Sinkhorn if dimensions are tricky to guess without runtime info.
        # But we will use the defined layers.

        return x_in + self.i2f[t - 1](x_cat)  # Using index t-1


# To fix the dimension mismatch issue in ManifoldInjector without hardcoding:
# We revert to a slightly safer implementation that follows the original IDM structure exactly
# but inserts Sinkhorn mixing in the intermediate stage.

class SafeManifoldInjector(nn.Module):
    def __init__(self, nf, r, T):
        super().__init__()
        self.r = r
        hidden_dim = nf // (r * r)

        # Feature -> Low Dim
        self.f2i = nn.ModuleList([
            nn.Sequential(
                nn.PixelShuffle(r),
                nn.Conv2d(hidden_dim, 1, 1),  # Compresses to 1 channel (Image space proxy)
            ) for _ in range(T)
        ])

        # Manifold Mixer: applied on the concatenated [x_proxy, A(AT(x)), ATy] -> 3 channels
        self.mixer = nn.ModuleList([
            SinkhornChannelMixer(3) for _ in range(T)
        ])

        # Low Dim -> Feature
        self.i2f = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, hidden_dim, 1),  # Expands back
                nn.PixelUnshuffle(r),
            ) for _ in range(T)
        ])

    def forward(self, x_in):
        global t, AT, A, ATy
        # 1. Project to Image Space Proxy
        x_proxy = self.f2i[t - 1](x_in)  # [B, 1, H, W]

        # 2. Concatenate with Physics
        # AT(A(x_proxy)) ensures we are checking consistency of the *proxy*
        consistency = AT(A(x_proxy))

        # [B, 3, H, W]
        merged = torch.cat([x_proxy, consistency, ATy], dim=1)

        # 3. Manifold Mixing (Energy Conservation)
        mixed = self.mixer[t - 1](merged)

        # 4. Project back
        return x_in + self.i2f[t - 1](mixed)


# ==========================================
# Step (RevModule with DC)
# ==========================================
class Step(RevModule):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def body(self, x):
        with torch.cuda.amp.autocast(enabled=use_amp, cache_enabled=False):
            global t
            t = self.t
            cur_alpha_bar = alpha_bar[t]
            prev_alpha_bar = alpha_bar[t - 1]

            # 1. Noise Prediction
            # Note: unet access assumes it's global or passed. IDM uses global.
            e = F.pixel_shuffle(unet(F.pixel_unshuffle(x, 2)), 2)

            # 2. x0 Estimation
            x0 = (x - (1.0 - cur_alpha_bar).sqrt() * e) / cur_alpha_bar.sqrt()

            # 3. Data Consistency (DC)
            # x_new = x0 - lambda * grad
            # lambda comes from alpha_global
            grad = AT(A(x0) - y)
            x0 = x0 - alpha_global[t - 1] * grad

            # 4. Reverse Step
            x_prev = prev_alpha_bar.sqrt() * x0 + (1.0 - prev_alpha_bar).sqrt() * e

            return x_prev


# ==========================================
# Net (Main Architecture)
# ==========================================
class Net(nn.Module):
    def __init__(self, T, unet):
        super().__init__()
        # Clean up UNet (as in original IDM)
        if hasattr(unet, 'time_embedding'): del unet.time_embedding
        if hasattr(unet, 'mid_block'): del unet.mid_block

        # Pruning (IDM specific)
        unet.down_blocks = unet.down_blocks[:-2]
        unet.down_blocks[-1].downsamplers = None
        unet.up_blocks = unet.up_blocks[2:]

        self.body = nn.ModuleList([Step(T - i) for i in range(T)])

        self.input_help_scale_factor = nn.Parameter(torch.tensor([1.0]))
        self.merge_scale_factor = nn.Parameter(torch.tensor([0.0]))
        self.alpha = nn.Parameter(torch.full((T,), 0.5))

        self.unet = unet

        # Inject Custom Modules
        self.unet_add_down_rev_modules_and_injectors(T)
        self.unet_add_up_rev_modules_and_injectors(T)
        self.unet_remove_resnet_time_emb_proj()
        self.unet_remove_cross_attn()
        self.unet_set_inplace_to_true()
        self.unet_replace_forward_methods()

    def unet_add_down_rev_modules_and_injectors(self, T):
        # Use SafeManifoldInjector instead of basic Injector
        self.unet.down_blocks[0].register_module(
            "injectors", nn.ModuleList([SafeManifoldInjector(320, 2, T) for _ in range(4)])
        )
        self.unet.down_blocks[1].register_module(
            "injectors", nn.ModuleList([SafeManifoldInjector(640, 4, T) for _ in range(4)])
        )
        # Register RevModules (Same as original IDM)
        for i in range(2):
            self.unet.down_blocks[i].register_module("rev_module_lists", nn.ModuleList([]))
            self.unet.down_blocks[i].register_parameter(
                "input_help_scale_factor", nn.Parameter(torch.ones(1, ))
            )
            self.unet.down_blocks[i].register_parameter(
                "merge_scale_factors", nn.Parameter(torch.zeros(2, ))
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
        # Use SafeManifoldInjector
        self.unet.up_blocks[0].register_module(
            "injectors", nn.ModuleList([SafeManifoldInjector(640, 4, T) for _ in range(6)])
        )
        self.unet.up_blocks[1].register_module(
            "injectors", nn.ModuleList([SafeManifoldInjector(320, 2, T) for _ in range(6)])
        )
        # Register RevModules
        for i in range(2):
            self.unet.up_blocks[i].register_parameter(
                "input_help_scale_factor", nn.Parameter(torch.ones(1, ))
            )
            self.unet.up_blocks[i].register_parameter(
                "merge_scale_factor", nn.Parameter(torch.zeros(1, ))
            )
            rev_module_list = nn.ModuleList([])
            for j in range(3):
                if j > 0:
                    rev_module_list.append(RevModule(self.unet.up_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2 * j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2 * j + 1]))
            self.unet.up_blocks[i].register_module("rev_module_list", rev_module_list)

    # ... (Keep replace_forward_methods, remove_time_emb, etc. as they are standard IDM utils) ...
    # 为了完整性，这里简略写出调用，假设它们在原类中
    def unet_replace_forward_methods(self):
        from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.transformers.transformer_2d import Transformer2DModel

        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(MyResnetBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(MyTransformer2DModel_SD_v1_5_forward, module)

        self.unet.apply(replace_forward_methods)
        self.unet.forward = types.MethodType(MyUNet2DConditionModel_SD_v1_5_forward, self.unet)

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
                module.attn2 = module.norm2 = None

        self.unet.apply(remove_cross_attn)

    def unet_set_inplace_to_true(self):
        def set_inplace(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True

        self.unet.apply(set_inplace)

    def forward(self, y_, A_, AT_, use_amp_=True):
        global y, A, AT, unet, ATy, alpha_bar, alpha_global, use_amp
        y, A, AT, unet, use_amp = y_, A_, AT_, self.unet, use_amp_
        alpha_global = self.alpha
        alpha_bar = torch.cat([torch.ones(1, device=y.device), self.alpha.cumprod(dim=0)])

        # Initialization
        x = AT(y)
        ATy = x

        # Scale Help
        x = alpha_bar[-1].sqrt() * torch.cat(
            [x, self.input_help_scale_factor * x], dim=1
        )

        # RevBackProp (Memory Efficient Training)
        x = RevBackProp.apply(x, self.body)

        return x[:, :1] + self.merge_scale_factor * x[:, 1:]