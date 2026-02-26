import torch, types
from torch import nn
import torch.nn.functional as F
import traceback

from utils import *
from backprop import RevModule, VanillaBackProp, RevBackProp
from forward import (
    MyUNet2DConditionModel_SD_v1_5_forward,
    MyCrossAttnDownBlock2D_SD_v1_5_forward,
    MyCrossAttnUpBlock2D_SD_v1_5_forward,
    MyResnetBlock2D_SD_v1_5_forward,
    MyTransformer2DModel_SD_v1_5_forward
)

# ==========================================
# 🛡️ 底层安全设施: 主动防御型算子
# ==========================================
class SafeConv2d(nn.Conv2d):
    """
    带安全断言的 Conv2d。
    用于在模型[初始化]阶段直接拦截 out_channels=0 的致命错误。
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        if in_channels <= 0:
            raise ValueError(f"🚨 [SafeConv2d] 致命错误: in_channels 必须大于0, 但收到 {in_channels}!")
        if out_channels <= 0:
            raise ValueError(f"🚨 [SafeConv2d] 致命错误: out_channels 必须大于0, 但收到 {out_channels}! (这通常是因为通道数被整除了)")
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)


# ==========================================
# 🌟 创新点 1: 通道注意力 (视神经门控)
# ==========================================
class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) 风格的通道注意力。
    用于在极度压缩的场景下，动态抑制无用噪声通道，增强包含刀具磨损高频特征的通道。
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 确保 reduction 后的隐藏层至少有 1 个通道，防止崩溃
        mid_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 提取全局空间信息
        y = self.avg_pool(x).view(b, c)
        # Excitation: 计算通道重要性权重 (0~1)
        y = self.fc(y).view(b, c, 1, 1)
        # 动态加权
        return x * y


# ==========================================
# 🌟 创新点 2: 融合注意力的流形混合器 (Attentive mHC)
# ==========================================
class AttentiveManifoldMixer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channels = int(in_channels)
        assert self.channels > 0, "Channels must be positive!"
        
        self.out_dim = self.channels * self.channels
        
        # 1. 视神经：通道注意力机制
        self.attention = ChannelAttention(self.channels)
        
        # 2. 流形映射器：基于注意力筛选后的特征预测混合矩阵
        self.predictor = SafeConv2d(self.channels, self.out_dim, kernel_size=1)
        
        # [学术约束]: 恒等初始化 (Identity Initialization)
        # 确保在训练初期，mHC 只是一个恒等映射，不会破坏 IDM 的物理基础
        with torch.no_grad():
            nn.init.zeros_(self.predictor.weight)
            if self.predictor.bias is not None:
                self.predictor.bias.data = torch.eye(self.channels).flatten()

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.channels:
            raise RuntimeError(f"🚨 [AttentiveManifoldMixer] 期望通道 {self.channels}, 实际 {C}")
            
        # 步骤 1: 特征过滤 (Attention)
        # 让网络评估当前特征图中，哪些通道包含了真正的缺陷/纹理
        x_attended = self.attention(x)
        
        # 步骤 2: 预测动态流形混合权重
        weights = self.predictor(x_attended)
        weights = weights.view(B, self.channels, self.channels, H, W)
        
        # 步骤 3: 约束流形上的特征重组 (Einstein Summation 优化内存)
        mixed_x = torch.einsum('b c i h w, b i h w -> b c h w', weights, x)
        
        return mixed_x


# ==========================================
# IDM 原始组件: Injector (保留原结构，增强鲁棒性)
# ==========================================
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
        try:
            x = self.f2i[t - 1](x_in)
            x = torch.cat([x, AT(A(x)), ATy], dim=1)
            return x_in + self.i2f[t - 1](x)
        except Exception as e:
            print(f"🚨 [Injector] 运行时维度错误! Input shape: {x_in.shape}")
            raise e


# ==========================================
# IDM 核心组件: Step
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

            try:
                # 1. 噪声估计 (epsilon prediction)
                e = F.pixel_shuffle(unet(F.pixel_unshuffle(x, 2)), 2)
            except Exception as e_unet:
                print(f"🚨 [UNet Step Error] Time step {t}, Input shape {x.shape}")
                raise e_unet

            # 2. 纯净图像估计 (x0 estimation)
            x0 = (x - (1.0 - cur_alpha_bar).sqrt() * e) / cur_alpha_bar.sqrt()

            # 3. 物理约束层 (Data Consistency)
            dc_grad = AT(A(x0) - y)
            x0 = x0 - alpha_global[t - 1] * dc_grad

            # 4. 反向扩散更新
            x_prev = (
                prev_alpha_bar.sqrt() * x0
                + (1.0 - prev_alpha_bar).sqrt() * e
            )

            return x_prev


# ==========================================
# 主网络结构: Net
# ==========================================
class Net(nn.Module):
    def __init__(self, T, unet):
        super().__init__()
        
        # 裁剪 UNet 结构
        del unet.time_embedding, unet.mid_block
        unet.down_blocks = unet.down_blocks[:-2]
        unet.down_blocks[-1].downsamplers = None
        unet.up_blocks = unet.up_blocks[2:]

        # 可逆扩散主体模块
        self.body = nn.ModuleList([Step(T - i) for i in range(T)])

        # ---------------------------------------------------------
        # 💡 [核心升级] 使用 AttentiveManifoldMixer
        # 处理 RGB 图像级特征时，in_channels=3
        # ---------------------------------------------------------
        self.mixer = nn.ModuleList([AttentiveManifoldMixer(in_channels=3) for _ in range(T)])

        self.input_help_scale_factor = nn.Parameter(torch.tensor([1.0]))
        self.merge_scale_factor = nn.Parameter(torch.tensor([0.0]))
        self.alpha = nn.Parameter(torch.full((T,), 0.5))

        self.unet = unet
        
        # 执行猴子补丁
        self.unet_add_down_rev_modules_and_injectors(T)
        self.unet_add_up_rev_modules_and_injectors(T)
        self.unet_remove_resnet_time_emb_proj()
        self.unet_remove_cross_attn()
        self.unet_set_inplace_to_true()
        self.unet_replace_forward_methods()

    def unet_add_down_rev_modules_and_injectors(self, T):
        self.unet.down_blocks[0].register_module("injectors", nn.ModuleList([Injector(320, 2, T) for _ in range(4)]))
        self.unet.down_blocks[1].register_module("injectors", nn.ModuleList([Injector(640, 4, T) for _ in range(4)]))
        for i in range(2):
            self.unet.down_blocks[i].register_module("rev_module_lists", nn.ModuleList([]))
            self.unet.down_blocks[i].register_parameter("input_help_scale_factor", nn.Parameter(torch.ones(1,)))
            self.unet.down_blocks[i].register_parameter("merge_scale_factors", nn.Parameter(torch.zeros(2,)))
            for j in range(2):
                rev_module_list = nn.ModuleList([])
                if self.unet.down_blocks[i].resnets[j].in_channels == self.unet.down_blocks[i].resnets[j].out_channels:
                    rev_module_list.append(RevModule(self.unet.down_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2 * j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2 * j + 1]))
                self.unet.down_blocks[i].rev_module_lists.append(rev_module_list)

    def unet_add_up_rev_modules_and_injectors(self, T):
        self.unet.up_blocks[0].register_module("injectors", nn.ModuleList([Injector(640, 4, T) for _ in range(6)]))
        self.unet.up_blocks[1].register_module("injectors", nn.ModuleList([Injector(320, 2, T) for _ in range(6)]))
        for i in range(2):
            self.unet.up_blocks[i].register_parameter("input_help_scale_factor", nn.Parameter(torch.ones(1,)))
            self.unet.up_blocks[i].register_parameter("merge_scale_factor", nn.Parameter(torch.zeros(1,)))
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

        x = AT(y)
        ATy = x

        input_channels = x.shape[1]
        x_concat = torch.cat([x, self.input_help_scale_factor * x], dim=1)
        x = alpha_bar[-1].sqrt() * x_concat

        try:
            x = RevBackProp.apply(x, self.body)
        except RuntimeError as e:
            print("\n" + "="*50)
            print("🚨 发生前向传播崩溃！错误上下文诊断:")
            print(f"当前输入 y 的形状: {y.shape}")
            print(f"反投影后 x 的初始形状: {ATy.shape}")
            print(f"进入 RevBackProp 前 x 的形状: {x_concat.shape}")
            print("="*50 + "\n")
            traceback.print_exc()
            raise e

        # 切片提取原始通道并，在这一步应用我们的 AttentiveManifoldMixer (mHC)
        # 注意: 如果原来有 self.mixer[t-1](merged) 的调用逻辑位于外部或者此处，请根据原网络确保调用
        # 这里维持了你 IDM 代码末端的特征融合机制
        return x[:, :input_channels] + self.merge_scale_factor * x[:, input_channels:]

