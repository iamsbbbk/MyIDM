from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, act_layer=nn.GELU):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct1d(in_ch, out_ch, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)


class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, C, T]
        """
        x_t = x.transpose(1, 2)  # [B, T, C]
        x_n = self.norm(x_t)
        attn_out, _ = self.attn(x_n, x_n, x_n, need_weights=False)
        x_t = x_t + self.dropout(attn_out)
        return x_t.transpose(1, 2)  # [B, C, T]


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SignalDecoder(nn.Module):
    """
    参考解码器：
    不是强绑定 MyIDM，而是作为训练和 fallback 的参考重构器。
    输入 latent: [B, latent_channels, latent_tokens]
    输出 signal: [B, in_channels, target_length]
    """
    def __init__(self, latent_channels: int, in_channels: int = 3, target_length: int = 1024):
        super().__init__()
        self.target_length = target_length

        self.net = nn.Sequential(
            nn.Conv1d(latent_channels, latent_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_channels * 2, latent_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.mid = nn.Sequential(
            nn.Conv1d(latent_channels * 2, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv1d(latent_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent):
        x = self.net(latent)
        x = F.interpolate(x, size=max(64, self.target_length // 4), mode="linear", align_corners=False)
        x = self.mid(x)
        x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        x = self.out(x)
        return x


def symmetric_quantize_tensor(x: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 batch 内每个样本做对称量化。
    返回:
        q: int16
        scale: [B, 1, 1]
    """
    if num_bits < 2:
        raise ValueError("num_bits 必须 >= 2")

    qmax = (1 << (num_bits - 1)) - 1
    reduce_dims = tuple(range(1, x.ndim))
    max_abs = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    scale = max_abs / qmax
    q = torch.clamp(torch.round(x / scale), min=-qmax, max=qmax).to(torch.int16)
    return q, scale


def symmetric_dequantize_tensor(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()


class SemanticSenderModel(nn.Module):
    """
    发送端语义模型：
    - contact head: idle / mixed / cutting
    - wear head: initial / steady / accelerating
    - mapped head: 0/1/2/3 兼容论文标签
    - latent encoder + reference decoder
    """

    def __init__(self, model_cfg: Dict[str, Any]):
        super().__init__()

        self.in_channels = model_cfg.get("in_channels", 3)
        base_channels = model_cfg.get("base_channels", 64)
        attn_heads = model_cfg.get("attn_heads", 4)
        dropout = model_cfg.get("dropout", 0.1)
        self.latent_channels = model_cfg.get("latent_channels", 48)
        self.latent_tokens = model_cfg.get("latent_tokens", 8)
        self.use_reference_decoder = model_cfg.get("use_reference_decoder", True)
        self.target_length = model_cfg.get("target_length", 1024)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = nn.Sequential(
            ConvBNAct1d(self.in_channels, c1, kernel_size=7, stride=2, padding=3),
            ResidualBlock1D(c1, c1, stride=1, dropout=dropout),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock1D(c1, c2, stride=2, dropout=dropout),
            ResidualBlock1D(c2, c2, stride=1, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(c2, c3, stride=2, dropout=dropout),
            ResidualBlock1D(c3, c3, stride=1, dropout=dropout),
        )
        self.attn = TemporalSelfAttention(c3, num_heads=attn_heads, dropout=dropout)
        self.post = nn.Sequential(
            ResidualBlock1D(c3, c3, stride=1, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        self.contact_head = MLPHead(c3, c3 // 2, 3, dropout=dropout)
        self.wear_head = MLPHead(c3, c3 // 2, 3, dropout=dropout)
        self.mapped_head = MLPHead(c3, c3 // 2, 4, dropout=dropout)

        self.latent_proj = nn.Sequential(
            nn.Conv1d(c3, self.latent_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(self.latent_tokens),
        )

        if self.use_reference_decoder:
            self.reference_decoder = SignalDecoder(
                latent_channels=self.latent_channels,
                in_channels=self.in_channels,
                target_length=self.target_length,
            )
        else:
            self.reference_decoder = None

    @staticmethod
    def hierarchical_mapped_probs(contact_probs: torch.Tensor, wear_probs: torch.Tensor) -> torch.Tensor:
        """
        由层级语义构造兼容论文的 4 类概率:
        class 0 = idle
        class 1/2/3 = non-idle * wear(initial/steady/accelerating)
        """
        idle = contact_probs[:, 0]
        non_idle = 1.0 - idle
        wear_probs = wear_probs / (wear_probs.sum(dim=1, keepdim=True) + 1e-12)

        mapped = torch.stack(
            [
                idle,
                non_idle * wear_probs[:, 0],
                non_idle * wear_probs[:, 1],
                non_idle * wear_probs[:, 2],
            ],
            dim=1,
        )
        mapped = mapped / (mapped.sum(dim=1, keepdim=True) + 1e-12)
        return mapped

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.attn(x)
        latent = self.latent_proj(x)  # [B, latent_channels, latent_tokens]
        global_feat = self.post(x).squeeze(-1)  # [B, C]
        return {
            "feature_map": x,
            "global_feat": global_feat,
            "latent": latent,
        }

    def decode_latent(self, latent: torch.Tensor) -> Optional[torch.Tensor]:
        if self.reference_decoder is None:
            return None
        return self.reference_decoder(latent)

    def forward(
        self,
        x: torch.Tensor,
        quant_bits: Optional[int] = None,
        force_quantize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats = self.extract_features(x)
        global_feat = feats["global_feat"]
        latent = feats["latent"]

        contact_logits = self.contact_head(global_feat)
        wear_logits = self.wear_head(global_feat)
        mapped_logits = self.mapped_head(global_feat)

        contact_probs = torch.softmax(contact_logits, dim=1)
        wear_probs = torch.softmax(wear_logits, dim=1)
        mapped_head_probs = torch.softmax(mapped_logits, dim=1)
        mapped_hier_probs = self.hierarchical_mapped_probs(contact_probs, wear_probs)
        mapped_probs = 0.6 * mapped_head_probs + 0.4 * mapped_hier_probs
        mapped_probs = mapped_probs / (mapped_probs.sum(dim=1, keepdim=True) + 1e-12)

        quantized_latent = None
        latent_scale = None
        latent_for_decode = latent

        if force_quantize and quant_bits is not None:
            quantized_latent, latent_scale = symmetric_quantize_tensor(latent, num_bits=quant_bits)
            latent_for_decode = symmetric_dequantize_tensor(quantized_latent, latent_scale)

        reconstruction = self.decode_latent(latent_for_decode)

        return {
            "contact_logits": contact_logits,
            "wear_logits": wear_logits,
            "mapped_logits": mapped_logits,
            "contact_probs": contact_probs,
            "wear_probs": wear_probs,
            "mapped_head_probs": mapped_head_probs,
            "mapped_hier_probs": mapped_hier_probs,
            "mapped_probs": mapped_probs,
            "latent": latent,
            "quantized_latent": quantized_latent,
            "latent_scale": latent_scale,
            "reconstruction": reconstruction,
        }


def load_sender_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    override_model_cfg: Optional[Dict[str, Any]] = None,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"发送端 checkpoint 不存在: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "config" in ckpt and isinstance(ckpt["config"], dict):
        model_cfg = ckpt["config"].get("model", {})
    else:
        model_cfg = ckpt.get("model_cfg", {})

    if override_model_cfg:
        model_cfg.update(override_model_cfg)

    model = SemanticSenderModel(model_cfg)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, ckpt