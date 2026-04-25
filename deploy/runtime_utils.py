# deploy/runtime_utils.py
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


def pad_to_block_np(x: np.ndarray, block_size: int):
    """
    x:
      - [H, W]
      - [C, H, W]
    返回:
      x_pad, old_h, old_w, new_h, new_w
    """
    if x.ndim == 2:
        x = x[None, ...]
        squeeze_back = True
    else:
        squeeze_back = False

    c, old_h, old_w = x.shape
    pad_h = (block_size - old_h % block_size) % block_size
    pad_w = (block_size - old_w % block_size) % block_size

    x_pad = np.pad(
        x,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="constant"
    )

    if squeeze_back:
        x_pad = x_pad[0]

    return x_pad, old_h, old_w, x_pad.shape[-2], x_pad.shape[-1]


def infer_hw_from_blocks(num_blocks: int, block_size: int):
    """
    如果 meta 没给 height/width，尝试从 block 数反推方形尺寸。
    """
    side_blocks = int(round(math.sqrt(num_blocks)))
    if side_blocks * side_blocks != num_blocks:
        raise ValueError(
            f"无法从 num_blocks={num_blocks} 推断方形尺寸，请在 meta 中显式提供 height / width"
        )
    h = side_blocks * block_size
    w = side_blocks * block_size
    return h, w


def make_A_AT_for_shape(height: int, width: int, block_size: int, Phi: torch.Tensor):
    """
    和 train.py 的 make_A_AT_for_patch 保持一致，但支持 H×W。
    Phi: [N, q]
    x:   [Bflat, 1, H, W]
    y:   [Bflat, q, L]
    """
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            f"height/width 必须能被 block_size 整除，当前 {(height, width)} vs block_size={block_size}"
        )

    unfold = nn.Unfold(kernel_size=block_size, stride=block_size)
    fold = nn.Fold(output_size=(height, width), kernel_size=block_size, stride=block_size)

    def A(x: torch.Tensor) -> torch.Tensor:
        # unfold(x): [Bflat, N, L]
        return torch.matmul(Phi.t(), unfold(x))  # [Bflat, q, L]

    def AT(y: torch.Tensor) -> torch.Tensor:
        # y: [Bflat, q, L]
        return fold(torch.matmul(Phi, y))  # [Bflat, 1, H, W]

    return A, AT