"""Learned Local Frequency Residual Branch for Milestone 13.3.

Provides spatially localized frequency representations:
1. BlockDCTConv2d: Fixed 8x8 2D-DCT orthogonal basis filter bank implemented as a convolution.
2. BayarConv2d: Constrained learnable high-pass filter suppressing semantic skin/facial content.
3. FrequencyResidualBranch: Fuses local Block DCT and learnable high-pass residuals into a compact
   spectral inconsistency representation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_dct_basis_filters(block_size: int = 8) -> torch.Tensor:
    """Generate the full set of 2D DCT orthogonal basis filters as a Conv2d kernel weight tensor.

    Returns:
        Tensor of shape (block_size * block_size, 1, block_size, block_size).
    """
    num_filters = block_size * block_size
    filters = torch.zeros(num_filters, 1, block_size, block_size, dtype=torch.float32)
    idx = 0
    for u in range(block_size):
        for v in range(block_size):
            alpha_u = 1.0 / math.sqrt(block_size) if u == 0 else math.sqrt(2.0 / block_size)
            alpha_v = 1.0 / math.sqrt(block_size) if v == 0 else math.sqrt(2.0 / block_size)
            for i in range(block_size):
                for j in range(block_size):
                    val = (
                        alpha_u
                        * alpha_v
                        * math.cos((2 * i + 1) * u * math.pi / (2 * block_size))
                        * math.cos((2 * j + 1) * v * math.pi / (2 * block_size))
                    )
                    filters[idx, 0, i, j] = val
            idx += 1
    return filters


class BlockDCTConv2d(nn.Module):
    """Local 8x8 Block-DCT decomposition implemented as a fixed 2D convolution.

    Transforms a single-channel image (B, 1, H, W) into (B, 64, H/8, W/8) where each spatial
    position contains the 64 orthogonal DCT frequency coefficients for that 8x8 block.
    """

    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        basis_weights = build_dct_basis_filters(block_size)
        self.register_buffer("weight", basis_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 1, H, W) or (B, C, H, W). If C=3, converts to luminance.

        Returns:
            Tensor of shape (B, block_size^2, H // block_size, W // block_size).
        """
        if x.dim() == 4 and x.size(1) == 3:
            # Luminance conversion
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return F.conv2d(x, self.weight, stride=self.block_size, padding=0)


class BayarConv2d(nn.Module):
    """Constrained high-pass convolutional layer based on Bayar and Stamm (2018).

    Constrains the center weight to -1.0 and normalizes surrounding weights via softmax
    such that their sum is strictly +1.0. This guarantees that constant and smooth semantic
    facial regions produce zero response, isolating high-frequency interpolation residuals.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 8, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.center = kernel_size // 2
        self.num_surround = kernel_size * kernel_size - 1

        # Learnable parameters for surrounding weights
        self.raw_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, self.num_surround) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply constrained high-pass filter.

        Args:
            x: Input image tensor (B, in_channels, H, W).

        Returns:
            Filtered high-pass residual tensor (B, out_channels, H, W).
        """
        if x.size(1) == 3 and self.in_channels == 1:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        w_norm = F.softmax(self.raw_weights, dim=-1)
        center_idx = self.center * self.kernel_size + self.center
        center_val = torch.full(
            (self.out_channels, self.in_channels, 1),
            -1.0,
            device=x.device,
            dtype=x.dtype,
        )
        w = torch.cat([w_norm[:, :, :center_idx], center_val, w_norm[:, :, center_idx:]], dim=-1)
        w = w.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        pad = self.kernel_size // 2
        return F.conv2d(x, w, padding=pad)


class FrequencyResidualBranch(nn.Module):
    """Learned Local Frequency Residual Branch for Milestone 13.3.

    Processes 16-frame video sequences:
    1. Extracts local 8x8 block-DCT coefficients.
    2. Extracts learnable Bayar high-pass residuals.
    3. Fuses both spectral representations via lightweight residual convolutions.
    4. Aggregates across blocks and frames to produce a local frequency embedding.
    """

    def __init__(
        self,
        block_size: int = 8,
        bayar_channels: int = 8,
        hidden_dim: int = 64,
        out_features: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block_dct = BlockDCTConv2d(block_size=block_size)
        self.bayar = BayarConv2d(in_channels=1, out_channels=bayar_channels, kernel_size=5)

        # DCT projection: 64 channels at (14x14) -> 32 channels
        self.dct_proj = nn.Sequential(
            nn.Conv2d(block_size * block_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Bayar downsampling: 8 channels at (112x112) -> stride 8 down to (14x14) -> 32 channels
        self.bayar_proj = nn.Sequential(
            nn.Conv2d(bayar_channels, 16, kernel_size=3, stride=2, padding=1),  # (56x56)
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),              # (14x14)
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Fusion at 14x14 resolution (32 + 32 = 64 channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Video tensor of shape (B, T, C, H, W) or (B*T, C, H, W).

        Returns:
            Tuple of:
            - z_freq: Video-level frequency representation of shape (B, out_features).
            - freq_spatial_map: Frame-level spatial frequency maps of shape (B*T, hidden_dim, 14, 14).
        """
        is_5d = x.dim() == 5
        if is_5d:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
        else:
            x_flat = x
            B = x_flat.size(0)
            T = 1

        # 1. Block DCT: (B*T, 64, 14, 14) -> (B*T, 32, 14, 14)
        dct_feat = self.block_dct(x_flat)
        dct_proj = self.dct_proj(dct_feat)

        # 2. Bayar High-pass: (B*T, 8, 112, 112) -> (B*T, 32, 14, 14)
        bayar_feat = self.bayar(x_flat)
        bayar_proj = self.bayar_proj(bayar_feat)

        # 3. Concatenate and fuse: (B*T, 64, 14, 14) -> (B*T, hidden_dim, 1, 1)
        fused_spatial = torch.cat([dct_proj, bayar_proj], dim=1)
        pooled = self.fusion(fused_spatial)  # (B*T, hidden_dim, 1, 1)

        # 4. Temporal average pooling if input was 5D
        if is_5d:
            pooled_bt = pooled.view(B, T, -1).mean(dim=1)  # (B, hidden_dim)
        else:
            pooled_bt = pooled.view(B, -1)

        z_freq = self.head(pooled_bt)  # (B, out_features)
        return z_freq, fused_spatial
