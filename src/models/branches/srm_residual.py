"""Multi-Scale Sensor and Residual Representation Branch for Milestone 17 Track A.

Implements:
1. SRMConv2d: Fixed Spatial Rich Model (SRM) linear high-pass filter bank.
2. BayarConv2d: Constrained learnable high-pass kernel isolating non-semantic residuals.
3. MultiScaleResidualStream: Multi-resolution residual pyramid processing:
   - Scale 1 (112x112): Micro-scale CMOS photon shot noise residuals.
   - Scale 2 (56x56): Mid-scale interpolation and resampling seam residuals.
   - Scale 3 (28x28): Macro-scale block DCT and compression quantization residuals.
4. Pyramidal Residual Aggregator: Spatiotemporal projection and temporal pooling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.branches.frequency_residual import BayarConv2d, BlockDCTConv2d


def build_srm_filters() -> torch.Tensor:
    """Generate canonical Spatial Rich Model (SRM) high-pass filter kernels.

    Returns:
        Tensor of shape (3, 1, 5, 5) containing 3 high-pass filter kernels:
        1. 1st-order difference (KB filter)
        2. 2nd-order edge/corner (KV filter)
        3. 3rd-order Laplacian (Point/Spot filter)
        All kernels have zero-sum weights to completely suppress smooth semantic regions.
    """
    filters = torch.zeros(3, 1, 5, 5, dtype=torch.float32)

    # Filter 1: 1st order difference (KB)
    f1 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 2.0, -1.0, 0.0],
            [0.0, 2.0, -4.0, 2.0, 0.0],
            [0.0, -1.0, 2.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ) / 4.0

    # Filter 2: 2nd order edge/corner (KV)
    f2 = torch.tensor(
        [
            [-1.0, 2.0, -2.0, 2.0, -1.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-2.0, 8.0, -12.0, 8.0, -2.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-1.0, 2.0, -2.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
    ) / 12.0

    # Filter 3: 5x5 Laplacian spot filter
    f3 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, -4.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ) / 4.0

    filters[0, 0] = f1
    filters[1, 0] = f2
    filters[2, 0] = f3
    return filters


class SRMConv2d(nn.Module):
    """Fixed Spatial Rich Model high-pass convolution layer.

    Buffers standard forensic SRM filters as non-trainable weights.
    Transforms input (B, C, H, W) into (B, 3, H, W) high-pass residuals.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        weights = build_srm_filters()
        self.register_buffer("weight", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SRM filtering.

        Args:
            x: Input tensor (B, C, H, W) or (B, 1, H, W). If C=3, converts to luminance.

        Returns:
            High-pass filtered residuals (B, 3, H, W).
        """
        if x.size(1) == 3 and self.in_channels == 1:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return F.conv2d(x, self.weight, padding=2)


class MultiScaleResidualStream(nn.Module):
    """Multi-Scale Sensor and Residual Representation Stream (M17 Track A).

    Extracts forensic residuals across 3 complementary spatial resolutions:
    - Scale 1 (112x112): Micro-scale photon shot noise via SRM + Bayar filters.
    - Scale 2 (56x56): Mid-scale resampling and spatial interpolation seams.
    - Scale 3 (28x28): Macro-scale block DCT and compression quantization residuals.

    Pyramidally aligns all scales to 28x28 and fuses them into a compact representation.
    """

    def __init__(
        self,
        bayar_channels: int = 8,
        out_features: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.out_features = out_features

        # Scale 1: 112x112 (SRM: 3 channels + Bayar: bayar_channels = 11 channels)
        self.srm_scale1 = SRMConv2d(in_channels=1)
        self.bayar_scale1 = BayarConv2d(in_channels=1, out_channels=bayar_channels, kernel_size=5)
        in_scale1 = 3 + bayar_channels

        self.scale1_down = nn.Sequential(
            nn.Conv2d(in_scale1, 32, kernel_size=3, stride=2, padding=1),  # -> 56x56
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),        # -> 28x28
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Scale 2: 56x56 (Interp / Resampling seams via Bayar)
        self.bayar_scale2 = BayarConv2d(in_channels=1, out_channels=bayar_channels, kernel_size=5)
        self.scale2_down = nn.Sequential(
            nn.Conv2d(bayar_channels, 32, kernel_size=3, stride=2, padding=1),  # -> 28x28
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Scale 3: 28x28 (Block DCT residuals)
        self.dct_scale3 = BlockDCTConv2d(block_size=8)  # 28x28 -> 3x3 patches, 64 channels
        # Project 64 DCT basis channels to 32 channels and upsample back to 28x28
        self.scale3_proj = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False),
        )

        # Pyramidal Fusion: 32 (Scale 1) + 32 (Scale 2) + 32 (Scale 3) = 96 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(96, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Lightweight Temporal Transformer Aggregator across sequence frames (T=16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_agg = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Final projection to out_features
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        active_scales: Optional[Tuple[bool, bool, bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Video tensor of shape (B, T, C, H, W) or (B*T, C, H, W) with H=W=112.
            active_scales: Optional 3-tuple (use_scale1, use_scale2, use_scale3) for frozen inference ablation.

        Returns:
            Tuple of:
            - z_res: Video-level sensor residual representation (B, out_features).
            - res_maps: Frame-level fused spatial residual maps (B*T, hidden_dim, 28, 28).
        """
        is_5d = x.dim() == 5
        if is_5d:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
        else:
            x_flat = x
            B = x_flat.size(0)
            T = 1

        # Scale 1: 112x112
        srm_1 = self.srm_scale1(x_flat)
        bayar_1 = self.bayar_scale1(x_flat)
        feat_1 = torch.cat([srm_1, bayar_1], dim=1)
        res_scale1 = self.scale1_down(feat_1)  # (B*T, 32, 28, 28)

        # Scale 2: 56x56
        x_down2 = F.interpolate(x_flat, size=(56, 56), mode="bilinear", align_corners=False)
        bayar_2 = self.bayar_scale2(x_down2)
        res_scale2 = self.scale2_down(bayar_2)  # (B*T, 32, 28, 28)

        # Scale 3: 28x28 (Block DCT)
        x_down4 = F.interpolate(x_flat, size=(28, 28), mode="bilinear", align_corners=False)
        # Pad to multiple of 8 if needed (28 -> 32)
        pad_h = (8 - (28 % 8)) % 8
        pad_w = (8 - (28 % 8)) % 8
        x_padded = F.pad(x_down4, (0, pad_w, 0, pad_h), mode="reflect")
        dct_3 = self.dct_scale3(x_padded)
        res_scale3 = self.scale3_proj(dct_3)    # (B*T, 32, 28, 28)

        # Apply inference scale mask if specified
        if active_scales is not None:
            use_s1, use_s2, use_s3 = active_scales
            if not use_s1:
                res_scale1 = torch.zeros_like(res_scale1)
            if not use_s2:
                res_scale2 = torch.zeros_like(res_scale2)
            if not use_s3:
                res_scale3 = torch.zeros_like(res_scale3)

        # Multi-scale concatenation at 28x28
        pyramid = torch.cat([res_scale1, res_scale2, res_scale3], dim=1)  # (B*T, 96, 28, 28)
        pooled = self.fusion(pyramid).view(B, T, -1) if is_5d else self.fusion(pyramid).view(B, -1)

        # Temporal aggregation
        if is_5d:
            temporal_out = self.temporal_agg(pooled)  # (B, T, hidden_dim)
            video_pooled = temporal_out.mean(dim=1)   # (B, hidden_dim)
        else:
            video_pooled = pooled

        z_res = self.proj(video_pooled)  # (B, out_features)
        return z_res, pyramid
