"""Convolutional Block Attention Module (CBAM) (Project Ablation)."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
import torch.nn as nn


class CBAMAttention(nn.Module):
    """Standard Convolutional Block Attention Module (CBAM) ablation.

    Applies sequential Channel Attention (CAM) followed by Spatial
    Attention (SAM) without residual modification, conforming strictly
    to the standard CBAM formulation (Woo et al., 2018).

    Operates strictly per frame with zero temporal mixing.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        reduction_ratio: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        reduced_channels = in_channels // reduction_ratio

        # Channel Attention components
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
        )

        # Spatial Attention components
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)

        # Output pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass for standard CBAM.

        Args:
            x: Input tensor of shape (B, 2048, 4, 4) or (B, T, 2048, 4, 4).
            return_attention_maps: If True, returns (pooled, {"channel_map": ..., "spatial_map": ...}).

        Returns:
            Pooled feature vector of shape (B, 2048) or (B, T, 2048).
        """
        is_5d = x.dim() == 5
        if is_5d:
            b, t, c, h, w = x.shape
            if (c, h, w) != (self.in_channels, 4, 4):
                raise ValueError(
                    f"Expected input with (C={self.in_channels}, H=4, W=4), got ({c}, {h}, {w})"
                )
            x_flat = x.view(b * t, c, h, w)
        elif x.dim() == 4:
            n, c, h, w = x.shape
            if (c, h, w) != (self.in_channels, 4, 4):
                raise ValueError(
                    f"Expected input with (C={self.in_channels}, H=4, W=4), got ({c}, {h}, {w})"
                )
            x_flat = x
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got {x.dim()}D with shape {tuple(x.shape)}")

        # 1. Channel Attention
        avg_c = self.mlp(self.avg_pool(x_flat).flatten(1))
        max_c = self.mlp(self.max_pool(x_flat).flatten(1))
        channel_weights = torch.sigmoid(avg_c + max_c).view(-1, self.in_channels, 1, 1)
        x_c = x_flat * channel_weights

        # 2. Spatial Attention
        avg_s = torch.mean(x_c, dim=1, keepdim=True)
        max_s, _ = torch.max(x_c, dim=1, keepdim=True)
        cat_s = torch.cat([avg_s, max_s], dim=1)
        spatial_weights = torch.sigmoid(self.conv(cat_s))
        x_s = x_c * spatial_weights

        # 3. Global Pooling
        pooled = self.global_pool(x_s).flatten(1)

        if is_5d:
            pooled = pooled.view(b, t, self.in_channels)
            channel_map = channel_weights.view(b, t, self.in_channels, 1, 1)
            spatial_map = spatial_weights.view(b, t, 1, 4, 4)
        else:
            channel_map = channel_weights
            spatial_map = spatial_weights

        if return_attention_maps:
            return pooled, {"channel_map": channel_map, "spatial_map": spatial_map}
        return pooled
