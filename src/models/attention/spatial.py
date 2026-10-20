"""Spatial Attention Module (Project Ablation)."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) ablation.

    Aggregates channel dimensions using average and max operations,
    concatenates the resulting 2D maps, and applies a 2D convolution
    followed by Sigmoid activation to compute spatial importance weights.

    Operates strictly per frame with zero temporal mixing.
    """

    def __init__(self, in_channels: int = 2048, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass for spatial attention.

        Args:
            x: Input tensor of shape (B, 2048, 4, 4) or (B, T, 2048, 4, 4).
            return_attention_maps: If True, returns (pooled, {"spatial_map": map}).

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

        avg_out = torch.mean(x_flat, dim=1, keepdim=True)
        max_out, _ = torch.max(x_flat, dim=1, keepdim=True)
        cat_out = torch.cat([avg_out, max_out], dim=1)

        spatial_weights = torch.sigmoid(self.conv(cat_out))
        modulated = x_flat * spatial_weights
        pooled = self.global_pool(modulated).flatten(1)

        if is_5d:
            pooled = pooled.view(b, t, self.in_channels)
            spatial_map = spatial_weights.view(b, t, 1, 4, 4)
        else:
            spatial_map = spatial_weights

        if return_attention_maps:
            return pooled, {"spatial_map": spatial_map}
        return pooled
