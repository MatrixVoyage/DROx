"""Channel Attention Module (Project Ablation)."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) ablation.

    Aggregates spatial dimensions using average and max pooling, processes
    them through a shared MLP (reduction ratio r), and scales feature channels
    via Sigmoid activation before global pooling.

    Operates strictly per frame with zero temporal mixing.
    """

    def __init__(self, in_channels: int = 2048, reduction_ratio: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        reduced_channels = in_channels // reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass for channel attention.

        Args:
            x: Input tensor of shape (B, 2048, 4, 4) or (B, T, 2048, 4, 4).
            return_attention_maps: If True, returns (pooled, {"channel_map": map}).

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

        avg_out = self.mlp(self.avg_pool(x_flat).flatten(1))
        max_out = self.mlp(self.max_pool(x_flat).flatten(1))
        channel_weights = torch.sigmoid(avg_out + max_out).view(-1, self.in_channels, 1, 1)

        modulated = x_flat * channel_weights
        pooled = self.global_pool(modulated).flatten(1)

        if is_5d:
            pooled = pooled.view(b, t, self.in_channels)
            channel_map = channel_weights.view(b, t, self.in_channels, 1, 1)
        else:
            channel_map = channel_weights

        if return_attention_maps:
            return pooled, {"channel_map": channel_map}
        return pooled
