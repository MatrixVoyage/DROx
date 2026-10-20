"""Identity Pooling Baseline (Project Ablation)."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
import torch.nn as nn


class IdentityPooling(nn.Module):
    """Identity baseline performing Global Average Pooling with no attention.

    Serves as the non-attention ablation baseline to measure the marginal
    contribution of attention mechanisms.

    Operates strictly per frame with zero temporal mixing.
    """

    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.in_channels = in_channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass for identity pooling.

        Args:
            x: Input tensor of shape (B, 2048, 4, 4) or (B, T, 2048, 4, 4).
            return_attention_maps: If True, returns (pooled, {}).

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

        pooled = self.global_pool(x_flat).flatten(1)

        if is_5d:
            pooled = pooled.view(b, t, self.in_channels)

        if return_attention_maps:
            return pooled, {}
        return pooled
