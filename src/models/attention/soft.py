"""Soft Spatial Attention mechanism (Paper-aligned baseline from Lal et al., 2025)."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
import torch.nn as nn


class SoftAttention(nn.Module):
    """Paper-aligned soft spatial attention over ResNeXt-50 feature maps.

    Computes spatial attention logits via a 1x1 2D convolution, normalizes
    them across spatial locations using 2D Softmax, and computes a 2048-D
    context vector as the weighted spatial sum of features.

    Operates strictly per frame with zero temporal mixing.
    Attention maps represent intermediate model feature visualizations,
    not verified ground-truth pixel manipulation masks.
    """

    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.in_channels = in_channels
        self.projector = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass for soft spatial attention.

        Args:
            x: Input feature maps of shape (B, 2048, 4, 4) or (B, T, 2048, 4, 4).
            return_attention_maps: If True, returns (context_vector, {"soft_map": map}).

        Returns:
            Context vector of shape (B, 2048) or (B, T, 2048).
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

        # Compute spatial attention logits: (N, 1, 4, 4)
        logits = self.projector(x_flat)

        # Spatial softmax across all 16 spatial cells: (N, 1, 4, 4)
        attn_weights = torch.softmax(logits.view(-1, 1, 16), dim=-1).view(-1, 1, 4, 4)

        # Weighted spatial sum: (N, C)
        context = (x_flat * attn_weights).sum(dim=(-1, -2))

        if is_5d:
            context = context.view(b, t, self.in_channels)
            attn_map = attn_weights.view(b, t, 1, 4, 4)
        else:
            attn_map = attn_weights

        if return_attention_maps:
            return context, {"soft_map": attn_map}
        return context
