"""Attention module factory."""

from __future__ import annotations

from typing import Optional
import torch.nn as nn

from src.models.attention.cbam import CBAMAttention
from src.models.attention.channel import ChannelAttention
from src.models.attention.identity import IdentityPooling
from src.models.attention.soft import SoftAttention
from src.models.attention.spatial import SpatialAttention
from src.utils.config import AttentionConfig


def build_attention_module(config: Optional[AttentionConfig] = None) -> nn.Module:
    """Build attention module from configuration.

    Args:
        config: AttentionConfig instance, or None for paper-aligned default (soft).

    Returns:
        Configured nn.Module conforming to the attention interface:
        forward(x, return_attention_maps=False) -> Tensor or (Tensor, Dict[str, Tensor]).
    """
    cfg = config or AttentionConfig()

    if cfg.type == "soft":
        return SoftAttention(in_channels=cfg.in_channels)
    elif cfg.type == "cbam":
        return CBAMAttention(
            in_channels=cfg.in_channels,
            reduction_ratio=cfg.reduction_ratio,
            kernel_size=cfg.kernel_size,
        )
    elif cfg.type == "channel":
        return ChannelAttention(
            in_channels=cfg.in_channels,
            reduction_ratio=cfg.reduction_ratio,
        )
    elif cfg.type == "spatial":
        return SpatialAttention(
            in_channels=cfg.in_channels,
            kernel_size=cfg.kernel_size,
        )
    elif cfg.type == "none":
        return IdentityPooling(in_channels=cfg.in_channels)
    else:
        raise ValueError(f"Unknown attention type: '{cfg.type}'")
