"""Spatial backbone factory supporting ResNeXt-50 and EfficientNet-B0."""

from __future__ import annotations

from typing import Optional
import torch.nn as nn

from src.models.backbones.efficientnet import EfficientNetBackbone
from src.models.backbones.resnext import ResNeXt50Backbone
from src.utils.config import BackboneConfig


def build_backbone(config: Optional[BackboneConfig] = None) -> nn.Module:
    """Instantiate spatial backbone from configuration.

    Args:
        config: BackboneConfig instance or None for default (resnext50_32x4d).

    Returns:
        Backbone nn.Module conforming to spatial feature extraction contract.
    """
    cfg = config or BackboneConfig()

    name = cfg.name.lower()
    if name == "resnext50_32x4d":
        return ResNeXt50Backbone(cfg)
    elif name in ("efficientnet_b0", "efficientnet"):
        return EfficientNetBackbone(cfg)
    else:
        raise ValueError(
            f"Unknown backbone name: '{cfg.name}'. Supported: 'resnext50_32x4d', 'efficientnet_b0'."
        )
