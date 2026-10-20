"""Temporal model factory."""

from __future__ import annotations

from typing import Optional
import torch.nn as nn

from src.models.temporal.lstm import LSTMTemporalModel
from src.models.temporal.transformer import TemporalTransformer
from src.utils.config import TemporalConfig


def build_temporal_model(config: Optional[TemporalConfig] = None) -> nn.Module:
    """Build temporal sequence aggregator from configuration.

    Args:
        config: TemporalConfig instance, or None for paper-aligned default (lstm).

    Returns:
        Configured nn.Module conforming to the temporal interface:
        forward(x: Tensor (B, T, D)) -> Tensor (B, D).
    """
    cfg = config or TemporalConfig()

    if cfg.type in ("lstm", "bilstm"):
        return LSTMTemporalModel(cfg)
    elif cfg.type == "transformer":
        return TemporalTransformer(cfg)
    else:
        raise ValueError(
            f"Unknown temporal type: '{cfg.type}'. Expected 'lstm', 'bilstm', or 'transformer'."
        )
