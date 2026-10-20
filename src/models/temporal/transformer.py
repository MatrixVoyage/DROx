"""Multi-Head Temporal Self-Attention Transformer Encoder for frame sequence aggregation."""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn

from src.utils.config import TemporalConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for frame sequence indexing."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to sequence: x shape (B, T, D)."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TemporalTransformer(nn.Module):
    """Temporal Transformer Encoder applying Multi-Head Self-Attention across frames.

    Transforms frame-level context vectors (B, T, D) into a comprehensive
    video-level representation (B, D).

    Unlike unidirectional LSTM, the Transformer computes all pairwise
    temporal relations directly, capturing non-adjacent temporal forgeries
    and facial inconsistencies without vanishing gradients.

    Args:
        config: Validated TemporalConfig instance.
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        super().__init__()
        self.config = config or TemporalConfig(type="transformer")

        d_model = self.config.input_dim
        nhead = getattr(self.config, "nhead", 4)
        num_layers = getattr(self.config, "transformer_layers", 2)
        dim_feedforward = getattr(self.config, "dim_feedforward", d_model * 2)
        dropout = self.config.dropout
        self.pooling = getattr(self.config, "pooling", "mean")

        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

        if self.pooling == "cls_token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal transformer.

        Args:
            x: Input frame sequence features of shape (B, T, D).

        Returns:
            Aggregated video representation of shape (B, D).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor of shape (B, T, D), got {tuple(x.shape)}")

        b, t, d = x.shape
        if d != self.config.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected D={self.config.input_dim}, got {d}"
            )

        if self.cls_token is not None:
            # Prepend class token: (B, 1, D)
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, T + 1, D)

        # Add positional encodings
        x = self.pos_encoder(x)

        # Multi-head self-attention layers
        out = self.transformer_encoder(x)
        out = self.norm(out)

        # Aggregate across sequence dimension
        if self.cls_token is not None:
            video_repr = out[:, 0]  # (B, D)
        elif self.pooling == "mean":
            video_repr = out.mean(dim=1)  # (B, D)
        elif self.pooling == "last":
            video_repr = out[:, -1]  # (B, D)
        else:
            raise ValueError(f"Unsupported pooling mode: '{self.pooling}'")

        return video_repr
