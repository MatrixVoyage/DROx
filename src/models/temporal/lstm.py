"""LSTM Temporal Modeling Module for sequence feature aggregation."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from src.utils.config import TemporalConfig


class LSTMTemporalModel(nn.Module):
    """LSTM-based temporal aggregator for frame feature sequences.

    Processes sequences of attention-weighted frame vectors of shape (B, T, 2048)
    through an LSTM layer followed by temporal pooling and explicit post-pooling
    dropout (p=0.5), outputting a 2048-D temporal representation.

    Note on Dropout: For a 1-layer LSTM, PyTorch's internal LSTM dropout is
    inapplicable and set to 0.0. The paper-required 0.5 dropout is applied
    explicitly post-pooling on the pooled temporal vector.
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        super().__init__()
        self.config = config or TemporalConfig()

        self.lstm = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=1,
            bidirectional=self.config.bidirectional,
            batch_first=True,
            dropout=0.0,
        )
        self.post_dropout = nn.Dropout(p=self.config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for temporal sequence modeling.

        Args:
            x: Tensor of shape (B, T, 2048).

        Returns:
            Pooled temporal representation of shape (B, 2048).
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor of shape (B, T, C), got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )

        b, t, c = x.shape
        if b <= 0 or t <= 0:
            raise ValueError(f"Batch size B and sequence length T must be > 0, got B={b}, T={t}")
        if c != self.config.input_dim:
            raise ValueError(
                f"Expected input feature dimension C={self.config.input_dim}, got C={c}"
            )

        out, _ = self.lstm(x)

        if self.config.pooling == "last":
            if not self.config.bidirectional:
                pooled = out[:, -1, :]
            else:
                h_fwd = out[:, -1, : self.config.hidden_dim]
                h_bwd = out[:, 0, self.config.hidden_dim :]
                pooled = torch.cat([h_fwd, h_bwd], dim=1)
        elif self.config.pooling == "mean":
            pooled = out.mean(dim=1)
        else:
            raise ValueError(f"Unknown temporal pooling policy: '{self.config.pooling}'")

        return self.post_dropout(pooled)
