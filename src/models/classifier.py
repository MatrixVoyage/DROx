"""Binary Classifier Head for Deepfake Detection."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from src.utils.config import ClassifierConfig


class ClassifierHead(nn.Module):
    """Two-layer MLP classification head outputting raw scalar logits.

    Architecture:
        Linear(in_features=2048 -> hidden_dim=512)
        -> ReLU()
        -> Dropout(p=0.5)
        -> Linear(hidden_dim=512 -> num_classes=1)

    Strictly outputs unnormalized logits of shape (B, 1) for direct
    compatibility with nn.BCEWithLogitsLoss().
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        super().__init__()
        self.config = config or ClassifierConfig()

        self.net = nn.Sequential(
            nn.Linear(self.config.in_features, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            x: Temporal feature representation of shape (B, 2048).

        Returns:
            Raw scalar logits of shape (B, 1).
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor of shape (B, D), got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )

        b, d = x.shape
        if b <= 0:
            raise ValueError(f"Batch size must be > 0, got B={b}")
        if d != self.config.in_features:
            raise ValueError(
                f"Expected input feature dimension D={self.config.in_features}, got D={d}"
            )

        return self.net(x)
