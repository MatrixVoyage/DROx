"""Auxiliary Manipulation Heads for Out-of-Family Generalization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standardized manipulation family index mapping
MANIPULATION_FAMILY_MAPPING: Dict[str, int] = {
    "original": 0,         # Pristine FF++
    "pristine": 0,         # Pristine Celeb-DF
    "Deepfakes": 1,        # Autoencoder face replacement
    "FaceSwap": 2,         # 2D/3D graphics landmark alignment & blending
    "NeuralTextures": 3,   # Neural rendering / facial texture modification
    "deepfake": 4,         # Cross-dataset Celeb-DF deepfake
    "Face2Face": 5,        # 3DMM expression reenactment (strictly held-out test family)
}

FAMILY_NAMES: List[str] = [
    "Real",
    "Deepfakes",
    "FaceSwap",
    "NeuralTextures",
    "CelebDF",
    "Face2Face",
]


class AuxiliaryManipulationHead(nn.Module):
    """Auxiliary classifier predicting manipulation families or fine-grained categories.

    Trained on seen training classes (K=5 by default):
        0: Real (original FF++ & pristine Celeb-DF)
        1: Deepfakes
        2: FaceSwap
        3: NeuralTextures
        4: Celeb-DF Deepfake

    Architecture:
        Linear(in_features -> hidden_dim)
        -> BatchNorm1d(hidden_dim)
        -> ReLU()
        -> Dropout(p=dropout)
        -> Linear(hidden_dim -> num_families)
    """

    def __init__(
        self,
        in_features: int = 1280,
        hidden_dim: int = 512,
        num_families: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_families = num_families
        self.dropout = dropout

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_families)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through auxiliary head.

        Args:
            x: Penultimate feature tensor of shape (B, in_features).

        Returns:
            Raw logits of shape (B, num_families).
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input (B, D), got {x.dim()}D tensor with shape {tuple(x.shape)}")

        b, d = x.shape
        if b <= 0:
            raise ValueError(f"Batch size must be > 0, got B={b}")
        if d != self.in_features:
            raise ValueError(f"Expected input dimension D={self.in_features}, got D={d}")

        h = self.fc1(x)
        # Apply BatchNorm only if batch size > 1 or in eval mode
        if b > 1 or not self.training:
            h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        logits = self.fc2(h)
        return logits

    def compute_fake_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute the composite probability of being any fake family (1 - P(Real)).

        Args:
            logits: Family logits of shape (B, num_families).

        Returns:
            Probability tensor of shape (B, 1) representing 1 - P(class 0).
        """
        probs = F.softmax(logits, dim=-1)
        p_real = probs[:, 0:1]
        p_fake = 1.0 - p_real
        return p_fake


class MidLevelSpatialHead(nn.Module):
    """Auxiliary head operating on Stage 5 intermediate spatial representations.

    Input can be:
        - 5D spatial feature map: (B, T, C=80, H=14, W=14)
        - 3D pooled sequence: (B, T, C=80)
        - 2D pooled vector: (B, C=80)

    Performs spatial adaptive pooling, temporal mean pooling, and MLP classification.
    """

    def __init__(
        self,
        in_channels: int = 112,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """Forward pass from intermediate spatial features.

        Args:
            x: Tensor of shape (B*T, C, H, W), (B, T, C, H, W), (B, T, C), or (B, C).
            batch_size: Optional batch size B when passing flattened (B*T, C, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        if x.dim() == 4:
            # (B * T, C, H, W) from backbone forward hook
            bt, c, h, w = x.shape
            b = batch_size if (batch_size is not None and batch_size > 0) else bt
            t = bt // b if b > 0 else 1
            pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(b, t, c).mean(dim=1)
            feat = pooled
        elif x.dim() == 5:
            # (B, T, C, H, W) -> spatially pool to (B, T, C, 1, 1) -> (B, T, C)
            b, t, c, h, w = x.shape
            x_flat = x.view(b * t, c, h, w)
            pooled = F.adaptive_avg_pool2d(x_flat, (1, 1)).view(b, t, c)
            # Temporally pool to (B, C)
            feat = pooled.mean(dim=1)
        elif x.dim() == 3:
            # (B, T, C) -> temporally pool to (B, C)
            feat = x.mean(dim=1)
        elif x.dim() == 2:
            feat = x
        else:
            raise ValueError(f"Expected 2D, 3D, 4D, or 5D input, got {x.dim()}D with shape {tuple(x.shape)}")

        b, c = feat.shape
        if c != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {c}")

        # Layer 1
        h = self.net[0](feat)
        if b > 1 or not self.training:
            h = self.net[1](h)
        h = self.net[2](h)
        h = self.net[3](h)
        out = self.net[4](h)
        return out
