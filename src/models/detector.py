"""Complete DeepfakeDetector composing Backbone, Attention, Temporal, and Classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from src.models.attention.factory import build_attention_module
from src.models.backbones.factory import build_backbone
from src.models.classifier import ClassifierHead
from src.models.temporal.factory import build_temporal_model
from src.utils.config import ModelConfig


@dataclass
class DetectorOutput:
    """Comprehensive diagnostic container for DeepfakeDetector intermediate representations."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    frame_features: torch.Tensor
    temporal_features: torch.Tensor
    attention_maps: Dict[str, torch.Tensor]


class DeepfakeDetector(nn.Module):
    """Unified Deepfake Video Detection model.

    Pipeline:
        Input Face Sequence (B, T, 3, 112, 112)
        -> Spatial Backbone -> Spatial Feature Maps (B, T, D, 4, 4)
        -> Spatial Attention / Pooling -> Frame Context Vectors (B, T, D)
        -> Temporal Aggregator -> Temporal Representation (B, D)
        -> Classifier Head MLP -> Raw Scalar Logits (B, 1)

    Operates in standard forward mode returning raw scalar logits (B, 1),
    embedding mode returning (logits, temporal_features),
    or diagnostic mode returning DetectorOutput with all intermediate feature
    representations, attention maps, and diagnostic probabilities.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.backbone = build_backbone(self.config.backbone)
        self.attention = build_attention_module(self.config.attention)
        self.temporal = build_temporal_model(self.config.temporal)
        self.classifier = ClassifierHead(self.config.classifier)

        # Construction-time verification of cumulative backbone freezing for resnext
        if getattr(self.config.backbone, "name", "resnext50_32x4d") == "resnext50_32x4d" and self.config.backbone.freeze_strategy == "layer2":
            for stage in [self.backbone.stem, self.backbone.layer1, self.backbone.layer2]:
                for p in stage.parameters():
                    assert not p.requires_grad, "Expected frozen stage parameter to have requires_grad=False"

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], DetectorOutput]:
        """Forward pass through complete detection model.

        Args:
            x: Raw input sequence of face crops of shape (B, T, 3, H, W).
            return_intermediates: If True, returns DetectorOutput container.
            return_embeddings: If True, returns tuple of (logits, temporal_features).

        Returns:
            Raw scalar logits (B, 1) by default, (logits, temporal_features) if return_embeddings=True,
            or DetectorOutput if return_intermediates=True.
        """
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D sequence tensor of shape (B, T, 3, H, W), got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )

        b, t, c, h, w = x.shape
        if b <= 0 or t <= 0:
            raise ValueError(f"Batch size B and sequence length T must be > 0, got B={b}, T={t}")
        expected_h, expected_w = self.config.backbone.input_size
        if (c, h, w) != (3, expected_h, expected_w):
            raise ValueError(f"Expected frame shape (3, {expected_h}, {expected_w}), got ({c}, {h}, {w})")

        # 1. Spatial feature extraction: (B, T, D, h_feat, w_feat)
        _, spatial_maps = self.backbone(x, return_spatial_map=True)

        # 2. Attention weighting: (B, T, D) and attention maps
        frame_features, attn_maps = self.attention(spatial_maps, return_attention_maps=True)

        # 3. Temporal modeling: (B, D)
        temporal_features = self.temporal(frame_features)

        # 4. Binary classification: (B, 1)
        logits = self.classifier(temporal_features)

        if return_embeddings:
            return logits, temporal_features

        if return_intermediates:
            probabilities = torch.sigmoid(logits)
            return DetectorOutput(
                logits=logits,
                probabilities=probabilities,
                frame_features=frame_features,
                temporal_features=temporal_features,
                attention_maps=attn_maps,
            )

        return logits
