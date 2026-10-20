"""Multi-Head Deepfake Detector for Milestone 13.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.detector import DeepfakeDetector
from src.models.heads.auxiliary import AuxiliaryManipulationHead, MidLevelSpatialHead
from src.utils.config import ModelConfig


@dataclass
class MultiHeadOutput:
    """Diagnostic container for multi-head detection outputs."""

    primary_logits: torch.Tensor          # (B, 1) binary classification logits
    aux_family_logits: torch.Tensor       # (B, K) multi-family manipulation logits
    mid_logits: Optional[torch.Tensor]    # Optional (B, 1) or (B, K) from mid-level Stage 5 head
    temporal_features: torch.Tensor       # (B, 1280) penultimate temporal representation
    mid_features: Optional[torch.Tensor]  # Optional intermediate spatial features
    p_primary: torch.Tensor               # Sigmoid probability from primary head
    p_aux_fake: torch.Tensor              # 1 - P(Real) from auxiliary family head
    p_fused: torch.Tensor                 # Fused probability estimate


class MultiHeadDeepfakeDetector(nn.Module):
    """Deepfake detector equipped with an auxiliary manipulation-family head.

    Architectural Invariants:
        - Base DeepfakeDetector trunk remains clean and modular.
        - Primary head outputs raw scalar logit (B, 1).
        - Auxiliary head outputs raw family logits (B, num_families).
        - Optional mid-level head connects to Stage 5 intermediate features (for M13.2b).
    """

    def __init__(
        self,
        base_detector: Optional[DeepfakeDetector] = None,
        config: Optional[ModelConfig] = None,
        num_families: int = 5,
        aux_hidden_dim: int = 512,
        aux_dropout: float = 0.5,
        enable_mid_head: bool = False,
        mid_in_channels: int = 112,
        mid_hidden_dim: int = 128,
    ):
        super().__init__()
        self.base_detector = base_detector or DeepfakeDetector(config=config)
        
        penultimate_dim = getattr(self.base_detector.config.temporal, "hidden_dim", 1280)
        self.aux_head = AuxiliaryManipulationHead(
            in_features=penultimate_dim,
            hidden_dim=aux_hidden_dim,
            num_families=num_families,
            dropout=aux_dropout,
        )

        self.enable_mid_head = enable_mid_head
        if enable_mid_head:
            self.mid_head: Optional[MidLevelSpatialHead] = MidLevelSpatialHead(
                in_channels=mid_in_channels,
                hidden_dim=mid_hidden_dim,
                num_classes=1,
                dropout=aux_dropout,
            )
            self._mid_features: Optional[torch.Tensor] = None
            self._register_mid_hook()
        else:
            self.mid_head = None
            self._mid_features = None

    def _register_mid_hook(self) -> None:
        """Register forward hook on Stage 5 of the spatial backbone."""
        def mid_hook(module, input, output):
            self._mid_features = output

        # EfficientNet-B0 Stage 5 is at features[5]
        if hasattr(self.base_detector.backbone, "features") and len(self.base_detector.backbone.features) > 5:
            self.base_detector.backbone.features[5].register_forward_hook(mid_hook)

    def freeze_trunk(self) -> None:
        """Freeze all parameters in the base detector (backbone, attention, temporal, primary classifier)."""
        for p in self.base_detector.parameters():
            p.requires_grad = False

    def unfreeze_temporal_and_heads(self) -> None:
        """Allow fine-tuning of the temporal Transformer and classification heads only."""
        for p in self.base_detector.backbone.parameters():
            p.requires_grad = False
        for p in self.base_detector.attention.parameters():
            p.requires_grad = False
        for p in self.base_detector.temporal.parameters():
            p.requires_grad = True
        for p in self.base_detector.classifier.parameters():
            p.requires_grad = True
        for p in self.aux_head.parameters():
            p.requires_grad = True
        if self.mid_head is not None:
            for p in self.mid_head.parameters():
                p.requires_grad = True

    def forward_from_features(
        self,
        temporal_features: torch.Tensor,
        mid_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast forward pass directly from pre-extracted features.

        Args:
            temporal_features: (B, 1280) penultimate feature representation.
            mid_features: Optional (B, 80) Stage 5 features.

        Returns:
            Tuple of (primary_logits, aux_family_logits).
        """
        primary_logits = self.base_detector.classifier(temporal_features)
        aux_logits = self.aux_head(temporal_features)
        return primary_logits, aux_logits

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
        return_embeddings: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], MultiHeadOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through complete multi-head detector.

        Args:
            x: Input video tensor of shape (B, T, 3, H, W).
            return_dict: If True, returns structured MultiHeadOutput container.
            return_embeddings: If True and not return_dict, returns (primary_logits, temporal_features).

        Returns:
            (primary_logits, aux_family_logits), MultiHeadOutput, or (primary_logits, temporal_features).
        """
        self._mid_features = None

        # 1. Forward through base detector to extract logits & temporal embeddings
        primary_logits, temporal_features = self.base_detector(x, return_embeddings=True)

        # 2. Forward through auxiliary manipulation-family head
        aux_logits = self.aux_head(temporal_features)

        # 3. Optional mid-level Stage 5 forward
        mid_logits = None
        if self.enable_mid_head and self.mid_head is not None and self._mid_features is not None:
            mid_logits = self.mid_head(self._mid_features, batch_size=x.shape[0])

        if return_dict:
            p_primary = torch.sigmoid(primary_logits)
            p_aux_fake = self.aux_head.compute_fake_probability(aux_logits)
            # Fused probability: geometric or arithmetic mean
            p_fused = 0.5 * p_primary + 0.5 * p_aux_fake
            return MultiHeadOutput(
                primary_logits=primary_logits,
                aux_family_logits=aux_logits,
                mid_logits=mid_logits,
                temporal_features=temporal_features,
                mid_features=self._mid_features,
                p_primary=p_primary,
                p_aux_fake=p_aux_fake,
                p_fused=p_fused,
            )

        if return_embeddings:
            return primary_logits, temporal_features

        return primary_logits, aux_logits
