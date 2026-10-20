"""Disentangled Deepfake Detector for Milestone 13.2c."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detector import DeepfakeDetector
from src.models.heads.auxiliary import MidLevelSpatialHead
from src.models.heads.disentangled import SubspaceProjectionHead
from src.utils.config import ModelConfig


@dataclass
class DisentangledOutput:
    """Structured diagnostic container for disentangled detector outputs."""

    primary_logits: torch.Tensor          # (B, 1) from manipulation subspace
    aux_family_logits: torch.Tensor       # (B, K) from manipulation subspace
    mid_logits: Optional[torch.Tensor]    # (B, 1) from Stage 5 spatial head
    z_id: torch.Tensor                    # (B, d) normalized identity embedding
    z_manip: torch.Tensor                 # (B, d) normalized manipulation embedding
    temporal_features: torch.Tensor       # (B, 1280) trunk penultimate features
    mid_features: Optional[torch.Tensor]  # Stage 5 spatial tensor
    p_primary: torch.Tensor               # Sigmoid probability from primary head
    p_aux_fake: torch.Tensor              # 1 - P(Real) from auxiliary family head
    p_mid: Optional[torch.Tensor]         # Sigmoid probability from Stage 5 head
    p_fused: torch.Tensor                 # Tri-head fused probability


class DisentangledDeepfakeDetector(nn.Module):
    """Deepfake detector with identity/manipulation representation disentanglement.

    Architectural Invariants:
        1. Penultimate temporal features (D=1280) are projected into two orthogonal subspaces:
           - z_id (D=128): captures facial identity and dyad lineage.
           - z_manip (D=128): captures manipulation artifacts and reenactment dynamics.
        2. Downstream classification heads (primary and auxiliary) operate STRICTLY on z_manip.
           They have zero access to z_id, mathematically preventing identity shortcuts.
        3. Stage-5 spatial head operates directly on intermediate spatial maps before temporal pooling.
    """

    def __init__(
        self,
        base_detector: Optional[DeepfakeDetector] = None,
        config: Optional[ModelConfig] = None,
        num_families: int = 5,
        subspace_dim: int = 128,
        projection_hidden_dim: int = 512,
        projection_dropout: float = 0.2,
        enable_mid_head: bool = True,
        mid_in_channels: int = 112,
        mid_hidden_dim: int = 128,
    ):
        super().__init__()
        self.base_detector = base_detector or DeepfakeDetector(config=config)
        penultimate_dim = getattr(self.base_detector.config.temporal, "hidden_dim", 1280)
        self.subspace_dim = subspace_dim
        self.num_families = num_families

        # 1. Dual Subspace Projection Heads
        self.id_head = SubspaceProjectionHead(
            in_features=penultimate_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=subspace_dim,
            dropout=projection_dropout,
        )
        self.manip_head = SubspaceProjectionHead(
            in_features=penultimate_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=subspace_dim,
            dropout=projection_dropout,
        )

        # 2. Downstream Heads Operating STRICTLY on z_manip
        self.primary_classifier = nn.Linear(subspace_dim, 1)
        self.aux_classifier = nn.Linear(subspace_dim, num_families)

        # 3. Stage-5 Spatial Head
        self.enable_mid_head = enable_mid_head
        if enable_mid_head:
            is_resnext = hasattr(self.base_detector.backbone, "layer3")
            effective_mid_channels = 1024 if is_resnext else mid_in_channels
            self.mid_head: Optional[MidLevelSpatialHead] = MidLevelSpatialHead(
                in_channels=effective_mid_channels,
                hidden_dim=mid_hidden_dim,
                num_classes=1,
                dropout=projection_dropout,
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

        if hasattr(self.base_detector.backbone, "features") and len(self.base_detector.backbone.features) > 5:
            self.base_detector.backbone.features[5].register_forward_hook(mid_hook)
        elif hasattr(self.base_detector.backbone, "layer3"):
            self.base_detector.backbone.layer3.register_forward_hook(mid_hook)

    def freeze_trunk(self) -> None:
        """Freeze all parameters in the base detector trunk."""
        for p in self.base_detector.parameters():
            p.requires_grad = False

    def unfreeze_temporal_and_heads(self) -> None:
        """Allow fine-tuning of the temporal Transformer, projection heads, and task classifiers."""
        for p in self.base_detector.backbone.parameters():
            p.requires_grad = False
        for p in self.base_detector.attention.parameters():
            p.requires_grad = False
        for p in self.base_detector.temporal.parameters():
            p.requires_grad = True
        # Base classifier is bypassed; disable its gradients
        for p in self.base_detector.classifier.parameters():
            p.requires_grad = False

        for p in self.id_head.parameters():
            p.requires_grad = True
        for p in self.manip_head.parameters():
            p.requires_grad = True
        for p in self.primary_classifier.parameters():
            p.requires_grad = True
        for p in self.aux_classifier.parameters():
            p.requires_grad = True
        if self.mid_head is not None:
            for p in self.mid_head.parameters():
                p.requires_grad = True

    def compute_fake_probability(self, aux_logits: torch.Tensor) -> torch.Tensor:
        """Compute composite probability of being any fake family (1 - P(Real))."""
        probs = F.softmax(aux_logits, dim=-1)
        p_real = probs[:, 0:1]
        return 1.0 - p_real

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], DisentangledOutput]:
        """Forward pass through disentangled detector.

        Args:
            x: Input video tensor of shape (B, T, 3, H, W).
            return_dict: If True, returns DisentangledOutput container.

        Returns:
            (primary_logits, aux_family_logits, z_id, z_manip) or DisentangledOutput.
        """
        self._mid_features = None

        # 1. Forward through trunk to extract temporal embeddings
        _, temporal_features = self.base_detector(x, return_embeddings=True)

        # 2. Project into dual normalized subspaces
        z_id = self.id_head(temporal_features, normalize=True)
        z_manip = self.manip_head(temporal_features, normalize=True)

        # 3. Classify strictly from manipulation subspace
        primary_logits = self.primary_classifier(z_manip)
        aux_logits = self.aux_classifier(z_manip)

        # 4. Stage-5 spatial forward
        mid_logits = None
        if self.enable_mid_head and self.mid_head is not None and self._mid_features is not None:
            mid_logits = self.mid_head(self._mid_features, batch_size=x.shape[0])

        if return_dict:
            p_primary = torch.sigmoid(primary_logits)
            p_aux_fake = self.compute_fake_probability(aux_logits)
            p_mid = torch.sigmoid(mid_logits) if mid_logits is not None else None

            if p_mid is not None:
                p_fused = 0.4 * p_primary + 0.4 * p_aux_fake + 0.2 * p_mid
            else:
                p_fused = 0.5 * p_primary + 0.5 * p_aux_fake

            return DisentangledOutput(
                primary_logits=primary_logits,
                aux_family_logits=aux_logits,
                mid_logits=mid_logits,
                z_id=z_id,
                z_manip=z_manip,
                temporal_features=temporal_features,
                mid_features=self._mid_features,
                p_primary=p_primary,
                p_aux_fake=p_aux_fake,
                p_mid=p_mid,
                p_fused=p_fused,
            )

        return primary_logits, aux_logits, z_id, z_manip
