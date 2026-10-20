"""Family-Invariant Deepfake Detector for Milestone 13.3.

Combines:
1. Spatial Trunk & Temporal Transformer: Base deepfake detector (EfficientNet-B0 or ResNeXt-50).
2. Learned Local Frequency Residual Branch: 8x8 Block-DCT and Bayar constrained high-pass residuals.
3. Local Artifact & Region-Specific Texture Branch: Mouth, eye, boundary ROIs and gradient texture.
4. Stage-5 Mid-Level Spatial Head: Hook on intermediate spatial representations (from M13.2b).
5. Multi-Domain Invariant Fusion: Fuses spatial-temporal, frequency, and texture embeddings.
6. Multi-Family Auxiliary Head: Supervised manipulation family classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.detector import DeepfakeDetector
from src.models.branches.frequency_residual import FrequencyResidualBranch
from src.models.branches.local_texture import LocalTextureROIBranch
from src.models.heads.auxiliary import AuxiliaryManipulationHead, MidLevelSpatialHead
from src.utils.config import ModelConfig


@dataclass
class FamilyInvariantOutput:
    """Diagnostic and evaluation output container for Family-Invariant Detector."""

    primary_logits: torch.Tensor          # (B, 1) binary classification logits from multi-domain fusion
    aux_family_logits: torch.Tensor       # (B, K) multi-family manipulation logits
    freq_logits: torch.Tensor             # (B, 1) dedicated binary logits from frequency residual branch
    texture_logits: torch.Tensor          # (B, 1) dedicated binary logits from local texture ROI branch
    mid_logits: Optional[torch.Tensor]    # Optional (B, 1) from Stage-5 mid-level spatial head
    temporal_features: torch.Tensor       # (B, D_temp) temporal representation from base detector
    freq_features: torch.Tensor           # (B, D_freq) frequency residual representation
    texture_features: torch.Tensor        # (B, D_tex) local texture ROI representation
    p_fused: torch.Tensor                 # Sigmoid probability from multi-domain primary head
    p_primary: torch.Tensor               # Sigmoid probability from primary head
    p_freq: torch.Tensor                  # Sigmoid probability from frequency branch
    p_texture: torch.Tensor               # Sigmoid probability from texture branch
    p_aux_fake: torch.Tensor              # 1 - P(Real) from auxiliary family head
    p_composite: torch.Tensor             # Balanced ensemble composite probability


class FamilyInvariantDeepfakeDetector(nn.Module):
    """Deepfake detector equipped with Local Frequency, Local Texture, and Auxiliary Heads.

    Architectural Invariants:
        - Base DeepfakeDetector trunk remains clean, modular, and uncoupled.
        - Frequency branch operates on spatially localized 8x8 DCT blocks + Bayar high-pass residuals.
        - Texture branch extracts mouth, eye, and boundary sub-crops + spatial gradient textures.
        - Multi-domain fusion head fuses temporal (1280), frequency (64), and texture (64) features.
        - Auxiliary family head provides multi-class regularizing competition.
        - Mid-level Stage 5 head preserves intermediate spatial artifact signals.
    """

    def __init__(
        self,
        base_detector: Optional[DeepfakeDetector] = None,
        config: Optional[ModelConfig] = None,
        num_families: int = 5,
        aux_hidden_dim: int = 512,
        aux_dropout: float = 0.5,
        freq_dim: int = 64,
        texture_dim: int = 64,
        fusion_dim: int = 512,
        enable_mid_head: bool = True,
        mid_in_channels: int = 112,
        mid_hidden_dim: int = 128,
    ):
        super().__init__()
        self.base_detector = base_detector or DeepfakeDetector(config=config)
        self.temporal_dim = getattr(self.base_detector.config.temporal, "hidden_dim", 1280)
        self.freq_dim = freq_dim
        self.texture_dim = texture_dim
        self.enable_mid_head = enable_mid_head

        # 1. Frequency Residual Branch
        self.freq_branch = FrequencyResidualBranch(
            block_size=8,
            bayar_channels=8,
            hidden_dim=64,
            out_features=freq_dim,
            dropout=0.2,
        )
        self.freq_classifier = nn.Sequential(
            nn.Linear(freq_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 2. Local Texture & ROI Boundary Branch
        self.texture_branch = LocalTextureROIBranch(
            in_channels=3,
            roi_feature_dim=32,
            out_features=texture_dim,
            dropout=0.2,
        )
        self.texture_classifier = nn.Sequential(
            nn.Linear(texture_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 3. Auxiliary Manipulation-Family Head
        self.aux_head = AuxiliaryManipulationHead(
            in_features=self.temporal_dim,
            hidden_dim=aux_hidden_dim,
            num_families=num_families,
            dropout=aux_dropout,
        )

        # 4. Optional Stage-5 Mid-Level Spatial Head
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

        # 5. Multi-Domain Invariant Fusion Head
        total_fused_in = self.temporal_dim + freq_dim + texture_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(total_fused_in, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(aux_dropout),
            nn.Linear(fusion_dim, 1),
        )

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
        """Allow fine-tuning of temporal transformer, all branch heads, and fusion module."""
        # Keep spatial backbone and attention frozen
        for p in self.base_detector.backbone.parameters():
            p.requires_grad = False
        for p in self.base_detector.attention.parameters():
            p.requires_grad = False

        # Unfreeze temporal Transformer & base classifier
        for p in self.base_detector.temporal.parameters():
            p.requires_grad = True
        for p in self.base_detector.classifier.parameters():
            p.requires_grad = True

        # Unfreeze new M13.3 branches & heads
        for p in self.freq_branch.parameters():
            p.requires_grad = True
        for p in self.freq_classifier.parameters():
            p.requires_grad = True
        for p in self.texture_branch.parameters():
            p.requires_grad = True
        for p in self.texture_classifier.parameters():
            p.requires_grad = True
        for p in self.aux_head.parameters():
            p.requires_grad = True
        for p in self.fusion_head.parameters():
            p.requires_grad = True
        if self.mid_head is not None:
            for p in self.mid_head.parameters():
                p.requires_grad = True

    def unfreeze_branches_only(self) -> None:
        """Freeze entire base detector (including temporal); train ONLY the frequency and texture branches."""
        self.freeze_trunk()
        for p in self.freq_branch.parameters():
            p.requires_grad = True
        for p in self.freq_classifier.parameters():
            p.requires_grad = True
        for p in self.texture_branch.parameters():
            p.requires_grad = True
        for p in self.texture_classifier.parameters():
            p.requires_grad = True
        for p in self.aux_head.parameters():
            p.requires_grad = True
        for p in self.fusion_head.parameters():
            p.requires_grad = True
        if self.mid_head is not None:
            for p in self.mid_head.parameters():
                p.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[FamilyInvariantOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """Full forward pass through all branches and heads.

        Args:
            x: Input video tensor of shape (B, T, 3, 112, 112).
            return_dict: If True, returns rich FamilyInvariantOutput dataclass.

        Returns:
            FamilyInvariantOutput containing all head predictions and intermediate representations.
        """
        B, T, C, H, W = x.shape

        # 1. Base detector pass (spatial backbone + attention + temporal transformer)
        base_logits, temporal_features = self.base_detector(x, return_embeddings=True)

        # 2. Local Frequency Residual Branch
        z_freq, _ = self.freq_branch(x)  # (B, freq_dim)
        freq_logits = self.freq_classifier(z_freq)  # (B, 1)

        # 3. Local Artifact & Region-Specific Texture Branch
        z_texture = self.texture_branch(x)  # (B, texture_dim)
        texture_logits = self.texture_classifier(z_texture)  # (B, 1)

        # 4. Multi-Family Auxiliary Head
        aux_family_logits = self.aux_head(temporal_features)  # (B, num_families)

        # 5. Optional Stage-5 Mid-Level Head
        mid_logits = None
        if self.enable_mid_head and self.mid_head is not None and self._mid_features is not None:
            # (B*T, 112, 7, 7) -> average pool over T
            mid_raw = self._mid_features  # (B*T, C_mid, H_mid, W_mid)
            _, C_m, H_m, W_m = mid_raw.shape
            mid_seq = mid_raw.view(B, T, C_m, H_m, W_m).mean(dim=1)  # (B, C_mid, H_mid, W_mid)
            mid_logits = self.mid_head(mid_seq)  # (B, 1)

        # 6. Multi-Domain Invariant Fusion Head
        fused_in = torch.cat([temporal_features, z_freq, z_texture], dim=-1)  # (B, total_fused_in)
        primary_logits = self.fusion_head(fused_in)  # (B, 1)

        # 7. Probability computations
        p_primary = torch.sigmoid(base_logits)
        p_fused = torch.sigmoid(primary_logits)
        p_freq = torch.sigmoid(freq_logits)
        p_texture = torch.sigmoid(texture_logits)

        # Auxiliary Fake Probability: 1 - P(Real) = sum_{k=1}^{K-1} P(c_k)
        aux_probs = torch.softmax(aux_family_logits, dim=-1)
        p_aux_fake = (1.0 - aux_probs[:, 0:1]).clamp(0.0, 1.0)

        # Balanced Composite Probability
        # Combines multi-domain fused head (50%), auxiliary family competition (25%),
        # frequency residual branch (12.5%), and local texture branch (12.5%)
        p_composite = (
            0.50 * p_fused
            + 0.25 * p_aux_fake
            + 0.125 * p_freq
            + 0.125 * p_texture
        ).clamp(0.0, 1.0)

        if not return_dict:
            return primary_logits, aux_family_logits

        return FamilyInvariantOutput(
            primary_logits=primary_logits,
            aux_family_logits=aux_family_logits,
            freq_logits=freq_logits,
            texture_logits=texture_logits,
            mid_logits=mid_logits,
            temporal_features=temporal_features,
            freq_features=z_freq,
            texture_features=z_texture,
            p_fused=p_fused,
            p_primary=p_primary,
            p_freq=p_freq,
            p_texture=p_texture,
            p_aux_fake=p_aux_fake,
            p_composite=p_composite,
        )
