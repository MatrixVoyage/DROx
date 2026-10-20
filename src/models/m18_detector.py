"""Milestone 18: Targeted Reenactment-Specific Representation Detector.

Implements the Dual-Stream Kinematic-Residual Architecture:
- Stream 1: Frozen M17 Track A Residual Trunk (Scale 1 + Scale 2 active, Scale 3 pruned).
- Stream 2: Learned Differential Kinematic Branch (Frame velocity + acceleration differentials
            with 3D Spatiotemporal Convolution and Spatial Motion Attention).
- Conditioned Reenactment Gating (Dynamic adaptive alpha weighting).
- Dual-Mode Evaluation Support:
  - 'dual_stream': Full conditioned gating.
  - 'kinematic_only': Standalone kinematic evaluation.
  - 'residual_only': Standalone M17 Track A trunk evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.m17_detector import M17TrackADetector
from src.models.branches.kinematic_reenactment import (
    KinematicReenactmentBranch,
    ReenactmentGatedFusion,
)


@dataclass
class M18DetectorOutput:
    """Output container for Milestone 18 Reenactment Detector."""

    primary_logits: torch.Tensor       # (B, 1) binary classification logits
    kin_logits: torch.Tensor           # (B, 1) standalone kinematic branch logits
    aux_logits: torch.Tensor           # (B, 5) 5-class manipulation family logits
    gate_values: torch.Tensor          # (B, 1) dynamic adaptive gate alpha in [0, 1]
    p_fused: torch.Tensor              # Sigmoid probability from primary logit
    p_kin: torch.Tensor                # Sigmoid probability from standalone kinematic head
    p_aux_fake: torch.Tensor           # 1 - P(Real) from auxiliary family head
    trunk_features: torch.Tensor       # (B, 1280) features from frozen M17 Track A trunk
    kin_features: torch.Tensor         # (B, 256) features from differential kinematic branch
    fused_features: torch.Tensor       # (B, 1280) final fused representation


class M18ReenactmentDetector(nn.Module):
    """Milestone 18: Dual-Stream Kinematic-Residual Deepfake Reenactment Detector."""

    def __init__(
        self,
        trunk_detector: M17TrackADetector,
        kin_dim: int = 256,
        aux_hidden_dim: int = 512,
        aux_dropout: float = 0.5,
        gate_bottleneck: int = 128,
        freeze_trunk: bool = True,
    ) -> None:
        super().__init__()
        self.trunk = trunk_detector
        self.trunk_dim = 1280
        self.kin_dim = kin_dim

        # 1. Permanently freeze trunk parameters if requested
        if freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = False
            self.trunk.eval()

        # 2. Raw Differential Kinematic Branch (Stream 2)
        self.kinematic_branch = KinematicReenactmentBranch(
            in_channels=6,
            out_dim=kin_dim,
            dropout=0.3,
        )

        # 3. Standalone Kinematic Classification Head (for isolated kinematic evaluation)
        self.kin_head = nn.Sequential(
            nn.Linear(kin_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # 4. Conditioned Reenactment Gated Fusion
        self.fusion = ReenactmentGatedFusion(
            trunk_dim=self.trunk_dim,
            kin_dim=kin_dim,
            gate_bottleneck=gate_bottleneck,
            dropout=0.3,
        )

        # 5. Primary Classification Head
        self.classifier = nn.Linear(self.trunk_dim, 1)

        # 6. Auxiliary 5-Class Manipulation Family Head
        # Classes: 0: Real, 1: Face2Face, 2: NeuralTextures, 3: FaceSwap, 4: Deepfakes
        self.aux_family_head = nn.Sequential(
            nn.Linear(self.trunk_dim, aux_hidden_dim),
            nn.LayerNorm(aux_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(aux_dropout),
            nn.Linear(aux_hidden_dim, 5),
        )

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "dual_stream",
    ) -> M18DetectorOutput:
        """Forward pass through M18 Reenactment Detector.

        Args:
            x: Input video tensor of shape (B, T, C, H, W).
            mode: One of 'dual_stream' (default), 'kinematic_only', or 'residual_only'.

        Returns:
            M18DetectorOutput with logits, gate values, probabilities, and features.
        """
        B = x.shape[0]

        # 1. Extract kinematic representations from Stream 2
        z_kin = self.kinematic_branch(x)  # (B, kin_dim=256)
        kin_logits = self.kin_head(z_kin)  # (B, 1)

        # 2. Extract trunk representations from Stream 1 (Frozen M17 Track A)
        # Strictly run with Scale 1 + Scale 2 active, Scale 3 pruned as per M17.1
        if mode == "kinematic_only":
            # For kinematic-only ablation: project z_kin directly into classification space
            z_trunk = torch.zeros(B, self.trunk_dim, device=x.device, dtype=z_kin.dtype)
            z_fused = self.fusion.kin_proj(z_kin)
            alpha = torch.ones(B, 1, device=x.device, dtype=z_kin.dtype)
        else:
            with torch.no_grad():
                # Trunk forward pass with Scale 1 and Scale 2 enabled
                trunk_out = self.trunk(x, active_scales=(True, True, False))
                z_trunk = trunk_out.fused_features.detach()  # (B, 1280)

            if mode == "residual_only":
                # For residual-only baseline: alpha = 0
                z_fused = z_trunk
                alpha = torch.zeros(B, 1, device=x.device, dtype=z_kin.dtype)
            else:
                # 'dual_stream': Full adaptive gated fusion
                z_fused, alpha = self.fusion(z_trunk, z_kin)

        # 3. Primary binary classification
        primary_logits = self.classifier(z_fused)  # (B, 1)

        # 4. Auxiliary multi-family manipulation classification
        aux_logits = self.aux_family_head(z_fused)  # (B, 5)

        # 5. Probabilities
        p_fused = torch.sigmoid(primary_logits)
        p_kin = torch.sigmoid(kin_logits)
        aux_softmax = torch.softmax(aux_logits, dim=-1)
        p_aux_fake = 1.0 - aux_softmax[:, 0:1]  # 1 - P(Real)

        return M18DetectorOutput(
            primary_logits=primary_logits,
            kin_logits=kin_logits,
            aux_logits=aux_logits,
            gate_values=alpha,
            p_fused=p_fused,
            p_kin=p_kin,
            p_aux_fake=p_aux_fake,
            trunk_features=z_trunk,
            kin_features=z_kin,
            fused_features=z_fused,
        )
