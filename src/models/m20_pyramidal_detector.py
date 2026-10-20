"""Milestone 20.2: Whole-Face Multi-Scale Pyramidal Dual-Stream Detector.

Integrates:
1. Pyramidal Detail Fusion Stem: Blends standard whole-face representation (G1) with
   native-resolution Laplacian high-frequency octave (L0) via learnable alpha gating.
2. Stream 1 (Frozen Trunk): M17 Track A Forensic Sensor/Residual Trunk (Scale 1 + Scale 2 active).
3. Stream 2 (Kinematic Stream): Learned Differential Spatiotemporal 3D Branch with Motion Attention.
4. Conditioned Reenactment Gated Fusion: Dynamic adaptive alpha weighting between forensic and kinematic streams.
5. Auxiliary 5-Class Manipulation Family Head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.m17_detector import M17TrackADetector
from src.models.branches.kinematic_reenactment import (
    KinematicReenactmentBranch,
    ReenactmentGatedFusion,
)


@dataclass
class M20DetectorOutput:
    """Output container for Milestone 20.2 Pyramidal Dual-Stream Detector."""

    primary_logits: torch.Tensor       # (B, 1) binary classification logits
    kin_logits: torch.Tensor           # (B, 1) standalone kinematic branch logits
    aux_logits: torch.Tensor           # (B, 5) 5-class manipulation family logits
    gate_values: torch.Tensor          # (B, 1) dynamic fusion gate alpha in [0, 1]
    p_fused: torch.Tensor              # Sigmoid probability from primary logit
    p_kin: torch.Tensor                # Sigmoid probability from kinematic head
    p_aux_fake: torch.Tensor           # 1 - P(Real) from auxiliary family head
    pyramidal_alpha: torch.Tensor      # Scalar learnable detail injection weight in [0, 1]
    trunk_features: torch.Tensor       # (B, 1280) features from forensic trunk
    kin_features: torch.Tensor         # (B, 256) features from kinematic branch
    fused_features: torch.Tensor       # (B, 1280) final fused representation


class PyramidalFusionStem(nn.Module):
    """Learnable detail fusion stem blending base face crops with native high-frequency residuals."""

    def __init__(self, init_alpha: float = 0.50) -> None:
        super().__init__()
        # Initialize raw parameter so sigmoid(raw_alpha) == init_alpha
        raw_val = float(torch.logit(torch.tensor(init_alpha)).item())
        self.raw_alpha = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        """Effective detail injection weight in [0, 1]."""
        return torch.sigmoid(self.raw_alpha)

    def forward(
        self,
        x_base: torch.Tensor,
        x_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blend base downsampled whole face with native high-frequency residual.

        Args:
            x_base: Base facial tensor of shape (B, T, 3, 112, 112).
            x_high: Optional native high-frequency residual tensor of shape (B, T, 3, 112, 112).

        Returns:
            Blended tensor of shape (B, T, 3, 112, 112).
        """
        if x_high is None:
            return x_base
        return x_base + self.alpha * x_high


class M20PyramidalDualStreamDetector(nn.Module):
    """Milestone 20.2: Whole-Face Multi-Scale Pyramidal Dual-Stream Detector."""

    def __init__(
        self,
        trunk_detector: M17TrackADetector,
        kin_dim: int = 256,
        aux_hidden_dim: int = 512,
        aux_dropout: float = 0.5,
        gate_bottleneck: int = 128,
        init_pyramid_alpha: float = 0.50,
        freeze_trunk: bool = True,
    ) -> None:
        super().__init__()
        self.trunk = trunk_detector
        self.trunk_dim = getattr(
            trunk_detector, "temporal_dim", getattr(trunk_detector, "trunk_dim", 1280)
        )
        self.kin_dim = kin_dim

        # 1. Pyramidal Detail Fusion Stem
        self.pyramidal_stem = PyramidalFusionStem(init_alpha=init_pyramid_alpha)

        # 2. Freeze trunk parameters if requested
        if freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = False
            self.trunk.eval()

        # 3. Kinematic Differential Branch (Stream 2)
        self.kinematic_branch = KinematicReenactmentBranch(
            in_channels=6,
            out_dim=kin_dim,
            dropout=0.3,
        )

        # 4. Standalone Kinematic Head
        self.kin_head = nn.Sequential(
            nn.Linear(kin_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # 5. Conditioned Reenactment Gated Fusion
        self.fusion = ReenactmentGatedFusion(
            trunk_dim=self.trunk_dim,
            kin_dim=kin_dim,
            gate_bottleneck=gate_bottleneck,
            dropout=0.3,
        )

        # 6. Primary Binary Classification Head
        self.classifier = nn.Linear(self.trunk_dim, 1)

        # 7. Auxiliary Multi-Family Classification Head
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
        x_high: Optional[torch.Tensor] = None,
        mode: str = "dual_stream",
    ) -> M20DetectorOutput:
        """Forward pass through Pyramidal Dual-Stream Detector.

        Args:
            x: Input base video tensor (B, T, 3, 112, 112).
            x_high: Optional native high-frequency residual tensor (B, T, 3, 112, 112).
            mode: One of 'dual_stream' (default), 'kinematic_only', or 'residual_only'.

        Returns:
            M20DetectorOutput with predictions, gate activations, and pyramidal alpha.
        """
        B = x.shape[0]

        # 1. Blend native high-frequency detail into whole-face stream
        x_pyr = self.pyramidal_stem(x, x_high)

        # 2. Extract kinematic representations from Stream 2
        z_kin = self.kinematic_branch(x_pyr)  # (B, kin_dim=256)
        kin_logits = self.kin_head(z_kin)     # (B, 1)

        # 3. Extract trunk representations from Stream 1 (Scale 1 + Scale 2 active)
        if mode == "kinematic_only":
            z_trunk = torch.zeros(B, self.trunk_dim, device=x.device, dtype=z_kin.dtype)
            z_fused = self.fusion.kin_proj(z_kin)
            alpha = torch.ones(B, 1, device=x.device, dtype=z_kin.dtype)
        else:
            with torch.no_grad():
                trunk_out = self.trunk(x_pyr, active_scales=(True, True, False))
                z_trunk = trunk_out.fused_features.detach()  # (B, 1280)

            if mode == "residual_only":
                z_fused = z_trunk
                alpha = torch.zeros(B, 1, device=x.device, dtype=z_kin.dtype)
            else:
                z_fused, alpha = self.fusion(z_trunk, z_kin)

        # 4. Primary binary classification
        primary_logits = self.classifier(z_fused)  # (B, 1)

        # 5. Auxiliary multi-family manipulation classification
        aux_logits = self.aux_family_head(z_fused)  # (B, 5)

        # 6. Probabilities
        p_fused = torch.sigmoid(primary_logits)
        p_kin = torch.sigmoid(kin_logits)
        aux_softmax = torch.softmax(aux_logits, dim=-1)
        p_aux_fake = 1.0 - aux_softmax[:, 0:1]

        return M20DetectorOutput(
            primary_logits=primary_logits,
            kin_logits=kin_logits,
            aux_logits=aux_logits,
            gate_values=alpha,
            p_fused=p_fused,
            p_kin=p_kin,
            p_aux_fake=p_aux_fake,
            pyramidal_alpha=self.pyramidal_stem.alpha,
            trunk_features=z_trunk,
            kin_features=z_kin,
            fused_features=z_fused,
        )
