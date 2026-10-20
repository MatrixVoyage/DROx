"""Differential Kinematic Reenactment Branch for Milestone 18.

Implements:
1. FrameDifferentialExtractor:
   Computes 1st-order temporal velocity differentials (Delta I_t = I_{t+1} - I_t)
   and 2nd-order temporal acceleration differentials (Delta^2 I_t).
2. SpatialMotionAttention:
   Learned lightweight 3D spatiotemporal spatial attention module that
   dynamically weights non-rigid facial deformation regions without hard-coded
   coordinate mask assumptions.
3. KinematicReenactmentBranch:
   3D spatiotemporal convolutional trunk operating directly on differential motion
   tensors to capture inner/outer facial shear and expression kinematics.
4. ReenactmentGatedFusion:
   Conditioned adaptive gating module that integrates the kinematic representation
   with the frozen M17 Track A trunk representation.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameDifferentialExtractor(nn.Module):
    """Extracts 1st-order velocity and 2nd-order acceleration differentials across frames."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temporal motion differentials.

        Args:
            x: Input video tensor of shape (B, T, C, H, W) where C=3.

        Returns:
            Differential motion tensor of shape (B, 6, T-1, H, W).
        """
        B, T, C, H, W = x.shape
        if T < 3:
            raise ValueError(f"FrameDifferentialExtractor requires T >= 3 frames, got T={T}")

        # 1st-order velocity differential: Delta I_t = I_{t+1} - I_t (shape: B, T-1, C, H, W)
        delta_1 = x[:, 1:] - x[:, :-1]

        # 2nd-order acceleration differential: Delta^2 I_t = Delta I_{t+1} - Delta I_t (shape: B, T-2, C, H, W)
        delta_2 = delta_1[:, 1:] - delta_1[:, :-1]

        # Replicate final acceleration step to align temporal lengths to T-1
        delta_2_padded = torch.cat([delta_2, delta_2[:, -1:]], dim=1)  # (B, T-1, C, H, W)

        # Concatenate along channel dimension: 3 + 3 = 6 channels
        motion = torch.cat([delta_1, delta_2_padded], dim=2)  # (B, T-1, 6, H, W)

        # Permute to 3D Conv format: (B, C_in=6, D=T-1, H, W)
        motion_3d = motion.permute(0, 2, 1, 3, 4).contiguous()
        return motion_3d


class SpatialMotionAttention(nn.Module):
    """Learned 3D spatiotemporal spatial attention module.

    Computes channel average and max pooling along the feature dimension, then
    projects through a 3D convolution to learn dynamic non-rigid deformation regions.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention weighting.

        Args:
            x: Feature tensor of shape (B, C, D, H, W).

        Returns:
            Attention-modulated tensor of shape (B, C, D, H, W).
        """
        # Pool across channel dimension: average and max
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, D, H, W)

        # Spatial attention map
        attn_map = self.sigmoid(self.conv(pooled))  # (B, 1, D, H, W)

        # Residual spatial attention scaling
        return x * attn_map + x


class KinematicReenactmentBranch(nn.Module):
    """Learned differential spatiotemporal kinematic representation branch."""

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.diff_extractor = FrameDifferentialExtractor()

        # Block 1: (B, 6, 15, 112, 112) -> (B, 32, 15, 56, 56)
        self.block1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                32,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
        )

        # Block 2: (B, 32, 15, 56, 56) -> (B, 64, 15, 28, 28)
        self.block2 = nn.Sequential(
            nn.Conv3d(
                32,
                64,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.SiLU(inplace=True),
        )

        # Block 3: (B, 64, 15, 28, 28) -> (B, 128, 8, 14, 14)
        self.block3 = nn.Sequential(
            nn.Conv3d(
                64,
                128,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(128),
            nn.SiLU(inplace=True),
        )

        # Spatial motion self-attention over dynamic deformation regions
        self.spatial_attn = SpatialMotionAttention(kernel_size=3)

        # Global spatiotemporal adaptive pooling
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Projection MLP to output kinematic embedding
        self.projector = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract kinematic features from consecutive video frames.

        Args:
            x: Input tensor of shape (B, T, C, H, W).

        Returns:
            Kinematic embedding of shape (B, out_dim).
        """
        # Extract 6-channel spatiotemporal motion tensor (B, 6, T-1, H, W)
        motion_3d = self.diff_extractor(x)

        # 3D Convolutional hierarchy
        h = self.block1(motion_3d)
        h = self.block2(h)
        h = self.block3(h)

        # Spatial motion attention
        h = self.spatial_attn(h)

        # Spatiotemporal pooling
        pooled = self.pool(h).flatten(1)  # (B, 128)

        # Projection
        z_kin = self.projector(pooled)  # (B, out_dim)
        return z_kin


class ReenactmentGatedFusion(nn.Module):
    """Conditioned adaptive gate integrating kinematic features into residual trunk."""

    def __init__(
        self,
        trunk_dim: int = 1280,
        kin_dim: int = 256,
        gate_bottleneck: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.trunk_dim = trunk_dim
        self.kin_dim = kin_dim

        # Gating bottleneck MLP computing adaptive weight alpha in [0, 1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(trunk_dim + kin_dim, gate_bottleneck),
            nn.LayerNorm(gate_bottleneck),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_bottleneck, 1),
            nn.Sigmoid(),
        )

        # Projection aligning kinematic dimension to trunk dimension
        self.kin_proj = nn.Sequential(
            nn.Linear(kin_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, z_trunk: torch.Tensor, z_kin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gated fusion between trunk and kinematic features.

        Args:
            z_trunk: Tensor of shape (B, trunk_dim) from M17 Track A trunk.
            z_kin: Tensor of shape (B, kin_dim) from KinematicReenactmentBranch.

        Returns:
            Tuple of:
            - z_fused: Fused tensor of shape (B, trunk_dim).
            - alpha: Gate values of shape (B, 1) in [0, 1].
        """
        combined = torch.cat([z_trunk, z_kin], dim=-1)
        alpha = self.gate_mlp(combined)  # (B, 1)

        projected_kin = self.kin_proj(z_kin)  # (B, trunk_dim)
        z_fused = z_trunk + alpha * projected_kin
        return z_fused, alpha
