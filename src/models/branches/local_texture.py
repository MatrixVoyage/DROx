"""Local Artifact & Region-Specific Texture Branch for Milestone 13.3.

Directly targets expression reenactment and boundary artifacts by processing:
1. Interior mouth/lip articulation region.
2. Periorbital eye/eyelid synthesis region.
3. Perimeter boundary blending margin.

Computes texture-to-geometry consistency representations to decouple facial
movement from graphics-based expression warping.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTextureROIBranch(nn.Module):
    """Local Artifact & Region-Specific Texture Branch.

    Extracts multi-region semantic ROIs (mouth, eyes, perimeter boundary) from 112x112
    face crops and learns localized textural representations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        roi_feature_dim: int = 32,
        out_features: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.roi_feature_dim = roi_feature_dim

        # ROI Processor for 32x32 extracted patches (mouth, eyes, boundary)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, roi_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(roi_feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Gradient texture layer to capture high-frequency texture gradient discrepancies
        self.gradient_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 16, roi_feature_dim),
        )

        # Final fusion layer combining 3 ROIs + gradient texture: (roi_feature_dim * 4) -> out_features
        total_in = roi_feature_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(total_in, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def extract_rois(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract mouth, eyes, and perimeter boundary regions from 112x112 face crops.

        Args:
            x: Tensor of shape (N, C, 112, 112).

        Returns:
            Tuple of resized patches (each N, C, 32, 32) for: (mouth, eyes, boundary).
        """
        N, C, H, W = x.shape

        # Mouth ROI: bottom-center (approx 55% to 90% H, 25% to 75% W)
        mouth_crop = x[:, :, int(H * 0.55) : int(H * 0.90), int(W * 0.25) : int(W * 0.75)]
        mouth_patch = F.interpolate(mouth_crop, size=(32, 32), mode="bilinear", align_corners=False)

        # Eye ROI: upper-center (approx 20% to 50% H, 15% to 85% W)
        eye_crop = x[:, :, int(H * 0.20) : int(H * 0.50), int(W * 0.15) : int(W * 0.85)]
        eye_patch = F.interpolate(eye_crop, size=(32, 32), mode="bilinear", align_corners=False)

        # Boundary ROI: outer perimeter rim (masked border)
        mask = torch.ones((1, 1, H, W), device=x.device, dtype=x.dtype)
        mask[:, :, int(H * 0.15) : int(H * 0.85), int(W * 0.15) : int(W * 0.85)] = 0.0
        boundary_crop = x * mask
        boundary_patch = F.interpolate(boundary_crop, size=(32, 32), mode="bilinear", align_corners=False)

        return mouth_patch, eye_patch, boundary_patch

    def compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel-like spatial gradient magnitude for single-channel luminance."""
        # Convert to luminance
        if x.size(1) == 3:
            lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            lum = x[:, 0:1]

        # Horizontal and vertical finite difference kernels
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device).view(1, 1, 3, 3)

        gx = F.conv2d(lum, kx, padding=1)
        gy = F.conv2d(lum, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Video tensor of shape (B, T, C, H, W) or (B*T, C, H, W).

        Returns:
            z_texture: Video-level local texture & boundary representation of shape (B, out_features).
        """
        is_5d = x.dim() == 5
        if is_5d:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
        else:
            x_flat = x
            B = x_flat.size(0)
            T = 1

        # 1. Extract multi-region semantic ROIs: each (B*T, C, 32, 32)
        mouth_patch, eye_patch, boundary_patch = self.extract_rois(x_flat)

        # 2. Encode patches: each (B*T, roi_feature_dim)
        z_mouth = self.patch_encoder(mouth_patch)
        z_eye = self.patch_encoder(eye_patch)
        z_boundary = self.patch_encoder(boundary_patch)

        # 3. Compute texture gradient representation: (B*T, roi_feature_dim)
        grad_mag = self.compute_gradient_magnitude(x_flat)
        z_grad = self.gradient_conv(grad_mag)

        # 4. Concatenate and fuse: (B*T, roi_feature_dim * 4) -> (B*T, out_features)
        combined_bt = torch.cat([z_mouth, z_eye, z_boundary, z_grad], dim=1)

        if is_5d:
            combined = combined_bt.view(B, T, -1).mean(dim=1)  # Temporal mean pool: (B, total_in)
        else:
            combined = combined_bt

        z_texture = self.fusion(combined)  # (B, out_features)
        return z_texture
