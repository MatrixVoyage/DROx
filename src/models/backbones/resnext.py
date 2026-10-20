"""ResNeXt-50 (32x4d) spatial feature extraction backbone."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d

from src.utils.config import BackboneConfig


class ResNeXt50Backbone(nn.Module):
    """ResNeXt-50 (32x4d) spatial backbone for face feature extraction.

    Enforces the (3, 112, 112) -> (2048, 4, 4) spatial contract, supports cumulative
    freezing strategies, decoupled BatchNorm running-statistics freezing, and
    provides provenance metadata.
    """

    def __init__(self, config: Optional[BackboneConfig] = None):
        super().__init__()
        self.config = config or BackboneConfig()

        # Instantiate base torchvision model
        if self.config.weights == "IMAGENET1K_V1":
            base_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        elif self.config.weights is None:
            base_model = resnext50_32x4d(weights=None)
        else:
            raise ValueError(
                f"Unsupported weights '{self.config.weights}'. Must be 'IMAGENET1K_V1' or None."
            )

        # Decompose into named stages
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
        )
        self.layer1 = base_model.layer1  # Residual stage 1 (256 channels)
        self.layer2 = base_model.layer2  # Residual stage 2 (512 channels)
        self.layer3 = base_model.layer3  # Residual stage 3 (1024 channels)
        self.layer4 = base_model.layer4  # Residual stage 4 (2048 channels)
        self.avgpool = base_model.avgpool  # AdaptiveAvgPool2d((1, 1))

        # Apply cumulative parameter freezing
        self.frozen_stages: List[nn.Module] = self._apply_freezing()

    def _apply_freezing(self) -> List[nn.Module]:
        """Apply cumulative parameter freezing based on freeze_strategy."""
        strat = self.config.freeze_strategy
        if strat == "none":
            frozen: List[nn.Module] = []
        elif strat == "stem":
            frozen = [self.stem]
        elif strat == "layer1":
            frozen = [self.stem, self.layer1]
        elif strat == "layer2":
            frozen = [self.stem, self.layer1, self.layer2]
        elif strat == "layer3":
            frozen = [self.stem, self.layer1, self.layer2, self.layer3]
        elif strat == "all":
            frozen = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        else:
            raise ValueError(f"Unknown freeze_strategy: '{strat}'")

        # Unfreeze all parameters first
        for param in self.parameters():
            param.requires_grad = True

        # Freeze selected stages
        for stage in frozen:
            for param in stage.parameters():
                param.requires_grad = False

        return frozen

    def train(self, mode: bool = True) -> ResNeXt50Backbone:
        """Set training mode, keeping frozen BatchNorm modules in eval() mode if freeze_batchnorm=True."""
        super().train(mode)
        if mode and self.config.freeze_batchnorm:
            for stage in self.frozen_stages:
                for m in stage.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        return_spatial_map: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through ResNeXt-50 spatial backbone.

        Args:
            x: Tensor of shape (B, 3, 112, 112) or (B, T, 3, 112, 112).
            return_spatial_map: If True, returns (pooled_features, spatial_features).

        Returns:
            pooled_features of shape (B, 2048) or (B, T, 2048).
            If return_spatial_map=True, returns tuple of (pooled, spatial) where
            spatial is (B, 2048, 4, 4) or (B, T, 2048, 4, 4).
        """
        is_5d = x.dim() == 5
        if is_5d:
            b, t, c, h, w = x.shape
            if (c, h, w) != (3, self.config.input_size[0], self.config.input_size[1]):
                raise ValueError(
                    f"Expected frame shape (3, {self.config.input_size[0]}, {self.config.input_size[1]}), "
                    f"got ({c}, {h}, {w})"
                )
            x_flat = x.view(b * t, c, h, w)
        elif x.dim() == 4:
            n, c, h, w = x.shape
            if (c, h, w) != (3, self.config.input_size[0], self.config.input_size[1]):
                raise ValueError(
                    f"Expected frame shape (3, {self.config.input_size[0]}, {self.config.input_size[1]}), "
                    f"got ({c}, {h}, {w})"
                )
            x_flat = x
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got {x.dim()}D with shape {x.shape}")

        # Sequential stage pass
        out = self.stem(x_flat)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        spatial = self.layer4(out)

        # Required 4x4 spatial shape contract check
        expected_spatial = (self.config.spatial_dim, self.config.spatial_dim)
        if spatial.shape[2:] != expected_spatial:
            raise RuntimeError(
                f"ResNeXt-50 spatial shape contract violated: "
                f"expected ({self.config.spatial_dim}, {self.config.spatial_dim}), "
                f"input_size={self.config.input_size}, "
                f"got {tuple(spatial.shape[2:])}"
            )

        pooled = self.avgpool(spatial).flatten(1)

        # Unflatten if input was 5D
        if is_5d:
            pooled = pooled.view(b, t, self.config.output_dim)
            spatial = spatial.view(b, t, self.config.output_dim, self.config.spatial_dim, self.config.spatial_dim)

        if return_spatial_map:
            return pooled, spatial
        return pooled

    def get_provenance(self) -> Dict[str, Any]:
        """Return provenance metadata dictionary for reproducibility tracking."""
        return {
            "backbone_name": self.config.name,
            "weights_name": self.config.weights,
            "torchvision_version": torchvision.__version__,
            "pretrained": self.config.pretrained,
            "freeze_strategy": self.config.freeze_strategy,
            "freeze_batchnorm": self.config.freeze_batchnorm,
            "input_size": tuple(self.config.input_size),
            "output_dim": self.config.output_dim,
            "spatial_dim": self.config.spatial_dim,
        }
