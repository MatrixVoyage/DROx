"""EfficientNet-B0 spatial feature extraction backbone."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.utils.config import BackboneConfig


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 spatial backbone for deepfake facial feature extraction.

    Transforms (3, 112, 112) faces into (1280, 4, 4) spatial feature representations
    (or (1280, 7, 7) for 224x224 inputs).

    Provides substantial computational efficiency (~5.3M parameters vs ResNeXt-50's ~25M)
    while preserving fine spatial cues across frames.
    """

    def __init__(self, config: Optional[BackboneConfig] = None):
        super().__init__()
        self.config = config or BackboneConfig(name="efficientnet_b0", output_dim=1280, spatial_dim=4)

        weights = None
        if self.config.weights in ("IMAGENET1K_V1", "DEFAULT"):
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
            except Exception:
                weights = None

        base_model = efficientnet_b0(weights=weights)

        # Extract features and pooling
        self.features = base_model.features  # stages 0 to 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Apply cumulative parameter freezing
        self.frozen_stages: List[nn.Module] = self._apply_freezing()

    def _apply_freezing(self) -> List[nn.Module]:
        """Apply parameter freezing based on freeze_strategy."""
        strat = getattr(self.config, "freeze_strategy", "none")
        frozen: List[nn.Module] = []

        if strat == "none":
            frozen = []
        elif strat == "stem":
            frozen = [self.features[0]]
        elif strat in ("early", "layer1"):
            frozen = [self.features[i] for i in range(4)]  # stages 0-3
        elif strat in ("mid", "layer2"):
            frozen = [self.features[i] for i in range(6)]  # stages 0-5
        elif strat in ("late", "layer3"):
            frozen = [self.features[i] for i in range(8)]  # stages 0-7
        elif strat == "all":
            frozen = [self.features]
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

    def train(self, mode: bool = True) -> EfficientNetBackbone:
        """Set training mode, keeping frozen BatchNorm modules in eval() mode if freeze_batchnorm=True."""
        super().train(mode)
        if mode and getattr(self.config, "freeze_batchnorm", True):
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
        """Forward pass through EfficientNet backbone.

        Args:
            x: Tensor of shape (B, 3, H, W) or (B, T, 3, H, W).
            return_spatial_map: If True, returns (pooled_features, spatial_features).

        Returns:
            Pooled feature vector (B, 1280) or (B, T, 1280),
            or tuple of (pooled, spatial) where spatial has shape (B, 1280, h, w) or (B, T, 1280, h, w).
        """
        if x.dim() not in (4, 5):
            raise ValueError(f"Expected 4D or 5D input tensor, got {x.dim()}D tensor with shape {tuple(x.shape)}")

        is_sequence = (x.dim() == 5)
        if is_sequence:
            b, t, c, h, w = x.shape
            if b <= 0 or t <= 0:
                raise ValueError(f"Batch size B and sequence length T must be > 0, got B={b}, T={t}")
            # Flatten B and T: (B * T, C, H, W)
            x_flat = x.view(b * t, c, h, w)
        else:
            b, c, h, w = x.shape
            if b <= 0:
                raise ValueError(f"Batch size B must be > 0, got B={b}")
            x_flat = x

        # Extract spatial features: (B', 1280, h', w')
        spatial = self.features(x_flat)
        pooled = self.avgpool(spatial).flatten(1)  # (B', 1280)

        if is_sequence:
            # Reshape back to sequence
            pooled = pooled.view(b, t, -1)  # (B, T, 1280)
            if return_spatial_map:
                c_out, h_out, w_out = spatial.shape[1:]
                spatial = spatial.view(b, t, c_out, h_out, w_out)

        if return_spatial_map:
            return pooled, spatial
        return pooled

    def get_provenance(self) -> Dict[str, Any]:
        """Return backbone provenance metadata."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": self.config.name,
            "weights": self.config.weights,
            "freeze_strategy": self.config.freeze_strategy,
            "freeze_batchnorm": self.config.freeze_batchnorm,
            "output_dim": self.config.output_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
