"""Milestone 15 Arm D: Ablation Arm (M15-Ablation-NoGate).

Preregistered Protocol Reference: context/m15_preregistration.md (Section 4, Table Arm D)
Governing Rule: Rule 2, Rule 6 & 10 (Research Integrity), Rule 7 (Modularity), Rule 8 (RTX 3050 6GB VRAM)

Ablation Formulation:
- Replaces Cross-Attention and Dynamic Gating with standard linear projection and addition.
- Tests whether empirical gains observed in Arm B (e.g. +4.67% external AUC on Celeb-DF) stem
  specifically from dynamic cross-modal attention gating versus merely injecting Stage-5 spatial
  features with extra parameter capacity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detector import DeepfakeDetector
from src.models.m15_cross_attention_detector import M15DetectorOutput


class LinearTemporalSpatialFusion(nn.Module):
    """Linear Projection and Addition Fusion Module (No Cross-Attention, No Dynamic Gating).

    Ablation counterpart to CrossAttentionTemporalSpatialFusion:
    - Spatially average pools Stage-5 feature maps: (B*T, C_s, H_s, W_s) -> (B*T, C_s)
    - Temporally average pools across sequence: (B, T, C_s) -> (B, C_s)
    - Linearly projects to temporal representation dimension D_temporal via a 2-layer MLP
      with matching parameter capacity.
    - Fuses via standard un-gated linear addition: LayerNorm(h_temp + proj(s_pooled)).
    """

    def __init__(
        self,
        d_temporal: int = 1280,
        c_spatial: int = 112,
        d_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_temporal = d_temporal
        self.c_spatial = c_spatial

        # Projection MLP to map pooled spatial features to temporal feature space
        self.proj = nn.Sequential(
            nn.Linear(c_spatial, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_temporal),
        )

        self.norm = nn.LayerNorm(d_temporal)

    def forward(
        self,
        h_temp: torch.Tensor,
        s_features: torch.Tensor,
        T: int = 16,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Forward pass through linear fusion.

        Args:
            h_temp: Temporal representations of shape (B, D_temporal).
            s_features: Stage-5 spatial feature maps of shape (B * T, C_s, H_s, W_s).
            T: Number of frames in the sequence (default: 16).

        Returns:
            Tuple matching CrossAttentionTemporalSpatialFusion signature:
            - h_fused: (B, D_temporal) fused representation
            - gate: (B, D_temporal) constant 1.0 (no gating applied)
            - attn_weights: None (no cross-attention)
            - temp_marginal: None
            - spatial_marginal: None
            - entropy: (B,) zeros (no attention entropy)
        """
        B = h_temp.size(0)
        _, C_s, H_s, W_s = s_features.shape

        # Global average pool over spatial dimensions (H_s, W_s)
        s_spatial_pooled = s_features.mean(dim=(-2, -1))  # (B * T, C_s)

        # Average pool over temporal dimension T
        s_seq = s_spatial_pooled.view(B, T, C_s)
        s_global = s_seq.mean(dim=1)  # (B, C_s)

        # Linear projection into temporal space
        s_proj = self.proj(s_global)  # (B, D_temporal)

        # Un-gated linear addition with normalization
        h_fused = self.norm(h_temp + s_proj)

        # Constant unitary gate (representing absence of dynamic suppression/gating)
        gate = torch.ones_like(h_temp)
        entropy = torch.zeros(B, device=h_temp.device, dtype=h_temp.dtype)

        return (
            h_fused,
            gate,
            None,
            None,
            None,
            entropy,
        )


class M15AblationNoGateDetector(nn.Module):
    """Milestone 15 Arm D Ablation Deepfake Detector (Linear Addition, No Gating).

    Directly tests whether dynamic attention gating is the active mechanism
    distinguishing M15-TS-CrossAttn from baseline architectures.
    """

    def __init__(
        self,
        base_detector: Optional[DeepfakeDetector] = None,
        config: Optional[Any] = None,
        num_families: int = 5,
        aux_hidden_dim: int = 512,
        aux_dropout: float = 0.5,
        d_hidden: int = 256,
        mid_in_channels: int = 112,
    ):
        super().__init__()
        self.base_detector = base_detector or DeepfakeDetector(config=config)
        self.temporal_dim = getattr(
            self.base_detector.config.temporal, "hidden_dim", 1280
        )
        self.num_families = num_families
        self.mid_in_channels = mid_in_channels

        # Stage-5 Activation Hook
        self._mid_features: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._register_stage5_hook()

        # Arm D Linear Temporal-Spatial Fusion Module (No Cross-Attention, No Gating)
        self.linear_fusion = LinearTemporalSpatialFusion(
            d_temporal=self.temporal_dim,
            c_spatial=mid_in_channels,
            d_hidden=d_hidden,
            dropout=0.1,
        )

        # Supervised Auxiliary Manipulation Family Classifier Head
        self.aux_head = nn.Sequential(
            nn.Linear(self.temporal_dim, aux_hidden_dim),
            nn.BatchNorm1d(aux_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=aux_dropout),
            nn.Linear(aux_hidden_dim, num_families),
        )

        # Primary Binary Classification Head on Fused Representations
        self.primary_head = nn.Sequential(
            nn.Linear(self.temporal_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 1),
        )

    def _register_stage5_hook(self) -> None:
        """Register forward hook on Stage 5 of EfficientNet-B0 backbone."""
        features = self.base_detector.backbone.features
        target_stage = features[5]

        def _hook_fn(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
            self._mid_features = output

        self._hook_handle = target_stage.register_forward_hook(_hook_fn)

    def freeze_stages_0_to_4(self) -> None:
        """Freeze backbone features[0..4] while keeping stage 5 trainable."""
        if hasattr(self.base_detector.backbone, "features"):
            for i in range(5):
                for p in self.base_detector.backbone.features[i].parameters():
                    p.requires_grad = False

            for p in self.base_detector.backbone.features[5].parameters():
                p.requires_grad = True

            for i in range(6, len(self.base_detector.backbone.features)):
                for p in self.base_detector.backbone.features[i].parameters():
                    p.requires_grad = False

        for p in self.base_detector.attention.parameters():
            p.requires_grad = False

        for p in self.base_detector.temporal.parameters():
            p.requires_grad = False

        for p in self.base_detector.classifier.parameters():
            p.requires_grad = False

        for p in self.linear_fusion.parameters():
            p.requires_grad = True

        for p in self.aux_head.parameters():
            p.requires_grad = True

        for p in self.primary_head.parameters():
            p.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert trainable_params <= 6_500_000, (
            f"Trainable parameter budget violated: {trainable_params:,} > 6,500,000"
        )

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[M15DetectorOutput, torch.Tensor]:
        """Forward pass for Arm D Ablation model."""
        B, T, C, H, W = x.shape
        self._mid_features = None

        # Base forward pass
        _, spatial_maps = self.base_detector.backbone(x, return_spatial_map=True)
        frame_features, _ = self.base_detector.attention(spatial_maps, return_attention_maps=True)
        temporal_features = self.base_detector.temporal(frame_features)
        logits_temp = self.base_detector.classifier(temporal_features)
        p_temp = torch.sigmoid(logits_temp)

        assert self._mid_features is not None, (
            "Stage-5 spatial features were not captured by forward hook."
        )
        s_features = self._mid_features

        # Linear Fusion without Attention or Dynamic Gating
        (
            h_fused,
            gate,
            attn_weights,
            temp_marginal,
            spatial_marginal,
            entropy,
        ) = self.linear_fusion(temporal_features, s_features, T=T)

        # Classification Heads
        primary_logits = self.primary_head(h_fused)
        p_fused = torch.sigmoid(primary_logits)

        aux_family_logits = self.aux_head(temporal_features)
        aux_probs = torch.softmax(aux_family_logits, dim=-1)
        p_aux_fake = (1.0 - aux_probs[:, 0:1]).clamp(0.0, 1.0)
        p_composite = (0.70 * p_fused + 0.30 * p_aux_fake).clamp(0.0, 1.0)

        if not return_dict:
            return primary_logits, aux_family_logits

        # Contract-compliant dummy attention tensors
        _, _, H_s, W_s = s_features.shape
        dummy_cross_attn = torch.zeros(B, 1, 1, T * H_s * W_s, device=x.device, dtype=x.dtype)
        dummy_temp_attn = torch.ones(B, T, device=x.device, dtype=x.dtype) / T
        dummy_spatial_map = torch.ones(B, H_s, W_s, device=x.device, dtype=x.dtype) / float(H_s * W_s)

        return M15DetectorOutput(
            primary_logits=primary_logits,
            aux_family_logits=aux_family_logits,
            temporal_features=temporal_features,
            spatial_features=s_features,
            fused_features=h_fused,
            gate_values=gate,
            cross_attention_weights=dummy_cross_attn,
            temporal_attention_weights=dummy_temp_attn,
            spatial_attention_map=dummy_spatial_map,
            attention_entropy=entropy,
            p_fused=p_fused,
            p_aux_fake=p_aux_fake,
            p_composite=p_composite,
        )
