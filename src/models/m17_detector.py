"""Milestone 17: Cross-Domain Forensic Representation Detectors.

Defines two independent research tracks:
- Track A (M17TrackADetector): Sensor/Residual Representation using multi-scale SRM
  and Bayar high-pass filter banks with dynamic residual gating.
- Track B (M17TrackBDetector): Learned Temporal/Facial Dynamics using anatomical region
  cross-attention across mouth, eyes, jaw, and forehead without hard-coded rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detector import DeepfakeDetector
from src.models.branches.srm_residual import MultiScaleResidualStream
from src.models.branches.region_dynamics import RegionDynamicsBranch
from src.models.heads.auxiliary import AuxiliaryManipulationHead
from src.utils.config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# TRACK A: SENSOR / RESIDUAL REPRESENTATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class M17TrackAOutput:
    """Diagnostic and evaluation output container for M17 Track A (Sensor/Residual)."""

    primary_logits: torch.Tensor          # (B, 1) binary logits from fused head
    res_logits: torch.Tensor              # (B, 1) binary logits from standalone residual stream
    aux_family_logits: torch.Tensor       # (B, K) multi-family manipulation logits
    temporal_features: torch.Tensor       # (B, D_temp) base temporal representation
    res_features: torch.Tensor            # (B, D_res) multi-scale residual representation
    fused_features: torch.Tensor          # (B, D_temp) dynamically gated fused representation
    gate_values: torch.Tensor             # (B, D_temp) gating vector in [0, 1]
    p_fused: torch.Tensor                 # Sigmoid probability from primary fused head
    p_res: torch.Tensor                   # Sigmoid probability from residual branch
    p_aux_fake: torch.Tensor              # 1 - P(Real) from auxiliary family head
    p_composite: torch.Tensor             # Balanced ensemble composite probability


class M17TrackADetector(nn.Module):
    """M17 Track A: Multi-Scale Sensor and Residual Representation Deepfake Detector.

    Combines:
    - Base DeepfakeDetector (EfficientNet-B0 + Temporal Transformer).
    - MultiScaleResidualStream: SRM + Bayar + Block DCT at 112x112, 56x56, 28x28.
    - Dynamic Residual Gating: Modulates residual injection into base representations.
    - Dedicated Residual Head: Evaluates standalone residual signal strength.
    - Multi-Family Auxiliary Head: Supervised regularization across manipulation families.
    - Primary Fused Classification Head.
    """

    def __init__(
        self,
        base_detector: Optional[DeepfakeDetector] = None,
        config: Optional[ModelConfig] = None,
        num_families: int = 5,
        res_dim: int = 128,
        aux_hidden_dim: int = 512,
        aux_dropout: float = 0.5,
        gate_bottleneck: int = 128,
    ):
        super().__init__()
        self.base_detector = base_detector or DeepfakeDetector(config=config)
        self.temporal_dim = getattr(
            self.base_detector.config.temporal, "hidden_dim", 1280
        )
        self.res_dim = res_dim
        self.num_families = num_families

        # 1. Multi-Scale Sensor Residual Stream
        self.residual_stream = MultiScaleResidualStream(
            bayar_channels=8,
            out_features=res_dim,
            hidden_dim=64,
            dropout=0.2,
        )

        # 2. Standalone Residual Classification Head
        self.res_head = nn.Sequential(
            nn.Linear(res_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # 3. Dynamic Adaptive Gating Network
        self.res_proj = nn.Linear(res_dim, self.temporal_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.temporal_dim * 2, gate_bottleneck),
            nn.LayerNorm(gate_bottleneck),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gate_bottleneck, self.temporal_dim),
        )
        # Initialize gate bias to -2.0 so residual stream warms up smoothly (~0.12 init gate)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)

        self.fused_norm = nn.LayerNorm(self.temporal_dim)

        # 4. Multi-Family Auxiliary Head
        self.aux_head = AuxiliaryManipulationHead(
            in_features=self.temporal_dim,
            hidden_dim=aux_hidden_dim,
            num_families=num_families,
            dropout=aux_dropout,
        )

        # 5. Primary Fused Classification Head
        self.fused_head = nn.Sequential(
            nn.Linear(self.temporal_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        active_scales: Optional[Tuple[bool, bool, bool]] = None,
        **kwargs: Any,
    ) -> M17TrackAOutput:
        """Forward pass.

        Args:
            x: Input video tensor of shape (B, T, C, H, W).
            active_scales: Optional 3-tuple (use_s1, use_s2, use_s3) for frozen inference ablation.

        Returns:
            M17TrackAOutput container.
        """
        B, T, C, H, W = x.shape

        # 1. Base Detector Forward Pass (Trunk)
        _, h_temp = self.base_detector(x, return_embeddings=True)  # (B, D_temp)

        # 2. Sensor Residual Stream Forward Pass
        z_res, _ = self.residual_stream(x, active_scales=active_scales)  # (B, D_res)
        res_logits = self.res_head(z_res)   # (B, 1)

        # 3. Dynamic Residual Gating
        z_res_proj = self.res_proj(z_res)   # (B, D_temp)
        gate_input = torch.cat([h_temp, z_res_proj], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # (B, D_temp)

        h_fused = self.fused_norm(h_temp + gate * z_res_proj)

        # 4. Classification Heads
        primary_logits = self.fused_head(h_fused)  # (B, 1)
        aux_logits = self.aux_head(h_temp)          # (B, K)

        # 5. Probabilities
        p_fused = torch.sigmoid(primary_logits)
        p_res = torch.sigmoid(res_logits)
        aux_probs = F.softmax(aux_logits, dim=-1)
        p_aux_fake = 1.0 - aux_probs[:, 0:1]

        # Composite probability: 0.70 fused + 0.15 residual + 0.15 auxiliary
        p_composite = 0.70 * p_fused + 0.15 * p_res + 0.15 * p_aux_fake

        return M17TrackAOutput(
            primary_logits=primary_logits,
            res_logits=res_logits,
            aux_family_logits=aux_logits,
            temporal_features=h_temp,
            res_features=z_res,
            fused_features=h_fused,
            gate_values=gate,
            p_fused=p_fused,
            p_res=p_res,
            p_aux_fake=p_aux_fake,
            p_composite=p_composite,
        )

    def freeze_stages_0_to_4(self) -> None:
        """Freeze backbone stages 0 to 4 (and stem) as well as base temporal trunk."""
        if hasattr(self.base_detector.backbone, "features"):
            for i in range(5):
                for p in self.base_detector.backbone.features[i].parameters():
                    p.requires_grad = False
            for p in self.base_detector.backbone.features[5].parameters():
                p.requires_grad = True
            for i in range(6, len(self.base_detector.backbone.features)):
                for p in self.base_detector.backbone.features[i].parameters():
                    p.requires_grad = False
        elif hasattr(self.base_detector.backbone, "layer1"):
            for stage in [self.base_detector.backbone.stem, self.base_detector.backbone.layer1, self.base_detector.backbone.layer2]:
                for p in stage.parameters():
                    p.requires_grad = False

        if hasattr(self.base_detector, "attention"):
            for p in self.base_detector.attention.parameters():
                p.requires_grad = False
        if hasattr(self.base_detector, "temporal"):
            for p in self.base_detector.temporal.parameters():
                p.requires_grad = False
        if hasattr(self.base_detector, "classifier"):
            for p in self.base_detector.classifier.parameters():
                p.requires_grad = False


# ─────────────────────────────────────────────────────────────────────────────
# TRACK B: LEARNED TEMPORAL / FACIAL DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class M17TrackBOutput:
    """Diagnostic and evaluation output container for M17 Track B (Facial Dynamics)."""

    primary_logits: torch.Tensor          # (B, 1) binary logits from fused head
    dyn_logits: torch.Tensor              # (B, 1) binary logits from standalone dynamics stream
    aux_family_logits: torch.Tensor       # (B, K) multi-family manipulation logits
    temporal_features: torch.Tensor       # (B, D_temp) base temporal representation
    dyn_features: torch.Tensor            # (B, D_dyn) facial dynamics representation
    fused_features: torch.Tensor          # (B, D_temp) dynamically gated fused representation
    gate_values: torch.Tensor             # (B, D_temp) gating vector in [0, 1]
    corr_matrix: torch.Tensor             # (B, K, K) regional temporal correlation matrix
    p_fused: torch.Tensor                 # Sigmoid probability from primary fused head
    p_dyn: torch.Tensor                   # Sigmoid probability from dynamics branch
    p_aux_fake: torch.Tensor              # 1 - P(Real) from auxiliary family head
    p_composite: torch.Tensor             # Balanced ensemble composite probability


class M17TrackBDetector(nn.Module):
    """M17 Track B: Region-Aware Learned Spatiotemporal Facial Dynamics Detector.

    Combines:
    - Base DeepfakeDetector with Stage-5 Spatial Activation Hook (112 x 7 x 7).
    - RegionDynamicsBranch: 4 anatomical region cross-attention + 64-token spatiotemporal transformer.
    - Dynamic Facial Gating: Modulates facial dynamics injection into base representations.
    - Dedicated Dynamics Head: Evaluates standalone biomechanical predictive power.
    - Multi-Family Auxiliary Head: Supervised regularization across manipulation families.
    - Primary Fused Classification Head.
    """

    def __init__(
        self,
        base_detector: Optional[DeepfakeDetector] = None,
        config: Optional[ModelConfig] = None,
        num_families: int = 5,
        dyn_dim: int = 128,
        mid_in_channels: int = 112,
        aux_hidden_dim: int = 512,
        aux_dropout: float = 0.5,
        gate_bottleneck: int = 128,
    ):
        super().__init__()
        self.base_detector = base_detector or DeepfakeDetector(config=config)
        self.temporal_dim = getattr(
            self.base_detector.config.temporal, "hidden_dim", 1280
        )
        self.dyn_dim = dyn_dim
        self.num_families = num_families
        # Auto-detect intermediate spatial channel depth (1024 for ResNeXt-50, 112 for EfficientNet-B0)
        backbone = self.base_detector.backbone
        if hasattr(backbone, "layer3") and mid_in_channels == 112:
            self.mid_in_channels = 1024
        else:
            self.mid_in_channels = mid_in_channels

        # 1. Forward Hook for Stage-5 Spatial Activations
        self._mid_features: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._register_mid_hook()

        # 2. Region-Aware Facial Dynamics Branch
        self.dynamics_branch = RegionDynamicsBranch(
            in_channels=self.mid_in_channels,
            embed_dim=128,
            num_frames=16,
            num_regions=4,
            out_features=dyn_dim,
            dropout=0.2,
        )

        # 3. Standalone Dynamics Classification Head
        self.dyn_head = nn.Sequential(
            nn.Linear(dyn_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # 4. Dynamic Adaptive Gating Network
        self.dyn_proj = nn.Linear(dyn_dim, self.temporal_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.temporal_dim * 2, gate_bottleneck),
            nn.LayerNorm(gate_bottleneck),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gate_bottleneck, self.temporal_dim),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)

        self.fused_norm = nn.LayerNorm(self.temporal_dim)

        # 5. Multi-Family Auxiliary Head
        self.aux_head = AuxiliaryManipulationHead(
            in_features=self.temporal_dim,
            hidden_dim=aux_hidden_dim,
            num_families=num_families,
            dropout=aux_dropout,
        )

        # 6. Primary Fused Classification Head
        self.fused_head = nn.Sequential(
            nn.Linear(self.temporal_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def _register_mid_hook(self) -> None:
        """Register forward hook on Stage 5 of the spatial backbone."""
        def mid_hook(module, input, output):
            self._mid_features = output

        backbone = self.base_detector.backbone
        # Check EfficientNet-B0 or ResNeXt
        if hasattr(backbone, "model") and hasattr(backbone.model, "features"):
            target_layer = backbone.model.features[5]
            self._hook_handle = target_layer.register_forward_hook(mid_hook)
        elif hasattr(backbone, "features") and isinstance(backbone.features, nn.Sequential):
            target_layer = backbone.features[5]
            self._hook_handle = target_layer.register_forward_hook(mid_hook)
        elif hasattr(backbone, "layer3"):  # ResNeXt
            self._hook_handle = backbone.layer3.register_forward_hook(mid_hook)
        else:
            raise RuntimeError("Could not find suitable intermediate layer for Stage-5 hook.")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> M17TrackBOutput:
        """Forward pass.

        Args:
            x: Input video tensor of shape (B, T, C, H, W).

        Returns:
            M17TrackBOutput container.
        """
        B, T, C, H, W = x.shape
        self._mid_features = None

        # 1. Base Detector Forward Pass (Trunk & triggers hook)
        _, h_temp = self.base_detector(x, return_embeddings=True)  # (B, D_temp)

        if self._mid_features is None:
            raise RuntimeError("Stage-5 hook failed to capture intermediate spatial features.")

        # _mid_features is (B*T, C_mid, H_mid, W_mid)
        s_feat = self._mid_features

        # 2. Region-Aware Facial Dynamics Forward Pass
        z_dyn, corr_matrix = self.dynamics_branch(s_feat, B=B, T=T)
        dyn_logits = self.dyn_head(z_dyn)

        # 3. Dynamic Facial Gating
        z_dyn_proj = self.dyn_proj(z_dyn)
        gate_input = torch.cat([h_temp, z_dyn_proj], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))

        h_fused = self.fused_norm(h_temp + gate * z_dyn_proj)

        # 4. Classification Heads
        primary_logits = self.fused_head(h_fused)
        aux_logits = self.aux_head(h_temp)

        # 5. Probabilities
        p_fused = torch.sigmoid(primary_logits)
        p_dyn = torch.sigmoid(dyn_logits)
        aux_probs = F.softmax(aux_logits, dim=-1)
        p_aux_fake = 1.0 - aux_probs[:, 0:1]

        p_composite = 0.70 * p_fused + 0.15 * p_dyn + 0.15 * p_aux_fake

        return M17TrackBOutput(
            primary_logits=primary_logits,
            dyn_logits=dyn_logits,
            aux_family_logits=aux_logits,
            temporal_features=h_temp,
            dyn_features=z_dyn,
            fused_features=h_fused,
            gate_values=gate,
            corr_matrix=corr_matrix,
            p_fused=p_fused,
            p_dyn=p_dyn,
            p_aux_fake=p_aux_fake,
            p_composite=p_composite,
        )

    def freeze_stages_0_to_4(self) -> None:
        """Freeze backbone stages 0 to 4 (and stem) as well as base temporal trunk."""
        if hasattr(self.base_detector.backbone, "features"):
            for i in range(5):
                for p in self.base_detector.backbone.features[i].parameters():
                    p.requires_grad = False
            for p in self.base_detector.backbone.features[5].parameters():
                p.requires_grad = True
            for i in range(6, len(self.base_detector.backbone.features)):
                for p in self.base_detector.backbone.features[i].parameters():
                    p.requires_grad = False
        elif hasattr(self.base_detector.backbone, "layer1"):
            for stage in [self.base_detector.backbone.stem, self.base_detector.backbone.layer1, self.base_detector.backbone.layer2]:
                for p in stage.parameters():
                    p.requires_grad = False

        if hasattr(self.base_detector, "attention"):
            for p in self.base_detector.attention.parameters():
                p.requires_grad = False
        if hasattr(self.base_detector, "temporal"):
            for p in self.base_detector.temporal.parameters():
                p.requires_grad = False
        if hasattr(self.base_detector, "classifier"):
            for p in self.base_detector.classifier.parameters():
                p.requires_grad = False


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS FOR M17 TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class M17TrackALoss(nn.Module):
    """Loss function for M17 Track A (Sensor/Residual).

    L_total = L_BCE(primary) + lambda_res * L_BCE(res) + lambda_aux * L_CE(aux) + lambda_gate * L_reg(gate).
    """

    def __init__(
        self,
        lambda_res: float = 0.20,
        lambda_aux: float = 0.30,
        lambda_gate: float = 0.05,
    ):
        super().__init__()
        self.lambda_res = lambda_res
        self.lambda_aux = lambda_aux
        self.lambda_gate = lambda_gate
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        out: M17TrackAOutput,
        binary_targets: torch.Tensor,
        family_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        binary_targets = binary_targets.view(-1, 1).float()
        l_prim = self.bce(out.primary_logits, binary_targets)
        l_res = self.bce(out.res_logits, binary_targets)

        l_aux = torch.tensor(0.0, device=binary_targets.device)
        if family_targets is not None:
            l_aux = self.ce(out.aux_family_logits, family_targets.long())

        # Gate L2 regularization
        l_gate = torch.mean(out.gate_values ** 2)

        total_loss = l_prim + self.lambda_res * l_res + self.lambda_aux * l_aux + self.lambda_gate * l_gate
        return {
            "loss": total_loss,
            "loss_primary": l_prim,
            "loss_res": l_res,
            "loss_aux": l_aux,
            "loss_gate": l_gate,
        }


class M17TrackBLoss(nn.Module):
    """Loss function for M17 Track B (Facial Dynamics).

    L_total = L_BCE(primary) + lambda_dyn * L_BCE(dyn) + lambda_aux * L_CE(aux) + lambda_gate * L_reg(gate).
    """

    def __init__(
        self,
        lambda_dyn: float = 0.20,
        lambda_aux: float = 0.30,
        lambda_gate: float = 0.05,
    ):
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_aux = lambda_aux
        self.lambda_gate = lambda_gate
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        out: M17TrackBOutput,
        binary_targets: torch.Tensor,
        family_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        binary_targets = binary_targets.view(-1, 1).float()
        l_prim = self.bce(out.primary_logits, binary_targets)
        l_dyn = self.bce(out.dyn_logits, binary_targets)

        l_aux = torch.tensor(0.0, device=binary_targets.device)
        if family_targets is not None:
            l_aux = self.ce(out.aux_family_logits, family_targets.long())

        l_gate = torch.mean(out.gate_values ** 2)

        total_loss = l_prim + self.lambda_dyn * l_dyn + self.lambda_aux * l_aux + self.lambda_gate * l_gate
        return {
            "loss": total_loss,
            "loss_primary": l_prim,
            "loss_dyn": l_dyn,
            "loss_aux": l_aux,
            "loss_gate": l_gate,
        }
