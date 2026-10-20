"""Milestone 15 Arm B: Temporal-Spatial Cross-Attention Detector (M15-TS-CrossAttn).

Primary Architectural Intervention for Hypothesis H-M15-1 and H-M15-2:
1. Spatial Backbone: EfficientNet-B0 with forward hook on Stage-5 (features[5] -> 112 x 7 x 7).
2. Temporal Aggregator: Global Multi-Head Self-Attention Temporal Transformer (D=1280).
3. Local Temporal Refinement (Optional): 1D depthwise short-window conv capturing inter-frame phase transitions.
4. Cross-Attention Fusion: Penultimate temporal representations query Stage-5 mid-level spatial maps.
5. Dynamic Adaptive Gating: Sigmoidal gate modulating temporal trunk dominance.
6. Multi-Family Auxiliary Head: Supervised manipulation family classification (K=5).
7. Primary Classification Head: Binary deepfake classification from cross-attended representation.

Pre-registered Constraints:
- Total Parameters: <= 31.0 M
- Trainable Parameters: <= 6.5 M
- Peak VRAM (Training): <= 5.2 GB on RTX 3050 CUDA
- Inference Latency: <= 35.0 ms P50
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detector import DeepfakeDetector
from src.models.heads.auxiliary import AuxiliaryManipulationHead
from src.utils.config import ModelConfig


@dataclass
class M15DetectorOutput:
    """Comprehensive diagnostic and evaluation container for M15-TS-CrossAttn."""

    primary_logits: torch.Tensor             # (B, 1) binary classification logits from fused head
    aux_family_logits: torch.Tensor          # (B, K) multi-family manipulation logits
    temporal_features: torch.Tensor          # (B, D_temp) video-level temporal representation
    spatial_features: torch.Tensor           # (B*T, C_s, H_s, W_s) raw Stage-5 spatial feature maps
    fused_features: torch.Tensor             # (B, D_temp) dynamically gated cross-attention representation
    gate_values: torch.Tensor                # (B, D_temp) gating vector g in [0, 1]
    cross_attention_weights: torch.Tensor    # (B, nhead, 1, N_s) full cross-attention weight matrix
    temporal_attention_weights: torch.Tensor # (B, T) marginal attention across sequence frames
    spatial_attention_map: torch.Tensor      # (B, H_s, W_s) marginal spatial attention across face grid
    attention_entropy: torch.Tensor          # (B,) scalar attention entropy for sparsity regularization
    p_fused: torch.Tensor                    # Sigmoid probability from primary fused head
    p_aux_fake: torch.Tensor                 # 1 - P(Real) from auxiliary family head
    p_composite: torch.Tensor                # Weighted ensemble composite probability


class LocalTemporalRefinementBlock(nn.Module):
    """Hierarchical short-window (2-4 frame) temporal refinement block.

    Addresses Hypothesis H-M15-2:
    Captures localized inter-frame expression phase desynchronization
    (e.g., micro-blinks, phoneme timing transitions) before global aggregation.
    """

    def __init__(
        self,
        d_model: int = 1280,
        bottleneck_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        padding = (kernel_size - 1) // 2

        # Parameter-efficient bottleneck 1D convolution along temporal dimension
        self.reduce = nn.Conv1d(
            in_channels=d_model,
            out_channels=bottleneck_dim,
            kernel_size=1,
            bias=False,
        )
        self.depthwise = nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=bottleneck_dim,
            bias=False,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.expand = nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=d_model,
            kernel_size=1,
            bias=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over sequence representations.

        Args:
            x: Frame features of shape (B, T, D).

        Returns:
            Refined frame features of shape (B, T, D).
        """
        # (B, T, D) -> (B, D, T) for 1D convolution
        residual = x
        out = x.transpose(1, 2)
        out = self.reduce(out)
        out = self.depthwise(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.expand(out)
        out = out.transpose(1, 2)
        return self.norm(residual + out)


class CrossAttentionTemporalSpatialFusion(nn.Module):
    """Multi-Head Cross-Attention Gated Temporal-Spatial Fusion Module.

    Addresses Hypothesis H-M15-1:
    Penultimate temporal sequence representations query Stage-5 mid-level
    spatial feature maps, preventing the temporal trunk from unconditionally
    suppressing primary detection scores when synthesis boundary seams are absent.
    """

    def __init__(
        self,
        d_temporal: int = 1280,
        c_spatial: int = 112,
        d_attn: int = 256,
        nhead: int = 4,
        gate_bottleneck: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_attn % nhead != 0:
            raise ValueError(f"d_attn ({d_attn}) must be divisible by nhead ({nhead})")

        self.d_temporal = d_temporal
        self.c_spatial = c_spatial
        self.d_attn = d_attn
        self.nhead = nhead
        self.head_dim = d_attn // nhead

        # Cross-Attention Projections
        self.q_proj = nn.Linear(d_temporal, d_attn)
        self.k_proj = nn.Linear(c_spatial, d_attn)
        self.v_proj = nn.Linear(c_spatial, d_attn)
        self.out_proj = nn.Linear(d_attn, d_temporal)
        self.dropout = nn.Dropout(dropout)

        # Dynamic Adaptive Gating Network
        # Bottleneck MLP mapping [h_temp; c_spatial_proj] -> gating vector in [0, 1]^D
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_temporal * 2, gate_bottleneck),
            nn.LayerNorm(gate_bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_bottleneck, d_temporal),
        )
        # Initialize gate bias to -2.0 so initial gate is close to zero (~0.12),
        # ensuring smooth warm-up without disrupting pretrained temporal representations
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)

        self.norm = nn.LayerNorm(d_temporal)

    def forward(
        self,
        h_temp: torch.Tensor,
        s_features: torch.Tensor,
        T: int = 16,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass through cross-attention fusion.

        Args:
            h_temp: Temporal representations of shape (B, D_temporal).
            s_features: Stage-5 spatial feature maps of shape (B * T, C_s, H_s, W_s).
            T: Number of frames in the sequence (default: 16).

        Returns:
            Tuple of:
            - h_fused: (B, D_temporal) fused representation
            - gate: (B, D_temporal) gating vector values
            - attn_weights: (B, nhead, 1, T * H_s * W_s) full attention weights
            - temp_marginal: (B, T) marginal attention weights across frames
            - spatial_marginal: (B, H_s, W_s) marginal spatial attention across grid
            - entropy: (B,) attention entropy for sparsity regularization
        """
        B = h_temp.size(0)
        _, C_s, H_s, W_s = s_features.shape
        N_spatial = H_s * W_s
        total_tokens = T * N_spatial

        # Reshape spatial features: (B * T, C_s, H_s, W_s) -> (B, total_tokens, C_s)
        s_seq = (
            s_features.view(B, T, C_s, N_spatial)
            .permute(0, 1, 3, 2)
            .reshape(B, total_tokens, C_s)
        )

        # Compute Query, Key, Value
        # Q: (B, 1, d_attn)
        Q = self.q_proj(h_temp).unsqueeze(1)
        # K, V: (B, total_tokens, d_attn)
        K = self.k_proj(s_seq)
        V = self.v_proj(s_seq)

        # Multi-Head Attention Reshape
        # Q -> (B, nhead, 1, head_dim)
        Q = Q.view(B, 1, self.nhead, self.head_dim).transpose(1, 2)
        # K, V -> (B, nhead, total_tokens, head_dim)
        K = K.view(B, total_tokens, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, total_tokens, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: (B, nhead, 1, total_tokens)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Context vector aggregation: (B, nhead, 1, head_dim) -> (B, 1, d_attn)
        context = torch.matmul(attn_weights_drop, V)
        context = context.transpose(1, 2).reshape(B, 1, self.d_attn)
        context_proj = self.out_proj(context).squeeze(1)  # (B, D_temporal)

        # Dynamic Gating Vector computation: (B, D_temporal)
        gate_input = torch.cat([h_temp, context_proj], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))

        # Gated Residual Fusion: h_fused = LayerNorm(h_temp + g * context_proj)
        h_fused = self.norm(h_temp + gate * context_proj)

        # Compute Attention Entropy for Sparsity Loss (H-M15-1 regularization)
        eps = 1e-8
        entropy = -torch.sum(
            attn_weights * torch.log(attn_weights + eps), dim=-1
        ).mean(dim=1).squeeze(1)  # (B,)

        # Compute Marginal Attention Profiles for Diagnostics & CAM
        attn_avg = attn_weights.mean(dim=1).squeeze(1)  # (B, total_tokens)
        attn_grid = attn_avg.view(B, T, H_s, W_s)
        temp_marginal = attn_grid.sum(dim=(2, 3))        # (B, T)
        spatial_marginal = attn_grid.sum(dim=1)          # (B, H_s, W_s)

        return (
            h_fused,
            gate,
            attn_weights,
            temp_marginal,
            spatial_marginal,
            entropy,
        )


class M15CrossAttentionDetector(nn.Module):
    """Milestone 15 Temporal-Spatial Cross-Attention Deepfake Detector.

    Integrates:
    - Base DeepfakeDetector (EfficientNet-B0 + Spatial Attention + Temporal Transformer).
    - Stage-5 Spatial Activation Hook (features[5] -> 112 x 7 x 7).
    - CrossAttentionTemporalSpatialFusion (Dynamic gating and localized spatial attention).
    - Auxiliary Manipulation Family Head (Supervised K=5 classification).
    - Primary Binary Classification Head on fused representations.

    Architectural Invariants & Governance:
    - Preserves modular base detector without hardcoded couplings.
    - Total parameters <= 31.0 M (strictly enforced).
    - Trainable parameters <= 6.5 M under stage-freezing (strictly enforced).
    - Bitwise safe: base detector weights can be initialized from frozen M12 baseline.
    """

    def __init__(
        self,
        base_detector: Optional[DeepfakeDetector] = None,
        config: Optional[ModelConfig] = None,
        num_families: int = 5,
        aux_hidden_dim: int = 512,
        aux_dropout: float = 0.5,
        d_attn: int = 256,
        nhead: int = 4,
        gate_bottleneck: int = 128,
        mid_in_channels: int = 112,
        enable_local_refinement: bool = False,
    ):
        super().__init__()
        self.base_detector = base_detector or DeepfakeDetector(config=config)
        self.temporal_dim = getattr(
            self.base_detector.config.temporal, "hidden_dim", 1280
        )
        self.num_families = num_families
        self.mid_in_channels = mid_in_channels
        self.enable_local_refinement = enable_local_refinement

        # 1. Optional Local Temporal Refinement Block (H-M15-2)
        if enable_local_refinement:
            self.local_temporal = LocalTemporalRefinementBlock(
                d_model=self.temporal_dim,
                kernel_size=3,
                dropout=0.1,
            )
        else:
            self.local_temporal = None

        # 2. Stage-5 Mid-Level Spatial Hook
        self._mid_features: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._register_mid_hook()

        # 3. Dynamic Cross-Attention Temporal-Spatial Fusion Module (H-M15-1)
        self.cross_attn_fusion = CrossAttentionTemporalSpatialFusion(
            d_temporal=self.temporal_dim,
            c_spatial=mid_in_channels,
            d_attn=d_attn,
            nhead=nhead,
            gate_bottleneck=gate_bottleneck,
            dropout=0.1,
        )

        # 4. Multi-Family Auxiliary Head (K=5)
        self.aux_head = AuxiliaryManipulationHead(
            in_features=self.temporal_dim,
            hidden_dim=aux_hidden_dim,
            num_families=num_families,
            dropout=aux_dropout,
        )

        # 5. Dedicated Primary Fused Classification Head
        self.primary_head = nn.Sequential(
            nn.Linear(self.temporal_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        self._verify_parameter_budgets()

    def _register_mid_hook(self) -> None:
        """Register forward hook on Stage 5 of the spatial backbone."""
        def mid_hook(module, input, output):
            self._mid_features = output

        # EfficientNet-B0 Stage 5 is features[5]
        if (
            hasattr(self.base_detector.backbone, "features")
            and len(self.base_detector.backbone.features) > 5
        ):
            self._hook_handle = self.base_detector.backbone.features[5].register_forward_hook(mid_hook)

    def _verify_parameter_budgets(self) -> None:
        """Verify compliance with preregistered parameter budgets."""
        total_params = sum(p.numel() for p in self.parameters())
        # Ceiling: 31.0 M parameters
        assert total_params <= 31_000_000, (
            f"Total parameter budget violated: {total_params:,} > 31,000,000"
        )

    def freeze_stages_0_to_4(self) -> None:
        """Freeze backbone stages 0 to 4 (and stem).

        Keeps Stage 5, Temporal Transformer, Cross-Attention Fusion,
        and Classification Heads trainable.
        Strictly satisfies the <= 6.5 M trainable parameter constraint when
        temporal transformer is kept frozen, or allows fine-tuning within budget.
        """
        # Freeze backbone stages 0 to 4
        if hasattr(self.base_detector.backbone, "features"):
            for i in range(5):
                for p in self.base_detector.backbone.features[i].parameters():
                    p.requires_grad = False

            # Stage 5 (index 5) trainable for spatial adaptation
            for p in self.base_detector.backbone.features[5].parameters():
                p.requires_grad = True

            # Keep stages 6 to 8 frozen to preserve budget
            for i in range(6, len(self.base_detector.backbone.features)):
                for p in self.base_detector.backbone.features[i].parameters():
                    p.requires_grad = False

        # Spatial attention frozen
        for p in self.base_detector.attention.parameters():
            p.requires_grad = False

        # Keep base temporal transformer frozen to stay within <= 6.5M trainable budget
        for p in self.base_detector.temporal.parameters():
            p.requires_grad = False

        # Base classifier frozen (we use primary_head)
        for p in self.base_detector.classifier.parameters():
            p.requires_grad = False

        # Trainable modules: Stage 5, Local Temporal (if enabled), Fusion, Aux Head, Primary Head
        if self.local_temporal is not None:
            for p in self.local_temporal.parameters():
                p.requires_grad = True
        for p in self.cross_attn_fusion.parameters():
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
    ) -> Union[M15DetectorOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through complete M15-TS-CrossAttn detector.

        Args:
            x: Raw input sequence of face crops: (B, T, 3, 112, 112).
            return_dict: If True, returns rich M15DetectorOutput dataclass.

        Returns:
            M15DetectorOutput or (primary_logits, aux_family_logits).
        """
        B, T, C, H, W = x.shape

        # 1. Base detector pass to extract spatial maps and temporal features
        # Note: hook on backbone.features[5] captures Stage-5 spatial activations
        _, spatial_maps = self.base_detector.backbone(x, return_spatial_map=True)
        frame_features, _ = self.base_detector.attention(spatial_maps, return_attention_maps=True)

        # Optional Local Temporal Refinement (H-M15-2)
        if self.local_temporal is not None:
            frame_features = self.local_temporal(frame_features)

        temporal_features = self.base_detector.temporal(frame_features)  # (B, 1280)

        # Ensure Stage-5 spatial features are captured by hook
        assert self._mid_features is not None, (
            "Stage-5 spatial features were not captured by forward hook."
        )
        s_features = self._mid_features  # (B * T, 112, 7, 7)

        # 2. Dynamic Cross-Attention Temporal-Spatial Fusion (H-M15-1)
        (
            h_fused,
            gate,
            attn_weights,
            temp_marginal,
            spatial_marginal,
            entropy,
        ) = self.cross_attn_fusion(
            h_temp=temporal_features,
            s_features=s_features,
            T=T,
        )

        # 3. Heads
        primary_logits = self.primary_head(h_fused)  # (B, 1)
        aux_family_logits = self.aux_head(h_fused)   # (B, K=5)

        # 4. Probabilities
        p_fused = torch.sigmoid(primary_logits)
        aux_probs = torch.softmax(aux_family_logits, dim=-1)
        p_aux_fake = (1.0 - aux_probs[:, 0:1]).clamp(0.0, 1.0)
        p_composite = (0.70 * p_fused + 0.30 * p_aux_fake).clamp(0.0, 1.0)

        if not return_dict:
            return primary_logits, aux_family_logits

        return M15DetectorOutput(
            primary_logits=primary_logits,
            aux_family_logits=aux_family_logits,
            temporal_features=temporal_features,
            spatial_features=s_features,
            fused_features=h_fused,
            gate_values=gate,
            cross_attention_weights=attn_weights,
            temporal_attention_weights=temp_marginal,
            spatial_attention_map=spatial_marginal,
            attention_entropy=entropy,
            p_fused=p_fused,
            p_aux_fake=p_aux_fake,
            p_composite=p_composite,
        )


class M15Loss(nn.Module):
    """Preregistered Milestone 15 Loss Function.

    Formulation:
        L_total = L_BCE(y_hat_fused, y) + lambda_aux * L_CE(z_hat_aux, y_fam) + lambda_gate * L_sparsity(A_cross)
    where:
        - lambda_aux = 0.30
        - lambda_gate = 0.05
    """

    def __init__(
        self,
        lambda_aux: float = 0.30,
        lambda_gate: float = 0.05,
    ):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_gate = lambda_gate
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(
        self,
        output: M15DetectorOutput,
        targets: torch.Tensor,
        family_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total and constituent losses.

        Args:
            output: M15DetectorOutput from model forward pass.
            targets: Binary ground truth (B, 1).
            family_targets: Multi-family ground truth (B,).

        Returns:
            Dictionary of loss tensors.
        """
        l_bce = self.criterion_bce(output.primary_logits, targets)
        l_aux = self.criterion_ce(output.aux_family_logits, family_targets)
        l_sparsity = output.attention_entropy.mean()

        l_total = l_bce + self.lambda_aux * l_aux + self.lambda_gate * l_sparsity

        return {
            "loss_total": l_total,
            "loss_bce": l_bce,
            "loss_aux": l_aux,
            "loss_sparsity": l_sparsity,
        }


# Standard JPEG Luminance Quantization Table
JPEG_LUMA_Q50 = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)


def create_dct_matrix(N: int = 8, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Create 1D DCT-II transform matrix of size N x N."""
    D = torch.zeros(N, N, device=device)
    for k in range(N):
        for n in range(N):
            if k == 0:
                D[k, n] = 1.0 / math.sqrt(N)
            else:
                D[k, n] = math.sqrt(2.0 / N) * math.cos(math.pi * (2 * n + 1) * k / (2.0 * N))
    return D


class CUDABlockDCTCompression(nn.Module):
    """Pure PyTorch CUDA 8x8 Block-DCT JPEG Quantization simulator.

    Operates directly on (B, T, 3, 112, 112) normalized video tensors on GPU.
    Simulates sequence-consistent lossy video compression blocking artifacts.
    """

    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.D = create_dct_matrix(8, device=device)
        self.register_buffer("q50", JPEG_LUMA_Q50.to(device))

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor, quality: int = 50) -> torch.Tensor:
        """Apply simulated 8x8 block-DCT quantization.

        Args:
            x: Video tensor of shape (B, T, 3, 112, 112) in ImageNet normalized space.
            quality: Simulated JPEG quality factor [10, 95].

        Returns:
            Quantized video tensor of same shape and normalization.
        """
        B, T, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0

        # Scale Q-table by quality factor
        if quality < 50:
            scale = 50.0 / quality
        else:
            scale = (100.0 - quality) / 50.0
        q_table = torch.clamp(self.q50 * scale, min=1.0, max=255.0)

        # Denormalize to centered [0, 255] - 128
        x_denorm = torch.clamp((x * self.std + self.mean) * 255.0 - 128.0, -128.0, 127.0)

        # Reshape to 8x8 patches: (B * T * C, n_h, 8, n_w, 8) -> (B * T * C * n_h * n_w, 8, 8)
        n_h, n_w = H // 8, W // 8
        x_flat = x_denorm.view(B * T * C, n_h, 8, n_w, 8).permute(0, 1, 3, 2, 4).contiguous()
        patches = x_flat.view(B * T * C * n_h * n_w, 8, 8)

        # 2D DCT: Y = D @ patches @ D.T
        dct_coeffs = torch.matmul(torch.matmul(self.D, patches), self.D.t())

        # Quantization & Dequantization: round(Y / Q) * Q
        quantized = torch.round(dct_coeffs / q_table) * q_table

        # 2D IDCT: X = D.T @ quantized @ D
        idct_patches = torch.matmul(torch.matmul(self.D.t(), quantized), self.D)

        # Reconstruct image
        idct_flat = idct_patches.view(B * T * C, n_h, n_w, 8, 8).permute(0, 1, 3, 2, 4).contiguous()
        recon = idct_flat.view(B, T, C, H, W) + 128.0
        recon = torch.clamp(recon, 0.0, 255.0)

        # Re-normalize to ImageNet space
        recon_norm = (recon / 255.0 - self.mean) / self.std
        return recon_norm


class M15RobustLoss(nn.Module):
    """Preregistered Milestone 15 Arm C Loss Function with Multi-Scale Frequency Consistency.

    Formulation:
        L_total = L_BCE(y_hat_fused, y) + lambda_aux * L_CE(z_hat_aux, y_fam) 
                + lambda_gate * L_sparsity(A_cross) + lambda_freq * L_freq_con
    where:
        - lambda_aux = 0.30
        - lambda_gate = 0.05
        - lambda_freq = 0.10
        - L_freq_con = L_feat_cos + 0.5 * L_attn_kl + L_bce_jitter
    """

    def __init__(
        self,
        lambda_aux: float = 0.30,
        lambda_gate: float = 0.05,
        lambda_freq: float = 0.10,
    ):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_gate = lambda_gate
        self.lambda_freq = lambda_freq
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        output_clean: M15DetectorOutput,
        output_jitter: Optional[M15DetectorOutput],
        targets: torch.Tensor,
        family_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total and constituent losses including frequency consistency.

        Args:
            output_clean: M15DetectorOutput on pristine input x.
            output_jitter: Optional M15DetectorOutput on compression-jittered input tilde_x.
            targets: Binary ground truth (B, 1).
            family_targets: Multi-family ground truth (B,).

        Returns:
            Dictionary of loss tensors.
        """
        l_bce = self.criterion_bce(output_clean.primary_logits, targets)
        l_aux = self.criterion_ce(output_clean.aux_family_logits, family_targets)
        l_sparsity = output_clean.attention_entropy.mean()

        l_freq_con = torch.tensor(0.0, device=output_clean.primary_logits.device)
        if output_jitter is not None:
            # 1. Feature cosine similarity loss (invariance to compression blocking)
            l_feat = (1.0 - self.cos_sim(output_clean.fused_features, output_jitter.fused_features)).mean()

            # 2. Attention consistency loss (spatial-temporal attention invariance)
            eps = 1e-8
            p_attn = output_clean.cross_attention_weights.clamp(min=eps)
            q_attn = output_jitter.cross_attention_weights.clamp(min=eps)
            l_attn = F.kl_div(q_attn.log(), p_attn, reduction="batchmean")

            # 3. Classification consistency under compression
            l_bce_jitter = self.criterion_bce(output_jitter.primary_logits, targets)

            l_freq_con = l_feat + 0.5 * l_attn + l_bce_jitter

        l_total = (
            l_bce
            + self.lambda_aux * l_aux
            + self.lambda_gate * l_sparsity
            + self.lambda_freq * l_freq_con
        )

        return {
            "loss_total": l_total,
            "loss_bce": l_bce,
            "loss_aux": l_aux,
            "loss_sparsity": l_sparsity,
            "loss_freq_con": l_freq_con,
        }

