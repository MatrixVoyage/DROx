"""Unit and Contract Tests for Milestone 15 Arm B Architecture.

Verifies:
1. CrossAttentionTemporalSpatialFusion tensor shapes and gradients.
2. LocalTemporalRefinementBlock short-window temporal modeling.
3. M15CrossAttentionDetector forward pass, dataclass output, and probabilities.
4. M15Loss computation and backward gradient propagation.
5. Parameter budget adherence (Total <= 31.0M, Trainable <= 6.5M).
6. Scoped deterministic inference in eval mode.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.detector import DeepfakeDetector
from src.models.m15_cross_attention_detector import (
    CrossAttentionTemporalSpatialFusion,
    LocalTemporalRefinementBlock,
    M15CrossAttentionDetector,
    M15DetectorOutput,
    M15Loss,
)
from src.utils.config import (
    AttentionConfig,
    BackboneConfig,
    ClassifierConfig,
    ModelConfig,
    TemporalConfig,
)


@pytest.fixture
def efficientnet_config() -> ModelConfig:
    """Fixture providing validated EfficientNet-B0 config."""
    return ModelConfig(
        backbone=BackboneConfig(
            name="efficientnet_b0",
            weights=None,
            input_size=[112, 112],
            output_dim=1280,
            spatial_dim=4,
            freeze_strategy="early",
        ),
        attention=AttentionConfig(type="soft", in_channels=1280),
        temporal=TemporalConfig(
            type="transformer",
            input_dim=1280,
            hidden_dim=1280,
            num_layers=2,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            pooling="cls_token",
        ),
        classifier=ClassifierConfig(
            in_features=1280,
            hidden_dim=512,
            num_classes=1,
            dropout=0.5,
        ),
    )


def test_cross_attention_fusion_shapes_and_gradient():
    """Verify CrossAttentionTemporalSpatialFusion produces exact tensor shapes and propagates gradients."""
    B, T, D_temp, C_s, H_s, W_s = 2, 8, 1280, 112, 7, 7
    fusion = CrossAttentionTemporalSpatialFusion(
        d_temporal=D_temp,
        c_spatial=C_s,
        d_attn=128,
        nhead=4,
        gate_bottleneck=64,
        dropout=0.0,
    )

    h_temp = torch.randn(B, D_temp, requires_grad=True)
    s_feat = torch.randn(B * T, C_s, H_s, W_s, requires_grad=True)

    (
        h_fused,
        gate,
        attn_w,
        t_marginal,
        s_marginal,
        entropy,
    ) = fusion(h_temp, s_feat, T=T)

    # 1. Shape assertions
    assert h_fused.shape == (B, D_temp), f"Expected ({B}, {D_temp}), got {h_fused.shape}"
    assert gate.shape == (B, D_temp), f"Expected ({B}, {D_temp}), got {gate.shape}"
    assert (gate >= 0.0).all() and (gate <= 1.0).all(), "Gate values must be in [0, 1]"
    assert attn_w.shape == (B, 4, 1, T * H_s * W_s)
    assert t_marginal.shape == (B, T)
    assert s_marginal.shape == (B, H_s, W_s)
    assert entropy.shape == (B,)

    # 2. Gradient flow assertion
    loss = h_fused.sum() + entropy.sum()
    loss.backward()

    assert h_temp.grad is not None and not torch.isnan(h_temp.grad).any(), "Missing h_temp gradient"
    assert s_feat.grad is not None and not torch.isnan(s_feat.grad).any(), "Missing s_feat gradient"


def test_local_temporal_refinement():
    """Verify LocalTemporalRefinementBlock short-window temporal convolutions."""
    B, T, D = 2, 16, 1280
    refiner = LocalTemporalRefinementBlock(d_model=D, kernel_size=3, dropout=0.0)
    x = torch.randn(B, T, D, requires_grad=True)

    out = refiner(x)
    assert out.shape == (B, T, D)

    out.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()


def test_m15_detector_forward_and_dataclass(efficientnet_config):
    """Verify M15CrossAttentionDetector forward pass and dataclass outputs."""
    base_detector = DeepfakeDetector(config=efficientnet_config)
    detector = M15CrossAttentionDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=256,
        d_attn=128,
        nhead=4,
        gate_bottleneck=64,
        mid_in_channels=112,
        enable_local_refinement=True,
    )

    B, T = 2, 4
    x = torch.randn(B, T, 3, 112, 112)

    detector.eval()
    with torch.no_grad():
        out: M15DetectorOutput = detector(x, return_dict=True)

    assert isinstance(out, M15DetectorOutput)
    assert out.primary_logits.shape == (B, 1)
    assert out.aux_family_logits.shape == (B, 5)
    assert out.temporal_features.shape == (B, 1280)
    assert out.fused_features.shape == (B, 1280)
    assert out.gate_values.shape == (B, 1280)
    assert out.temporal_attention_weights.shape == (B, T)
    assert out.spatial_attention_map.shape == (B, 7, 7)
    assert out.attention_entropy.shape == (B,)

    # Probability bounds
    assert (out.p_fused >= 0.0).all() and (out.p_fused <= 1.0).all()
    assert (out.p_aux_fake >= 0.0).all() and (out.p_aux_fake <= 1.0).all()
    assert (out.p_composite >= 0.0).all() and (out.p_composite <= 1.0).all()

    # Tuple return mode
    p_logits, aux_logits = detector(x, return_dict=False)
    assert p_logits.shape == (B, 1)
    assert aux_logits.shape == (B, 5)


def test_m15_loss_computation(efficientnet_config):
    """Verify M15Loss computes composite loss and backpropagates cleanly."""
    base_detector = DeepfakeDetector(config=efficientnet_config)
    detector = M15CrossAttentionDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=256,
        d_attn=128,
        nhead=4,
        gate_bottleneck=64,
    )
    loss_fn = M15Loss(lambda_aux=0.30, lambda_gate=0.05)

    B, T = 2, 4
    x = torch.randn(B, T, 3, 112, 112)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    family_targets = torch.tensor([1, 0], dtype=torch.long)

    detector.train()
    out = detector(x, return_dict=True)
    loss_dict = loss_fn(out, targets, family_targets)

    assert "loss_total" in loss_dict
    assert "loss_bce" in loss_dict
    assert "loss_aux" in loss_dict
    assert "loss_sparsity" in loss_dict

    loss_dict["loss_total"].backward()

    # Ensure gradients propagate to primary and auxiliary head parameters
    for p in detector.primary_head.parameters():
        assert p.grad is not None
    for p in detector.aux_head.parameters():
        assert p.grad is not None


def test_parameter_budget_invariants(efficientnet_config):
    """Verify M15 preregistered parameter budgets: <= 31.0M total, <= 6.5M trainable."""
    base_detector = DeepfakeDetector(config=efficientnet_config)
    detector = M15CrossAttentionDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=512,
        d_attn=256,
        nhead=4,
        gate_bottleneck=128,
        mid_in_channels=112,
        enable_local_refinement=False,
    )

    total_params = sum(p.numel() for p in detector.parameters())
    assert total_params <= 31_000_000, f"Total params budget violated: {total_params:,} > 31.0M"

    detector.freeze_stages_0_to_4()
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    assert trainable_params <= 6_500_000, f"Trainable params budget violated: {trainable_params:,} > 6.5M"

    # Verify Stage 5 is trainable and Stages 0-4 are frozen
    for i in range(5):
        for p in detector.base_detector.backbone.features[i].parameters():
            assert not p.requires_grad, f"Stage {i} should be frozen"

    for p in detector.base_detector.backbone.features[5].parameters():
        assert p.requires_grad, "Stage 5 should be trainable"


def test_deterministic_inference_eval_mode(efficientnet_config):
    """Verify deterministic repeatability of forward pass in eval mode."""
    base_detector = DeepfakeDetector(config=efficientnet_config)
    detector = M15CrossAttentionDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=256,
        d_attn=128,
        nhead=4,
        gate_bottleneck=64,
    )
    detector.eval()

    x = torch.randn(1, 4, 3, 112, 112)

    with torch.no_grad():
        out1 = detector(x, return_dict=True)
        out2 = detector(x, return_dict=True)

    diff_primary = (out1.primary_logits - out2.primary_logits).abs().max().item()
    diff_aux = (out1.aux_family_logits - out2.aux_family_logits).abs().max().item()
    diff_entropy = (out1.attention_entropy - out2.attention_entropy).abs().max().item()

    assert diff_primary == 0.0, f"Non-zero primary diff: {diff_primary}"
    assert diff_aux == 0.0, f"Non-zero aux diff: {diff_aux}"
    assert diff_entropy == 0.0, f"Non-zero entropy diff: {diff_entropy}"


def test_cuda_block_dct_compression():
    """Verify CUDABlockDCTCompression preserves tensor shape, normalization, and applies quantization."""
    from src.models.m15_cross_attention_detector import CUDABlockDCTCompression

    device = torch.device("cpu")
    compressor = CUDABlockDCTCompression(device=device)

    B, T, C, H, W = 2, 4, 3, 112, 112
    x = torch.randn(B, T, C, H, W)

    x_q50 = compressor(x, quality=50)
    assert x_q50.shape == (B, T, C, H, W)
    assert not torch.isnan(x_q50).any()

    # Verify that quality=25 produces more distortion than quality=90
    x_q25 = compressor(x, quality=25)
    x_q90 = compressor(x, quality=90)

    diff_q25 = (x - x_q25).abs().mean().item()
    diff_q90 = (x - x_q90).abs().mean().item()

    assert diff_q25 > diff_q90, f"Expected Q25 distortion ({diff_q25:.4f}) > Q90 distortion ({diff_q90:.4f})"


def test_m15_robust_loss_computation(efficientnet_config):
    """Verify M15RobustLoss computes frequency consistency and backpropagates cleanly."""
    from src.models.m15_cross_attention_detector import M15RobustLoss, CUDABlockDCTCompression

    base_detector = DeepfakeDetector(config=efficientnet_config)
    detector = M15CrossAttentionDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=256,
        d_attn=128,
        nhead=4,
        gate_bottleneck=64,
    )
    compressor = CUDABlockDCTCompression(device=torch.device("cpu"))
    loss_fn = M15RobustLoss(lambda_aux=0.30, lambda_gate=0.05, lambda_freq=0.10)

    B, T = 2, 4
    x = torch.randn(B, T, 3, 112, 112)
    x_jitter = compressor(x, quality=50)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    family_targets = torch.tensor([1, 0], dtype=torch.long)

    detector.train()
    out_clean = detector(x, return_dict=True)
    out_jitter = detector(x_jitter, return_dict=True)

    loss_dict = loss_fn(out_clean, out_jitter, targets, family_targets)

    assert "loss_total" in loss_dict
    assert "loss_bce" in loss_dict
    assert "loss_aux" in loss_dict
    assert "loss_sparsity" in loss_dict
    assert "loss_freq_con" in loss_dict
    assert loss_dict["loss_freq_con"].item() > 0.0

    loss_dict["loss_total"].backward()

    for p in detector.primary_head.parameters():
        assert p.grad is not None
    for p in detector.aux_head.parameters():
        assert p.grad is not None


def test_m15_ablation_no_gate_detector(efficientnet_config):
    """Verify Arm D M15AblationNoGateDetector forward pass and parameter budgets."""
    from src.models.m15_ablation_detector import LinearTemporalSpatialFusion, M15AblationNoGateDetector

    base_detector = DeepfakeDetector(config=efficientnet_config)
    detector = M15AblationNoGateDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=256,
        d_hidden=128,
        mid_in_channels=112,
    )
    detector.freeze_stages_0_to_4()

    # Parameter count verification
    total_params = sum(p.numel() for p in detector.parameters())
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)

    assert total_params <= 31_000_000, f"Total params exceeded budget: {total_params}"
    assert trainable_params <= 6_500_000, f"Trainable params exceeded budget: {trainable_params}"

    # Forward pass verification
    B, T = 2, 4
    x = torch.randn(B, T, 3, 112, 112)
    out = detector(x, return_dict=True)

    assert out.primary_logits.shape == (B, 1)
    assert out.p_fused.shape == (B, 1)
    assert out.p_composite.shape == (B, 1)
    assert out.aux_family_logits.shape == (B, 5)
    assert out.p_aux_fake.shape == (B, 1)
    assert (out.gate_values == 1.0).all(), "Arm D gate should be constant 1.0 (no dynamic gating)"
    assert out.attention_entropy.shape == (B,)

    # Gradient flow test
    loss = out.p_composite.sum()
    loss.backward()
    for p in detector.primary_head.parameters():
        assert p.grad is not None
    for p in detector.linear_fusion.parameters():
        assert p.grad is not None

