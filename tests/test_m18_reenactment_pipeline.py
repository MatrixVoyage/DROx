"""Unit and integration test suite for Milestone 18 Reenactment Pipeline.

Verifies:
1. Differential extraction tensor dimensions and velocity/acceleration calculations.
2. Spatial motion attention dimensions and residual scaling.
3. Kinematic branch forward pass, embedding shapes, and gradient flow.
4. Conditioned gating fusion bounds (alpha in [0, 1]).
5. M18ReenactmentDetector forward passes in dual_stream, kinematic_only, and residual_only modes.
6. Invariance of frozen M17 Track A trunk weights under backpropagation.
7. Batch size 8 FP16 forward pass compatibility.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.detector import DeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.branches.kinematic_reenactment import (
    FrameDifferentialExtractor,
    SpatialMotionAttention,
    KinematicReenactmentBranch,
    ReenactmentGatedFusion,
)
from src.models.m18_detector import M18ReenactmentDetector, M18DetectorOutput
from src.utils.config import load_config


@pytest.fixture
def dummy_video_batch() -> torch.Tensor:
    """Fixture providing synthetic 16-frame video batch of shape (2, 16, 3, 112, 112)."""
    return torch.randn(2, 16, 3, 112, 112, dtype=torch.float32)


@pytest.fixture
def mock_track_a_detector() -> M17TrackADetector:
    """Fixture providing lightweight M17 Track A detector."""
    cfg = load_config("configs/experiments/m12_full.yaml")
    base = DeepfakeDetector(config=cfg.model)
    detector = M17TrackADetector(
        base_detector=base,
        num_families=5,
        res_dim=128,
        aux_hidden_dim=256,
        aux_dropout=0.1,
        gate_bottleneck=64,
    )
    return detector


def test_frame_differential_extractor_shapes(dummy_video_batch: torch.Tensor):
    """Test differential extractor produces expected 6-channel spatiotemporal tensor."""
    extractor = FrameDifferentialExtractor()
    motion_3d = extractor(dummy_video_batch)

    # Expected shape: (B, C_in=6, D=T-1=15, H=112, W=112)
    assert motion_3d.shape == (2, 6, 15, 112, 112)

    # Test error handling when T < 3
    short_batch = torch.randn(2, 2, 3, 112, 112)
    with pytest.raises(ValueError, match="requires T >= 3"):
        extractor(short_batch)


def test_spatial_motion_attention_shapes_and_residuals():
    """Test 3D spatial motion attention preserves dimensions and produces valid activations."""
    attn = SpatialMotionAttention(kernel_size=3)
    feat = torch.randn(2, 64, 15, 28, 28)
    out = attn(feat)

    assert out.shape == feat.shape
    # Verify non-trivial residual contribution
    assert not torch.allclose(out, feat)


def test_kinematic_branch_forward_and_backward(dummy_video_batch: torch.Tensor):
    """Test kinematic branch outputs (B, 256) and propagates gradients cleanly."""
    branch = KinematicReenactmentBranch(in_channels=6, out_dim=256, dropout=0.1)
    z_kin = branch(dummy_video_batch)

    assert z_kin.shape == (2, 256)
    loss = z_kin.sum()
    loss.backward()

    # Verify gradients reach initial 3D convolutional weights
    assert branch.block1[0].weight.grad is not None
    assert torch.any(branch.block1[0].weight.grad != 0.0)


def test_reenactment_gated_fusion_bounds():
    """Test gated fusion produces valid alpha in [0, 1] and fused shape (B, 1280)."""
    fusion = ReenactmentGatedFusion(trunk_dim=1280, kin_dim=256, gate_bottleneck=64)
    z_trunk = torch.randn(2, 1280)
    z_kin = torch.randn(2, 256)

    z_fused, alpha = fusion(z_trunk, z_kin)

    assert z_fused.shape == (2, 1280)
    assert alpha.shape == (2, 1)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)


def test_m18_reenactment_detector_modes(
    dummy_video_batch: torch.Tensor,
    mock_track_a_detector: M17TrackADetector,
):
    """Test M18 detector in dual_stream, kinematic_only, and residual_only modes."""
    detector = M18ReenactmentDetector(
        trunk_detector=mock_track_a_detector,
        kin_dim=256,
        aux_hidden_dim=256,
        gate_bottleneck=64,
        freeze_trunk=True,
    )
    detector.eval()

    with torch.no_grad():
        # 1. Dual stream (default)
        out_dual = detector(dummy_video_batch, mode="dual_stream")
        assert isinstance(out_dual, M18DetectorOutput)
        assert out_dual.primary_logits.shape == (2, 1)
        assert out_dual.kin_logits.shape == (2, 1)
        assert out_dual.aux_logits.shape == (2, 5)
        assert out_dual.gate_values.shape == (2, 1)
        assert torch.all(out_dual.gate_values >= 0.0) and torch.all(out_dual.gate_values <= 1.0)
        assert out_dual.trunk_features.shape == (2, 1280)
        assert out_dual.kin_features.shape == (2, 256)
        assert out_dual.fused_features.shape == (2, 1280)

        # 2. Kinematic only
        out_kin = detector(dummy_video_batch, mode="kinematic_only")
        assert out_kin.primary_logits.shape == (2, 1)
        assert torch.allclose(out_kin.trunk_features, torch.zeros_like(out_kin.trunk_features))

        # 3. Residual only
        out_res = detector(dummy_video_batch, mode="residual_only")
        assert out_res.primary_logits.shape == (2, 1)
        assert torch.allclose(out_res.gate_values, torch.zeros_like(out_res.gate_values))


def test_trunk_weights_frozen_invariant(
    dummy_video_batch: torch.Tensor,
    mock_track_a_detector: M17TrackADetector,
):
    """Verify that trunk weights receive zero gradient while kinematic branch is trainable."""
    detector = M18ReenactmentDetector(
        trunk_detector=mock_track_a_detector,
        kin_dim=256,
        aux_hidden_dim=256,
        gate_bottleneck=64,
        freeze_trunk=True,
    )
    detector.train()

    # Verify all trunk parameters have requires_grad == False
    for p in detector.trunk.parameters():
        assert p.requires_grad is False

    # Forward pass and backward
    out = detector(dummy_video_batch, mode="dual_stream")
    total_loss = out.primary_logits.sum() + out.aux_logits.sum()
    total_loss.backward()

    # Trunk weights must have grad is None
    for p in detector.trunk.parameters():
        assert p.grad is None

    # Kinematic branch and classifier heads must have received gradients
    assert detector.kinematic_branch.block1[0].weight.grad is not None
    assert detector.classifier.weight.grad is not None
    assert detector.aux_family_head[0].weight.grad is not None


def test_batch_size_8_fp16_rtx3050_compatibility(
    mock_track_a_detector: M17TrackADetector,
):
    """Test batch size 8 forward pass with FP16 autocast."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = M18ReenactmentDetector(
        trunk_detector=mock_track_a_detector.to(device),
        kin_dim=256,
        aux_hidden_dim=256,
        gate_bottleneck=64,
        freeze_trunk=True,
    ).to(device)
    detector.eval()

    batch_8 = torch.randn(8, 16, 3, 112, 112, device=device)

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                out = detector(batch_8, mode="dual_stream")
        else:
            out = detector(batch_8, mode="dual_stream")

    assert out.primary_logits.shape == (8, 1)
    assert out.aux_logits.shape == (8, 5)
    assert out.gate_values.shape == (8, 1)
