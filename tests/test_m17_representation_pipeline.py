"""Unit and Integration Tests for Milestone 17 (M17) Representation Pipeline.

Validates:
1. Track A: SRM high-pass filter bank, zero-sum constraints, and Bayar high-pass convolution.
2. Track A: Multi-Scale Residual Stream (Scale 1: 112x112, Scale 2: 56x56, Scale 3: 28x28).
3. Track A: M17TrackADetector forward pass, backward gradient flow, and parameter freezing.
4. Track B: AnatomicalRegionExtractor (K=4 regions: mouth, eyes, jaw, forehead).
5. Track B: SpatiotemporalRegionTransformer (64 tokens, kinematic correlation matrix).
6. Track B: M17TrackBDetector forward hook on Stage 5, backward pass, and parameter freezing.
7. M17 Loss Functions: M17TrackALoss and M17TrackBLoss.
8. Simulated JPEG compression robustness function.
9. Verification of frozen baseline (M12) and promoted champion (v5) hashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.branches.srm_residual import (
    SRMConv2d,
    build_srm_filters,
    MultiScaleResidualStream,
)
from src.models.branches.frequency_residual import BayarConv2d
from src.models.branches.region_dynamics import (
    AnatomicalRegionExtractor,
    SpatiotemporalRegionTransformer,
    RegionDynamicsBranch,
)
from src.models.m17_detector import (
    M17TrackADetector,
    M17TrackAOutput,
    M17TrackALoss,
    M17TrackBDetector,
    M17TrackBOutput,
    M17TrackBLoss,
)
from scripts.evaluate_m17_track_a import apply_simulated_jpeg


# ─────────────────────────────────────────────────────────────────────────────
# TRACK A TESTS: SENSOR RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────

def test_srm_filters_zero_sum():
    """Verify SRM linear high-pass filter kernels satisfy the zero-sum high-pass constraint."""
    weights = build_srm_filters()
    assert weights.shape == (3, 1, 5, 5)

    for i in range(3):
        kernel_sum = weights[i, 0].sum().item()
        assert abs(kernel_sum) < 1e-5, f"SRM filter {i} does not sum to 0 (got {kernel_sum})"


def test_bayar_conv_suppression():
    """Verify BayarConv2d produces zero response on constant/flat semantic regions."""
    bayar = BayarConv2d(in_channels=1, out_channels=4, kernel_size=5)
    flat_image = torch.full((2, 1, 32, 32), 128.0)
    filtered = bayar(flat_image)
    # Interior pixels (away from borders) must be strictly 0.0
    interior = filtered[:, :, 4:-4, 4:-4]
    assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-4), (
        f"BayarConv2d failed to suppress flat image: max={interior.abs().max()}"
    )


def test_multiscale_residual_stream_shapes():
    """Verify MultiScaleResidualStream handles 5D video inputs and returns correct shapes."""
    stream = MultiScaleResidualStream(bayar_channels=8, out_features=128, hidden_dim=64)
    x = torch.randn(2, 16, 3, 112, 112)
    z_res, pyramid = stream(x)

    assert z_res.shape == (2, 128)
    assert pyramid.shape == (32, 96, 28, 28)

    # Test backward pass
    loss = z_res.sum()
    loss.backward()
    assert stream.proj[0].weight.grad is not None


def test_track_a_detector_forward_backward():
    """Verify M17TrackADetector forward output container, probabilities, and backward pass."""
    model = M17TrackADetector(res_dim=128, num_families=5)
    model.freeze_stages_0_to_4()

    x = torch.randn(2, 16, 3, 112, 112)
    out: M17TrackAOutput = model(x)

    assert out.primary_logits.shape == (2, 1)
    assert out.res_logits.shape == (2, 1)
    assert out.aux_family_logits.shape == (2, 5)
    assert out.p_fused.shape == (2, 1)
    assert out.p_composite.shape == (2, 1)
    assert out.gate_values.shape == (2, model.temporal_dim)

    loss_fn = M17TrackALoss()
    binary_targets = torch.tensor([1.0, 0.0])
    family_targets = torch.tensor([1, 0])
    loss_dict = loss_fn(out, binary_targets, family_targets)

    assert "loss" in loss_dict
    loss_dict["loss"].backward()


# ─────────────────────────────────────────────────────────────────────────────
# TRACK B TESTS: FACIAL DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def test_anatomical_region_extractor():
    """Verify AnatomicalRegionExtractor partitions spatial feature maps into K=4 regional tokens."""
    extractor = AnatomicalRegionExtractor(in_channels=128, embed_dim=128, num_regions=4)
    feat_maps = torch.randn(32, 128, 7, 7)  # (BT, C, H, W)
    regional_tokens = extractor(feat_maps)

    assert regional_tokens.shape == (32, 4, 128)


def test_spatiotemporal_region_transformer():
    """Verify SpatiotemporalRegionTransformer processes 64 tokens and computes correlation matrix."""
    transformer = SpatiotemporalRegionTransformer(
        embed_dim=128,
        num_frames=16,
        num_regions=4,
        out_features=128,
    )
    tokens = torch.randn(32, 4, 128)  # (B*T, K, D) with B=2, T=16
    z_dyn, corr_matrix = transformer(tokens, B=2, T=16)

    assert z_dyn.shape == (2, 128)
    assert corr_matrix.shape == (2, 4, 4)
    # Check diagonal of correlation matrix is approximately 1.0
    for b in range(2):
        for k in range(4):
            assert abs(corr_matrix[b, k, k].item() - 1.0) < 1e-2


def test_track_b_detector_forward_backward():
    """Verify M17TrackBDetector forward hook, correlation outputs, and backward pass."""
    model = M17TrackBDetector(dyn_dim=128, num_families=5)
    model.freeze_stages_0_to_4()

    x = torch.randn(2, 16, 3, 112, 112)
    out: M17TrackBOutput = model(x)

    assert out.primary_logits.shape == (2, 1)
    assert out.dyn_logits.shape == (2, 1)
    assert out.aux_family_logits.shape == (2, 5)
    assert out.p_fused.shape == (2, 1)
    assert out.p_composite.shape == (2, 1)
    assert out.corr_matrix.shape == (2, 4, 4)

    loss_fn = M17TrackBLoss()
    binary_targets = torch.tensor([1.0, 0.0])
    family_targets = torch.tensor([2, 0])
    loss_dict = loss_fn(out, binary_targets, family_targets)

    assert "loss" in loss_dict
    loss_dict["loss"].backward()


# ─────────────────────────────────────────────────────────────────────────────
# STRESS TESTING & CONTROL INVARIANT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_simulated_jpeg_stress():
    """Verify apply_simulated_jpeg applies JPEG compression without altering tensor dimensions."""
    x = torch.randn(16, 3, 112, 112)
    compressed = apply_simulated_jpeg(x, quality=50)

    assert compressed.shape == (16, 3, 112, 112)
    # Verify values changed due to lossy JPEG DCT quantization
    assert not torch.equal(x, compressed)


def test_immutable_control_checkpoints():
    """Verify M12 baseline and v5 champion checkpoints are 100% bitwise immutable."""
    m12_path = Path("checkpoints/m12_full/best.pth")
    v5_path = Path("checkpoints/m13_3_f2f_seed42/best.pth")

    assert m12_path.exists(), "M12 checkpoint missing!"
    assert v5_path.exists(), "v5 champion checkpoint missing!"

    def hash_file(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    expected_m12 = "c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56"
    expected_v5 = "4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7"

    assert hash_file(m12_path) == expected_m12, "CRITICAL: M12 baseline checkpoint modified!"
    assert hash_file(v5_path) == expected_v5, "CRITICAL: v5 champion checkpoint modified!"


def test_active_scales_inference_ablation():
    """Verify that M17TrackADetector active_scales allows masking individual residual scales."""
    model = M17TrackADetector()
    model.eval()

    x = torch.randn(2, 16, 3, 112, 112)
    with torch.no_grad():
        out_full = model(x, active_scales=(True, True, True))
        out_s1 = model(x, active_scales=(True, False, False))
        out_s2 = model(x, active_scales=(False, True, False))
        out_s3 = model(x, active_scales=(False, False, True))

    assert out_full.primary_logits.shape == (2, 1)
    assert out_s1.primary_logits.shape == (2, 1)
    assert not torch.allclose(out_full.res_features, out_s1.res_features)
    assert not torch.allclose(out_s1.res_features, out_s2.res_features)
    assert not torch.allclose(out_s2.res_features, out_s3.res_features)

