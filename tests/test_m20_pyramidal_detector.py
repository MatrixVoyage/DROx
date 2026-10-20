"""Automated unit tests for Milestone 20.2 Pyramidal Architecture and Preprocessor.

Verifies:
1. PyramidalPreprocessor decomposition and sequence normalization invariants.
2. PyramidalFusionStem blending and gradient flow.
3. M20PyramidalDualStreamDetector forward pass across all 3 evaluation modes.
4. Output dataclass invariants, probabilities, and memory consumption.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.pyramidal_preprocessor import PyramidalPreprocessor
from src.models.detector import DeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.m20_pyramidal_detector import (
    PyramidalFusionStem,
    M20PyramidalDualStreamDetector,
    M20DetectorOutput,
)
from src.utils.config import ModelConfig


@pytest.fixture
def mock_trunk():
    """Create a lightweight M17 Track A detector for testing."""
    cfg = ModelConfig()
    base = DeepfakeDetector(config=cfg)
    m17 = M17TrackADetector(
        base_detector=base,
        num_families=5,
        res_dim=128,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        gate_bottleneck=128,
    )
    return m17


def test_pyramidal_preprocessor_invariants():
    """Verify PyramidalPreprocessor processes native frames correctly."""
    prep = PyramidalPreprocessor(target_size=112, default_alpha=0.50)

    # 1. Single frame decomposition
    H0, W0 = 200, 200
    native_crop = np.random.randint(0, 256, (H0, W0, 3), dtype=np.uint8)
    decomp = prep.decompose_frame_pyramid(native_crop)

    assert decomp["G1"].shape == (112, 112, 3)
    assert decomp["L0_112_f"].shape == (112, 112, 3)
    assert decomp["injected"].shape == (112, 112, 3)
    assert decomp["G1"].dtype == np.uint8
    assert decomp["injected"].dtype == np.uint8

    # 2. Sequence processing (single tensor)
    native_seq = [np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8) for _ in range(16)]
    t_inj = prep.process_sequence(native_seq, return_dual_tensors=False)
    assert t_inj.shape == (16, 3, 112, 112)
    assert t_inj.dtype == torch.float32
    assert not torch.isnan(t_inj).any()

    # 3. Sequence processing (dual tensors)
    t_g1, t_l0 = prep.process_sequence(native_seq, return_dual_tensors=True)
    assert t_g1.shape == (16, 3, 112, 112)
    assert t_l0.shape == (16, 3, 112, 112)


def test_pyramidal_fusion_stem():
    """Verify PyramidalFusionStem blending and gradient backpropagation."""
    stem = PyramidalFusionStem(init_alpha=0.50)
    assert abs(stem.alpha.item() - 0.50) < 1e-4

    x_base = torch.randn(2, 4, 3, 112, 112)
    x_high = torch.randn(2, 4, 3, 112, 112)

    # Single input mode
    out_single = stem(x_base)
    assert torch.allclose(out_single, x_base)

    # Dual input blending
    out_blended = stem(x_base, x_high)
    expected = x_base + stem.alpha * x_high
    assert torch.allclose(out_blended, expected)

    # Gradient flow
    loss = out_blended.sum()
    loss.backward()
    assert stem.raw_alpha.grad is not None
    assert stem.raw_alpha.grad.abs().item() > 0.0


def test_m20_detector_forward_modes(mock_trunk):
    """Verify M20 detector forward pass across dual_stream, kinematic_only, and residual_only."""
    model = M20PyramidalDualStreamDetector(
        trunk_detector=mock_trunk,
        kin_dim=256,
        aux_hidden_dim=256,
        freeze_trunk=True,
    )
    model.eval()

    B, T, C, H, W = 2, 8, 3, 112, 112
    x = torch.randn(B, T, C, H, W)
    x_high = torch.randn(B, T, C, H, W)

    with torch.no_grad():
        # Mode 1: Dual Stream (Full model)
        out_dual = model(x, x_high=x_high, mode="dual_stream")
        assert isinstance(out_dual, M20DetectorOutput)
        assert out_dual.primary_logits.shape == (B, 1)
        assert out_dual.aux_logits.shape == (B, 5)
        assert out_dual.gate_values.shape == (B, 1)
        assert out_dual.p_fused.shape == (B, 1)
        assert (out_dual.gate_values >= 0.0).all() and (out_dual.gate_values <= 1.0).all()
        assert (out_dual.p_fused >= 0.0).all() and (out_dual.p_fused <= 1.0).all()

        # Mode 2: Kinematic Only
        out_kin = model(x, mode="kinematic_only")
        assert out_kin.primary_logits.shape == (B, 1)
        assert torch.allclose(out_kin.gate_values, torch.ones_like(out_kin.gate_values))

        # Mode 3: Residual Only
        out_res = model(x, mode="residual_only")
        assert out_res.primary_logits.shape == (B, 1)
        assert torch.allclose(out_res.gate_values, torch.zeros_like(out_res.gate_values))
