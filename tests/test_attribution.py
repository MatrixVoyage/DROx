"""Unit tests for spatial attention attribution utilities."""

import numpy as np
import pytest
import torch

from src.evaluation.attribution import (
    FrameAttentionStats,
    SequenceAttentionSummary,
    compute_sequence_attention_stats,
    create_attention_overlay,
    create_attribution_panel,
    extract_attention_weights,
    interpolate_attention_heatmap,
)
from src.models.detector import DeepfakeDetector
from src.utils.config import ModelConfig


def test_attention_shape_and_sum():
    """Attention extraction must return (T, 4, 4) weights summing to 1.0 per frame."""
    config = ModelConfig()
    model = DeepfakeDetector(config=config)
    model.eval()

    tensor = torch.randn(1, 16, 3, 112, 112)
    weights = extract_attention_weights(model, tensor)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (16, 4, 4)
    # Check that spatial softmax sums to 1.0 per frame
    for t in range(16):
        frame_sum = weights[t].sum().item()
        assert pytest.approx(frame_sum, abs=1e-5) == 1.0


def test_compute_sequence_attention_stats():
    """Sequence stats must calculate valid Shannon entropy and top-4 mass."""
    # Create uniform attention across 4x4: each cell = 1/16 = 0.0625
    uniform_attn = np.full((16, 4, 4), 1.0 / 16.0, dtype=np.float32)
    summary = compute_sequence_attention_stats(uniform_attn)

    assert isinstance(summary, SequenceAttentionSummary)
    assert summary.sequence_length == 16
    # For uniform distribution over 16 cells, entropy = ln(16) ~ 2.7726 nats
    assert pytest.approx(summary.mean_entropy, abs=1e-2) == np.log(16)
    assert pytest.approx(summary.mean_max_weight, abs=1e-3) == 0.0625
    assert pytest.approx(summary.mean_top4_mass, abs=1e-3) == 0.25


def test_interpolate_attention_heatmap():
    """Heatmap interpolation must upsample to (16, 112, 112) with values in [0, 1]."""
    attn = np.random.uniform(0.01, 0.2, size=(16, 4, 4)).astype(np.float32)
    # Normalize per frame so it sums to 1
    attn = attn / attn.sum(axis=(-1, -2), keepdims=True)

    heatmaps = interpolate_attention_heatmap(attn, target_size=(112, 112))
    assert heatmaps.shape == (16, 112, 112)
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0


def test_attention_overlay_and_panel():
    """Overlay and panel generator must produce valid uint8 image dimensions."""
    frame_rgb = np.zeros((112, 112, 3), dtype=np.uint8)
    heatmap = np.ones((112, 112), dtype=np.float32)

    overlay = create_attention_overlay(frame_rgb, heatmap, alpha=0.5)
    assert overlay.shape == (112, 112, 3)
    assert overlay.dtype == np.uint8

    panel = create_attribution_panel(frame_rgb, heatmap, overlay)
    assert panel.shape == (112, 336, 3)  # 112 x (112 * 3)
    assert panel.dtype == np.uint8
