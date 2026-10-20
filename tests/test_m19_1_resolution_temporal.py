"""Unit tests for Milestone 19.1 Resolution & Temporal Attribution."""

import numpy as np
import pytest
import torch

from scripts.diagnose_m19_1_resolution_temporal import (
    compute_cliffs_delta,
    compute_paired_bootstrap,
    modulate_spatial_resolution,
    modulate_temporal_sampling,
)


def test_modulate_temporal_sampling():
    x = torch.randn(2, 16, 3, 112, 112)
    
    # Target T = 8
    x_8 = modulate_temporal_sampling(x, target_t=8)
    assert x_8.shape == (2, 8, 3, 112, 112)
    
    # Target T = 16 (identity)
    x_16 = modulate_temporal_sampling(x, target_t=16)
    assert x_16.shape == (2, 16, 3, 112, 112)
    assert torch.equal(x, x_16)
    
    # Target T = 32
    x_32 = modulate_temporal_sampling(x, target_t=32)
    assert x_32.shape == (2, 32, 3, 112, 112)


def test_modulate_spatial_resolution():
    x = torch.randn(2, 16, 3, 112, 112)
    
    # Target R = 64
    x_64 = modulate_spatial_resolution(x, target_r=64)
    assert x_64.shape == (2, 16, 3, 112, 112)
    
    # Target R = 112 (identity)
    x_112 = modulate_spatial_resolution(x, target_r=112)
    assert x_112.shape == (2, 16, 3, 112, 112)
    assert torch.equal(x, x_112)
    
    # Target R = 224
    x_224 = modulate_spatial_resolution(x, target_r=224)
    assert x_224.shape == (2, 16, 3, 112, 112)


def test_compute_paired_bootstrap():
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 50 + [1] * 50)
    scores_ref = rng.normal(loc=0.0, scale=1.0, size=100) + y_true * 0.5
    scores_cand = scores_ref + rng.normal(loc=0.1, scale=0.2, size=100)
    
    boot = compute_paired_bootstrap(y_true, scores_cand, scores_ref, n_resamples=100, seed=42)
    assert "delta_mean" in boot
    assert "ci_lower" in boot
    assert "ci_upper" in boot
    assert "p_value" in boot
    assert 0.0 <= boot["p_value"] <= 1.0
    assert boot["ci_lower"] <= boot["ci_upper"]


def test_compute_cliffs_delta():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 1, 2, 3, 4])
    cd = compute_cliffs_delta(x, y)
    assert -1.0 <= cd <= 1.0
    assert cd > 0.0
