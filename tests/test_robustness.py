"""Unit tests for video robustness perturbations."""

import pytest
import torch

from src.evaluation.robustness import (
    PerturbationConfig,
    apply_additive_noise,
    apply_contrast_reduction,
    apply_gaussian_blur,
    apply_jpeg_compression_simulation,
    apply_spatial_downsampling,
)


def test_spatial_downsampling():
    """Spatial downsampling must preserve (T, C, H, W) shape and alter high-frequency content."""
    x = torch.randn(16, 3, 112, 112)
    # 56x56
    out_56 = apply_spatial_downsampling(x, low_res_size=(56, 56), final_size=(112, 112))
    assert out_56.shape == x.shape
    assert not torch.allclose(out_56, x)

    # 28x28
    out_28 = apply_spatial_downsampling(x, low_res_size=(28, 28), final_size=(112, 112))
    assert out_28.shape == x.shape
    assert not torch.allclose(out_28, out_56)


def test_gaussian_blur():
    """Gaussian blur must preserve shape and handle varying sigmas."""
    x = torch.randn(16, 3, 112, 112)
    for sigma in [0.5, 1.0, 1.5]:
        blurred = apply_gaussian_blur(x, sigma=sigma)
        assert blurred.shape == x.shape
        assert not torch.allclose(blurred, x)


def test_additive_noise():
    """Additive Gaussian noise must be reproducible with manual seed."""
    x = torch.zeros(16, 3, 112, 112)
    noisy1 = apply_additive_noise(x, std=0.05, seed=42)
    noisy2 = apply_additive_noise(x, std=0.05, seed=42)
    noisy3 = apply_additive_noise(x, std=0.05, seed=99)

    assert noisy1.shape == x.shape
    assert torch.allclose(noisy1, noisy2)
    assert not torch.allclose(noisy1, noisy3)


def test_contrast_reduction():
    """Contrast reduction must reduce variance around per-frame channel mean."""
    x = torch.randn(16, 3, 112, 112)
    reduced = apply_contrast_reduction(x, factor=0.5)

    assert reduced.shape == x.shape
    assert torch.std(reduced) < torch.std(x)


def test_jpeg_compression_simulation():
    """JPEG compression simulation must preserve (16, 3, 112, 112) shape."""
    x = torch.randn(16, 3, 112, 112)
    compressed = apply_jpeg_compression_simulation(x, quality=30)
    assert compressed.shape == x.shape
    assert not torch.isnan(compressed).any()
    assert not torch.isinf(compressed).any()


def test_perturbation_config():
    """PerturbationConfig serialization test."""
    cfg = PerturbationConfig(
        name="blur_1.0",
        category="photometric",
        parameter_name="sigma",
        parameter_value=1.0,
        description="Gaussian blur with sigma=1.0",
    )
    d = cfg.to_dict()
    assert d["name"] == "blur_1.0"
    assert d["category"] == "photometric"
    assert d["parameter_value"] == 1.0
