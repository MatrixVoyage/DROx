"""Unit tests for RobustAugmentor (M12.1).

Validates:
- Tensor shape/dtype/NaN invariants after augmentation.
- Sequence consistency (same degradation params across all T frames).
- Severity distribution sampling proportions.
- Individual degradation function correctness.
- Re-encoding pipeline output integrity.
- Temporal frame dropping behavior.
- Config enabled/disabled toggle.
- Reproducibility with fixed seed.
"""

import numpy as np
import pytest
import torch

from src.data.augment import (
    RobustAugmentor,
    RobustAugmentorConfig,
    SequenceDegradationParams,
    SeverityDistribution,
    SeverityTier,
    apply_downsample_upsample,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_reencode_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a synthetic (T=16, C=3, H=112, W=112) float32 tensor in [0,1]."""
    torch.manual_seed(42)
    return torch.rand(16, 3, 112, 112, dtype=torch.float32)


@pytest.fixture
def sample_crops() -> list:
    """Create a list of 16 synthetic 112x112x3 uint8 ndarrays."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(16)]


@pytest.fixture
def default_augmentor() -> RobustAugmentor:
    return RobustAugmentor(seed=42)


@pytest.fixture
def disabled_augmentor() -> RobustAugmentor:
    config = RobustAugmentorConfig(enabled=False)
    return RobustAugmentor(config=config, seed=42)


@pytest.fixture
def aggressive_augmentor() -> RobustAugmentor:
    """Augmentor with all probabilities set to 1.0 for deterministic testing."""
    config = RobustAugmentorConfig(
        enabled=True,
        jpeg_p=1.0,
        blur_p=1.0,
        noise_p=1.0,
        resize_p=1.0,
        reencode_p=1.0,
        frame_drop_p=1.0,
        max_drop_frames=2,
    )
    return RobustAugmentor(config=config, seed=123)


# ---------------------------------------------------------------------------
# Severity distribution tests
# ---------------------------------------------------------------------------

class TestSeverityDistribution:
    def test_default_sums_to_one(self):
        dist = SeverityDistribution()
        assert abs(dist.mild + dist.moderate + dist.severe - 1.0) < 1e-6

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            SeverityDistribution(mild=0.5, moderate=0.5, severe=0.5)

    def test_sampling_proportions(self):
        """Verify that over many samples the distribution roughly matches."""
        dist = SeverityDistribution(mild=0.75, moderate=0.20, severe=0.05)
        rng = np.random.RandomState(99)
        counts = {SeverityTier.MILD: 0, SeverityTier.MODERATE: 0, SeverityTier.SEVERE: 0}
        n = 10000
        for _ in range(n):
            tier = dist.sample(rng)
            counts[tier] += 1

        assert counts[SeverityTier.MILD] / n == pytest.approx(0.75, abs=0.03)
        assert counts[SeverityTier.MODERATE] / n == pytest.approx(0.20, abs=0.03)
        assert counts[SeverityTier.SEVERE] / n == pytest.approx(0.05, abs=0.02)


# ---------------------------------------------------------------------------
# Individual degradation function tests
# ---------------------------------------------------------------------------

class TestDegradationFunctions:
    def test_jpeg_preserves_shape_and_dtype(self):
        frame = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = apply_jpeg_compression(frame, quality=50)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_jpeg_quality_1_degrades_heavily(self):
        """Very low quality should produce visible artifacts (different pixel values)."""
        frame = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        result = apply_jpeg_compression(frame, quality=1)
        assert result.shape == frame.shape
        # With quality=1, the result should be substantially different
        diff = np.abs(frame.astype(float) - result.astype(float)).mean()
        assert diff > 1.0  # at least some degradation

    def test_blur_preserves_shape(self):
        frame = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = apply_gaussian_blur(frame, kernel_size=5)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_noise_preserves_shape_and_range(self):
        rng = np.random.RandomState(10)
        frame = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = apply_gaussian_noise(frame, std=10.0, rng=rng)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_downsample_upsample_preserves_shape(self):
        frame = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = apply_downsample_upsample(frame, scale_factor=0.5)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_reencode_preserves_shape(self):
        frame = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = apply_reencode_simulation(frame, quality=50, scale_factor=0.5)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# RobustAugmentor tensor-level tests
# ---------------------------------------------------------------------------

class TestRobustAugmentorTensor:
    def test_shape_preserved(self, sample_tensor, default_augmentor):
        """Output tensor must have same shape as input."""
        result, params = default_augmentor.augment_tensor(sample_tensor)
        assert result.shape == sample_tensor.shape

    def test_dtype_preserved(self, sample_tensor, default_augmentor):
        """Output must remain float32."""
        result, _ = default_augmentor.augment_tensor(sample_tensor)
        assert result.dtype == torch.float32

    def test_no_nan_or_inf(self, sample_tensor, aggressive_augmentor):
        """Even with aggressive augmentation, no NaN/Inf should appear."""
        result, _ = aggressive_augmentor.augment_tensor(sample_tensor)
        assert not torch.isnan(result).any(), "NaN detected in augmented tensor"
        assert not torch.isinf(result).any(), "Inf detected in augmented tensor"

    def test_output_range(self, sample_tensor, default_augmentor):
        """After augmentation, values should be in approximately [0, 1]."""
        result, _ = default_augmentor.augment_tensor(sample_tensor)
        assert result.min() >= -0.01  # small tolerance for rounding
        assert result.max() <= 1.01

    def test_disabled_returns_identity(self, sample_tensor, disabled_augmentor):
        """When disabled, the tensor should be returned unchanged."""
        result, _ = disabled_augmentor.augment_tensor(sample_tensor)
        assert torch.equal(result, sample_tensor)

    def test_aggressive_produces_change(self, sample_tensor, aggressive_augmentor):
        """With all augmentations active, the output must differ from input."""
        result, _ = aggressive_augmentor.augment_tensor(sample_tensor)
        assert not torch.equal(result, sample_tensor)

    def test_reproducibility_with_seed(self, sample_tensor):
        """Same seed produces identical augmentation."""
        aug1 = RobustAugmentor(seed=999)
        aug2 = RobustAugmentor(seed=999)

        r1, p1 = aug1.augment_tensor(sample_tensor.clone())
        r2, p2 = aug2.augment_tensor(sample_tensor.clone())

        assert torch.allclose(r1, r2, atol=1e-6)
        assert p1.to_dict() == p2.to_dict()

    def test_params_returned_match_applied(self, sample_tensor, aggressive_augmentor):
        """Returned params should reflect what was applied."""
        _, params = aggressive_augmentor.augment_tensor(sample_tensor)
        # With all probabilities = 1.0, all degradations should be active
        assert params.do_jpeg is True
        assert params.do_blur is True
        assert params.do_noise is True
        assert params.do_resize is True
        assert params.do_reencode is True

    def test_params_serialization(self, sample_tensor, default_augmentor):
        """Params should serialize to a dict without error."""
        _, params = default_augmentor.augment_tensor(sample_tensor)
        d = params.to_dict()
        assert isinstance(d, dict)
        assert "severity" in d
        assert d["severity"] in ("mild", "moderate", "severe")


# ---------------------------------------------------------------------------
# RobustAugmentor crops-level tests
# ---------------------------------------------------------------------------

class TestRobustAugmentorCrops:
    def test_crops_shape_preserved(self, sample_crops, default_augmentor):
        result, _ = default_augmentor.augment_crops(sample_crops)
        assert len(result) == len(sample_crops)
        for i, crop in enumerate(result):
            assert crop.shape == sample_crops[i].shape
            assert crop.dtype == np.uint8

    def test_crops_no_nan(self, sample_crops, aggressive_augmentor):
        result, _ = aggressive_augmentor.augment_crops(sample_crops)
        for crop in result:
            assert not np.isnan(crop).any()

    def test_crops_disabled_identity(self, sample_crops, disabled_augmentor):
        result, _ = disabled_augmentor.augment_crops(sample_crops)
        for i in range(len(result)):
            np.testing.assert_array_equal(result[i], sample_crops[i])


# ---------------------------------------------------------------------------
# Sequence consistency tests
# ---------------------------------------------------------------------------

class TestSequenceConsistency:
    """Verify that the SAME degradation parameters are used for all frames."""

    def test_jpeg_quality_consistent_across_frames(self):
        """Apply JPEG-only augmentation and verify quality is uniform."""
        config = RobustAugmentorConfig(
            enabled=True,
            jpeg_p=1.0,
            blur_p=0.0,
            noise_p=0.0,
            resize_p=0.0,
            reencode_p=0.0,
            frame_drop_p=0.0,
        )
        aug = RobustAugmentor(config=config, seed=42)

        # Create a uniform sequence (all frames identical)
        base_frame = np.full((112, 112, 3), 128, dtype=np.uint8)
        crops = [base_frame.copy() for _ in range(16)]

        result, params = aug.augment_crops(crops)

        # All frames received the same JPEG quality, so identical input →
        # identical degraded output (JPEG is deterministic for identical input)
        for i in range(1, len(result)):
            np.testing.assert_array_equal(
                result[0], result[i],
                err_msg=f"Frame 0 and frame {i} differ after uniform JPEG augmentation"
            )

    def test_blur_consistent_across_frames(self):
        """All frames should receive the same blur kernel."""
        config = RobustAugmentorConfig(
            enabled=True,
            jpeg_p=0.0,
            blur_p=1.0,
            noise_p=0.0,
            resize_p=0.0,
            reencode_p=0.0,
            frame_drop_p=0.0,
        )
        aug = RobustAugmentor(config=config, seed=42)

        base_frame = np.full((112, 112, 3), 128, dtype=np.uint8)
        crops = [base_frame.copy() for _ in range(16)]

        result, _ = aug.augment_crops(crops)

        for i in range(1, len(result)):
            np.testing.assert_array_equal(
                result[0], result[i],
                err_msg=f"Frame 0 and frame {i} differ after uniform blur augmentation"
            )


# ---------------------------------------------------------------------------
# Temporal perturbation tests
# ---------------------------------------------------------------------------

class TestTemporalPerturbations:
    def test_frame_drop_duplicates_neighbor(self, sample_tensor):
        """Dropped frames should be replaced by their neighbor."""
        config = RobustAugmentorConfig(
            enabled=True,
            jpeg_p=0.0,
            blur_p=0.0,
            noise_p=0.0,
            resize_p=0.0,
            reencode_p=0.0,
            frame_drop_p=1.0,
            max_drop_frames=2,
        )
        aug = RobustAugmentor(config=config, seed=42)
        result, params = aug.augment_tensor(sample_tensor.clone())

        assert params.do_frame_drop is True
        assert len(params.drop_indices) > 0

        # Dropped frames should be identical to their source neighbor
        for drop_idx in params.drop_indices:
            src_idx = max(0, drop_idx - 1)
            if src_idx == drop_idx:
                src_idx = min(sample_tensor.shape[0] - 1, drop_idx + 1)
            assert torch.equal(result[drop_idx], result[src_idx])

    def test_frame_drop_with_short_sequence(self):
        """Frame drop on a 2-frame sequence should not crash."""
        config = RobustAugmentorConfig(
            enabled=True, jpeg_p=0.0, blur_p=0.0, noise_p=0.0,
            resize_p=0.0, reencode_p=0.0,
            frame_drop_p=1.0, max_drop_frames=1,
        )
        aug = RobustAugmentor(config=config, seed=42)
        short = torch.rand(3, 3, 112, 112)
        result, params = aug.augment_tensor(short)
        assert result.shape == short.shape


# ---------------------------------------------------------------------------
# Pre-sampled params test
# ---------------------------------------------------------------------------

class TestPreSampledParams:
    def test_explicit_params_are_honored(self, sample_tensor):
        """Passing explicit params should override internal sampling."""
        aug = RobustAugmentor(seed=42)
        explicit = SequenceDegradationParams(
            severity=SeverityTier.SEVERE,
            do_jpeg=True,
            jpeg_quality=10,
            do_blur=False,
            do_noise=False,
            do_resize=False,
            do_reencode=False,
            do_frame_drop=False,
        )
        result, returned_params = aug.augment_tensor(sample_tensor, params=explicit)
        assert returned_params.jpeg_quality == 10
        assert returned_params.do_blur is False
        assert result.shape == sample_tensor.shape
