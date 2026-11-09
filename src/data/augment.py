"""Robust sequence-consistent augmentations for degradation robustness training.

M12.1 — All degradation parameters are sampled ONCE per sequence and applied
identically to every frame, preventing the model from learning
"frame-to-frame augmentation flicker" as fake.

Severity distribution follows a configurable mild/moderate/severe ratio
(default 75%/20%/5%) rather than uniform extreme degradation.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Severity tiers
# ---------------------------------------------------------------------------

class SeverityTier(Enum):
    """Degradation severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class SeverityDistribution:
    """Probability mass over severity tiers. Must sum to 1.0."""
    mild: float = 0.75
    moderate: float = 0.20
    severe: float = 0.05

    def __post_init__(self) -> None:
        total = self.mild + self.moderate + self.severe
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Severity probabilities must sum to 1.0, got {total:.6f}"
            )

    def sample(self, rng: np.random.RandomState) -> SeverityTier:
        """Sample a severity tier."""
        r = rng.random()
        if r < self.mild:
            return SeverityTier.MILD
        elif r < self.mild + self.moderate:
            return SeverityTier.MODERATE
        else:
            return SeverityTier.SEVERE


# ---------------------------------------------------------------------------
# Per-degradation parameter ranges indexed by severity
# ---------------------------------------------------------------------------

# JPEG quality ranges: (min_quality, max_quality)
JPEG_QUALITY_RANGES: Dict[SeverityTier, Tuple[int, int]] = {
    SeverityTier.MILD: (75, 95),
    SeverityTier.MODERATE: (40, 74),
    SeverityTier.SEVERE: (10, 39),
}

# Gaussian blur kernel size (always odd): (min_k, max_k)
BLUR_KERNEL_RANGES: Dict[SeverityTier, Tuple[int, int]] = {
    SeverityTier.MILD: (3, 3),
    SeverityTier.MODERATE: (3, 5),
    SeverityTier.SEVERE: (5, 7),
}

# Gaussian noise standard deviation (applied to [0,255] uint8 scale)
NOISE_STD_RANGES: Dict[SeverityTier, Tuple[float, float]] = {
    SeverityTier.MILD: (1.0, 5.0),
    SeverityTier.MODERATE: (5.0, 15.0),
    SeverityTier.SEVERE: (15.0, 30.0),
}

# Resize scale factor: frame is downscaled then upscaled back
RESIZE_SCALE_RANGES: Dict[SeverityTier, Tuple[float, float]] = {
    SeverityTier.MILD: (0.75, 0.95),
    SeverityTier.MODERATE: (0.50, 0.74),
    SeverityTier.SEVERE: (0.25, 0.49),
}


# ---------------------------------------------------------------------------
# Individual degradation functions (operate on single uint8 HWC frame)
# ---------------------------------------------------------------------------

def apply_jpeg_compression(frame: np.ndarray, quality: int) -> np.ndarray:
    """Simulate JPEG compression artifacts via encode/decode cycle.

    Args:
        frame: HxWx3 uint8 BGR or RGB ndarray.
        quality: JPEG quality factor [1, 100].

    Returns:
        Degraded HxWx3 uint8 ndarray of same shape.
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        return frame  # fallback: return original
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != frame.shape:
        return frame
    return decoded


def apply_gaussian_blur(frame: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur with the given kernel size.

    Args:
        frame: HxWx3 uint8 ndarray.
        kernel_size: Odd positive integer.

    Returns:
        Blurred frame.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def apply_gaussian_noise(
    frame: np.ndarray, std: float, rng: np.random.RandomState
) -> np.ndarray:
    """Add Gaussian noise to a frame.

    Args:
        frame: HxWx3 uint8 ndarray.
        std: Noise standard deviation on [0,255] scale.
        rng: RandomState for reproducibility.

    Returns:
        Noisy frame clipped to [0, 255].
    """
    noise = rng.normal(0, std, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_downsample_upsample(
    frame: np.ndarray, scale_factor: float
) -> np.ndarray:
    """Simulate resolution loss by downscaling then upscaling back.

    Args:
        frame: HxWx3 uint8 ndarray.
        scale_factor: Float in (0, 1); fraction of original resolution.

    Returns:
        Degraded frame at original resolution.
    """
    h, w = frame.shape[:2]
    new_h = max(4, int(h * scale_factor))
    new_w = max(4, int(w * scale_factor))
    down = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    return up


def apply_reencode_simulation(
    frame: np.ndarray, quality: int, scale_factor: float
) -> np.ndarray:
    """Simulate a social-media-style re-encoding pipeline.

    Pipeline: JPEG encode → decode → downscale → upscale → JPEG encode → decode.

    Args:
        frame: HxWx3 uint8 ndarray.
        quality: JPEG quality for both encode passes.
        scale_factor: Intermediate downscale factor.

    Returns:
        Re-encoded frame.
    """
    # First JPEG pass
    frame = apply_jpeg_compression(frame, quality)
    # Resize pass
    frame = apply_downsample_upsample(frame, scale_factor)
    # Second JPEG pass (slightly better quality to simulate re-upload)
    second_quality = min(100, quality + 10)
    frame = apply_jpeg_compression(frame, second_quality)
    return frame


# ---------------------------------------------------------------------------
# Sequence-level augmentation configuration (sampled once per sequence)
# ---------------------------------------------------------------------------

@dataclass
class SequenceDegradationParams:
    """Parameters sampled once and applied identically to all T frames."""
    severity: SeverityTier

    # Which degradations to apply (determined probabilistically)
    do_jpeg: bool = False
    jpeg_quality: int = 95

    do_blur: bool = False
    blur_kernel: int = 3

    do_noise: bool = False
    noise_std: float = 0.0
    # We store the full noise array per-frame to ensure reproducibility,
    # but the std is sampled once.

    do_resize: bool = False
    resize_scale: float = 1.0

    do_reencode: bool = False
    reencode_quality: int = 80
    reencode_scale: float = 0.75

    # Temporal perturbations
    do_frame_drop: bool = False
    drop_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize for logging/audit."""
        return {
            "severity": self.severity.value,
            "do_jpeg": self.do_jpeg,
            "jpeg_quality": self.jpeg_quality,
            "do_blur": self.do_blur,
            "blur_kernel": self.blur_kernel,
            "do_noise": self.do_noise,
            "noise_std": self.noise_std,
            "do_resize": self.do_resize,
            "resize_scale": self.resize_scale,
            "do_reencode": self.do_reencode,
            "reencode_quality": self.reencode_quality,
            "reencode_scale": self.reencode_scale,
            "do_frame_drop": self.do_frame_drop,
            "drop_indices": self.drop_indices,
        }


# ---------------------------------------------------------------------------
# RobustAugmentor
# ---------------------------------------------------------------------------

@dataclass
class RobustAugmentorConfig:
    """Configuration for the RobustAugmentor.

    Attributes:
        enabled: Master switch; if False, no augmentation is applied.
        jpeg_p: Probability of applying JPEG compression per sequence.
        blur_p: Probability of applying Gaussian blur per sequence.
        noise_p: Probability of applying Gaussian noise per sequence.
        resize_p: Probability of applying downsample/upsample per sequence.
        reencode_p: Probability of applying re-encoding simulation per sequence.
        frame_drop_p: Probability of temporal frame dropping per sequence.
        max_drop_frames: Maximum number of frames to drop (replaced by duplication).
        severity_distribution: Probability mass over mild/moderate/severe.
    """
    enabled: bool = True
    jpeg_p: float = 0.4
    blur_p: float = 0.3
    noise_p: float = 0.3
    resize_p: float = 0.2
    reencode_p: float = 0.15
    frame_drop_p: float = 0.1
    max_drop_frames: int = 2
    severity_distribution: SeverityDistribution = field(
        default_factory=SeverityDistribution
    )


class RobustAugmentor:
    """Sequence-consistent robust augmentor for training degradation invariance.

    CRITICAL INVARIANT: All spatial degradation parameters (JPEG quality,
    blur kernel, noise std, resize factor) are sampled ONCE per sequence
    and applied identically across all T frames.

    Temporal perturbations (frame dropping) are applied separately and
    operate on the frame index dimension.

    This augmentor operates on tensors of shape (T, C, H, W) in [0, 1]
    float32 range (pre-normalization), or on lists of uint8 HWC ndarrays.
    """

    def __init__(
        self,
        config: Optional[RobustAugmentorConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or RobustAugmentorConfig()
        self.seed = seed
        self._call_count = 0

    def _get_rng(self) -> np.random.RandomState:
        """Create a per-call RNG for reproducibility."""
        if self.seed is not None:
            return np.random.RandomState(self.seed + self._call_count)
        return np.random.RandomState()

    def sample_params(
        self, num_frames: int, rng: Optional[np.random.RandomState] = None
    ) -> SequenceDegradationParams:
        """Sample degradation parameters for a sequence.

        Args:
            num_frames: Number of frames T in the sequence.
            rng: Optional RNG; if None, one is created.

        Returns:
            SequenceDegradationParams with all decisions made.
        """
        if rng is None:
            rng = self._get_rng()

        severity = self.config.severity_distribution.sample(rng)

        params = SequenceDegradationParams(severity=severity)

        # JPEG
        if rng.random() < self.config.jpeg_p:
            params.do_jpeg = True
            lo, hi = JPEG_QUALITY_RANGES[severity]
            params.jpeg_quality = int(rng.randint(lo, hi + 1))

        # Blur
        if rng.random() < self.config.blur_p:
            params.do_blur = True
            lo, hi = BLUR_KERNEL_RANGES[severity]
            # Only odd values
            candidates = list(range(lo, hi + 1, 2))
            if not candidates:
                candidates = [lo if lo % 2 == 1 else lo + 1]
            params.blur_kernel = int(rng.choice(candidates))

        # Noise
        if rng.random() < self.config.noise_p:
            params.do_noise = True
            lo, hi = NOISE_STD_RANGES[severity]
            params.noise_std = float(lo + (hi - lo) * rng.random())

        # Resize
        if rng.random() < self.config.resize_p:
            params.do_resize = True
            lo, hi = RESIZE_SCALE_RANGES[severity]
            params.resize_scale = float(lo + (hi - lo) * rng.random())

        # Re-encoding simulation
        if rng.random() < self.config.reencode_p:
            params.do_reencode = True
            lo_q, hi_q = JPEG_QUALITY_RANGES[severity]
            params.reencode_quality = int(rng.randint(lo_q, hi_q + 1))
            lo_s, hi_s = RESIZE_SCALE_RANGES[severity]
            params.reencode_scale = float(lo_s + (hi_s - lo_s) * rng.random())

        # Temporal: frame drop
        if rng.random() < self.config.frame_drop_p and num_frames > 2:
            params.do_frame_drop = True
            max_drop = min(self.config.max_drop_frames, num_frames - 1)
            n_drop = int(rng.randint(1, max_drop + 1))
            params.drop_indices = sorted(
                rng.choice(num_frames, size=n_drop, replace=False).tolist()
            )

        return params

    def augment_tensor(
        self,
        sequence: torch.Tensor,
        params: Optional[SequenceDegradationParams] = None,
    ) -> Tuple[torch.Tensor, SequenceDegradationParams]:
        """Apply sequence-consistent degradation to a (T, C, H, W) float32 tensor.

        The tensor is expected to be in [0, 1] range BEFORE ImageNet normalization.
        This augmentor should be called before normalization in the pipeline.

        Args:
            sequence: (T, C, H, W) float32 tensor in [0, 1].
            params: Optional pre-sampled params; if None, they are sampled fresh.

        Returns:
            Tuple of (augmented_tensor, params_used).
        """
        if not self.config.enabled:
            if params is None:
                params = SequenceDegradationParams(severity=SeverityTier.MILD)
            return sequence, params

        self._call_count += 1
        rng = self._get_rng()

        t, c, h, w = sequence.shape

        if params is None:
            params = self.sample_params(t, rng)

        # Check if any spatial degradation is needed
        needs_spatial = any([
            params.do_jpeg, params.do_blur, params.do_noise,
            params.do_resize, params.do_reencode,
        ])

        if needs_spatial:
            # Convert to uint8 HWC for OpenCV operations
            frames_np = (sequence.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()

            augmented_frames = []
            for i in range(t):
                frame = frames_np[i].copy()

                # Apply degradations in a fixed order for reproducibility
                if params.do_reencode:
                    frame = apply_reencode_simulation(
                        frame, params.reencode_quality, params.reencode_scale
                    )
                if params.do_jpeg:
                    frame = apply_jpeg_compression(frame, params.jpeg_quality)
                if params.do_blur:
                    frame = apply_gaussian_blur(frame, params.blur_kernel)
                if params.do_resize:
                    frame = apply_downsample_upsample(frame, params.resize_scale)
                if params.do_noise:
                    # Use frame-specific seed derived from the global rng + frame index
                    # to get different noise patterns per frame (realistic)
                    # but with the same std (sequence-consistent parameter).
                    frame_rng = np.random.RandomState(
                        rng.randint(0, 2**31) + i
                    )
                    frame = apply_gaussian_noise(frame, params.noise_std, frame_rng)

                augmented_frames.append(frame)

            # Convert back to float32 tensor (T, C, H, W) in [0, 1]
            stacked = np.stack(augmented_frames, axis=0)  # (T, H, W, C)
            sequence = (
                torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
            )

        # Temporal perturbation: frame dropping (duplicate neighbor)
        if params.do_frame_drop and params.drop_indices:
            for drop_idx in params.drop_indices:
                # Replace dropped frame with previous frame (or next if first)
                src_idx = max(0, drop_idx - 1)
                if src_idx == drop_idx:
                    src_idx = min(t - 1, drop_idx + 1)
                sequence[drop_idx] = sequence[src_idx].clone()

        return sequence, params

    def augment_crops(
        self,
        crop_images: List[np.ndarray],
        params: Optional[SequenceDegradationParams] = None,
    ) -> Tuple[List[np.ndarray], SequenceDegradationParams]:
        """Apply sequence-consistent degradation to a list of HWC uint8 crop images.

        This operates before the TensorPreprocessor step.

        Args:
            crop_images: List of T HxWxC uint8 RGB ndarrays.
            params: Optional pre-sampled params.

        Returns:
            Tuple of (augmented_crops, params_used).
        """
        if not self.config.enabled:
            if params is None:
                params = SequenceDegradationParams(severity=SeverityTier.MILD)
            return crop_images, params

        self._call_count += 1
        rng = self._get_rng()
        t = len(crop_images)

        if params is None:
            params = self.sample_params(t, rng)

        augmented = []
        for i, frame in enumerate(crop_images):
            frame = frame.copy()

            if params.do_reencode:
                frame = apply_reencode_simulation(
                    frame, params.reencode_quality, params.reencode_scale
                )
            if params.do_jpeg:
                frame = apply_jpeg_compression(frame, params.jpeg_quality)
            if params.do_blur:
                frame = apply_gaussian_blur(frame, params.blur_kernel)
            if params.do_resize:
                frame = apply_downsample_upsample(frame, params.resize_scale)
            if params.do_noise:
                frame_rng = np.random.RandomState(
                    rng.randint(0, 2**31) + i
                )
                frame = apply_gaussian_noise(frame, params.noise_std, frame_rng)

            augmented.append(frame)

        # Temporal: frame dropping
        if params.do_frame_drop and params.drop_indices:
            for drop_idx in params.drop_indices:
                src_idx = max(0, drop_idx - 1)
                if src_idx == drop_idx:
                    src_idx = min(t - 1, drop_idx + 1)
                augmented[drop_idx] = augmented[src_idx].copy()

        return augmented, params

    def __repr__(self) -> str:
        return (
            f"RobustAugmentor(enabled={self.config.enabled}, "
            f"jpeg_p={self.config.jpeg_p}, blur_p={self.config.blur_p}, "
            f"noise_p={self.config.noise_p}, resize_p={self.config.resize_p}, "
            f"reencode_p={self.config.reencode_p}, "
            f"severity={self.config.severity_distribution})"
        )
