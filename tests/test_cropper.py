"""Tests for FaceCropper, margin expansion, fallback chain, and quality thresholds."""

import numpy as np
import pytest

from src.utils.config import CropConfig, FallbackConfig
from src.data.cropper import FaceCropper
from src.data.types import BoundingBox, CropType
from tests.conftest import MockFaceDetector


def test_expand_bbox_clamping():
    """Verify margin expansion calculates correctly and clamps to image borders."""
    cropper = FaceCropper(CropConfig(margin_ratio=0.20))
    # Box: (20, 20, 80, 80) -> width 60, height 60, margin 12
    bbox = BoundingBox(x1=20.0, y1=20.0, x2=80.0, y2=80.0, confidence=0.95)
    x1, y1, x2, y2 = cropper.expand_bbox(bbox, frame_width=100, frame_height=100)
    assert (x1, y1, x2, y2) == (8, 8, 92, 92)

    # Box near boundary: clamped to 0 and frame bounds
    bbox_edge = BoundingBox(x1=5.0, y1=5.0, x2=98.0, y2=98.0, confidence=0.95)
    ex1, ey1, ex2, ey2 = cropper.expand_bbox(bbox_edge, frame_width=100, frame_height=100)
    assert ex1 == 0
    assert ey1 == 0
    assert ex2 == 100
    assert ey2 == 100


def test_fallback_chain_and_counters():
    """Verify fallback order (detected -> previous_bbox -> center_crop) and counter accounting."""
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]

    # Frame 0: No face detected -> falls back to center crop
    # Frame 1: Face detected -> DETECTED
    # Frame 2: No face detected -> falls back to PREVIOUS_BBOX (from frame 1)
    # Frame 3: Face detected -> DETECTED
    pattern = [False, True, False, True]
    detector = MockFaceDetector(detection_pattern=pattern)

    config = CropConfig(
        fallback=FallbackConfig(use_previous_bbox=True, allow_center_crop=True),
        min_valid_frames=2,
        max_fallback_ratio=0.80,
    )
    cropper = FaceCropper(config)
    results, summary = cropper.crop_sequence(frames, detector)

    assert len(results) == 4
    assert results[0].crop_type == CropType.CENTER_CROP
    assert results[1].crop_type == CropType.DETECTED
    assert results[2].crop_type == CropType.PREVIOUS_BBOX
    assert results[3].crop_type == CropType.DETECTED

    assert summary["detected_frames"] == 2
    assert summary["previous_bbox_frames"] == 1
    assert summary["center_crop_frames"] == 1
    assert summary["failed_frames"] == 0
    assert summary["valid_frames"] == 4

    # fallback_ratio = (1 + 1) / 4 = 0.50
    assert summary["fallback_ratio"] == 0.50
    assert summary["failure_ratio"] == 0.0
    assert summary["is_rejected"] is False


def test_quality_rejection_min_valid_frames():
    """Verify sequence is rejected when valid_frames < min_valid_frames."""
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(16)]

    # Center crop not allowed, only 8 frames detected
    pattern = [True] * 8 + [False] * 8
    detector = MockFaceDetector(detection_pattern=pattern)

    config = CropConfig(
        fallback=FallbackConfig(use_previous_bbox=False, allow_center_crop=False),
        min_valid_frames=12,
    )
    cropper = FaceCropper(config)
    results, summary = cropper.crop_sequence(frames, detector)

    assert summary["detected_frames"] == 8
    assert summary["failed_frames"] == 8
    assert summary["valid_frames"] == 8
    assert summary["is_rejected"] is True


def test_quality_rejection_max_fallback_ratio():
    """Verify sequence is rejected when fallback_ratio exceeds max_fallback_ratio."""
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]

    # Frame 0 detected, remaining 9 frames use previous_bbox
    pattern = [True] + [False] * 9
    detector = MockFaceDetector(detection_pattern=pattern)

    config = CropConfig(
        fallback=FallbackConfig(use_previous_bbox=True, allow_center_crop=False),
        min_valid_frames=5,
        max_fallback_ratio=0.50,  # 9/10 = 0.90 > 0.50
    )
    cropper = FaceCropper(config)
    results, summary = cropper.crop_sequence(frames, detector)

    assert summary["fallback_ratio"] == 0.90
    assert summary["is_rejected"] is True
    assert "fallback_ratio" in summary["reject_reason"]
