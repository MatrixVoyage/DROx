"""Pytest fixtures and helpers for testing video sampling, face cropping, and pipelines."""

from pathlib import Path
from typing import Callable, List, Optional
import cv2
import numpy as np
import pytest

from src.data.detector import FaceDetector
from src.data.types import BoundingBox


class MockFaceDetector(FaceDetector):
    """Deterministic mock face detector for fast unit testing."""

    def __init__(
        self,
        box_to_return: Optional[BoundingBox] = None,
        detection_pattern: Optional[List[bool]] = None,
    ):
        self.default_box = box_to_return or BoundingBox(
            x1=20.0, y1=20.0, x2=80.0, y2=80.0, confidence=0.95
        )
        self.pattern = detection_pattern
        self.call_count = 0

    def detect_face(self, image: np.ndarray) -> Optional[BoundingBox]:
        if self.pattern is not None:
            if self.call_count < len(self.pattern):
                should_detect = self.pattern[self.call_count]
            else:
                should_detect = False
            self.call_count += 1
            return self.default_box if should_detect else None

        self.call_count += 1
        return self.default_box


@pytest.fixture
def create_synthetic_video(tmp_path: Path) -> Callable[..., Path]:
    """Fixture providing a helper function to create synthetic MP4 test videos."""

    def _create_video(
        filename: str = "test_video.mp4",
        num_frames: int = 30,
        width: int = 120,
        height: int = 120,
        fps: float = 30.0,
    ) -> Path:
        out_path = tmp_path / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        for i in range(num_frames):
            # Create a simple frame with a gradient and circle
            frame = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
            cv2.circle(frame, (width // 2, height // 2), 25, (200, 150, 100), -1)
            writer.write(frame)

        writer.release()
        return out_path

    return _create_video


@pytest.fixture
def mock_face_detector() -> MockFaceDetector:
    return MockFaceDetector()
