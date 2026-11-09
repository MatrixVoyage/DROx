"""Data types, error codes, and result dataclasses for the video preprocessing pipeline."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VideoErrorCode(str, Enum):
    """Enumeration of standard video error codes."""
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    UNREADABLE_VIDEO = "UNREADABLE_VIDEO"
    ZERO_FRAMES = "ZERO_FRAMES"
    DECODER_ERROR = "DECODER_ERROR"
    INVALID_FPS = "INVALID_FPS"
    TOO_SHORT = "TOO_SHORT"


class VideoReadError(Exception):
    """Raised when an error occurs during video file validation or frame decoding."""

    def __init__(self, error_code: VideoErrorCode, message: str):
        super().__init__(f"[{error_code.value}] {message}")
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates (x1, y1, x2, y2) and detection confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class CropType(str, Enum):
    """Origin type of the face crop."""
    DETECTED = "detected"
    PREVIOUS_BBOX = "previous_bbox"
    CENTER_CROP = "center_crop"
    FAILED = "failed"


@dataclass
class CropResult:
    """Result of cropping a single frame."""
    image: Optional[np.ndarray]
    crop_type: CropType
    bbox: Optional[BoundingBox] = None


@dataclass
class PreprocessResult:
    """Comprehensive outcome of the video-to-tensor preprocessing pipeline."""
    video_id: str
    group_id: str
    split: str = "unassigned"
    tensor: Optional[Any] = None  # torch.Tensor of shape (T, 3, 112, 112), float32
    frame_indices: List[int] = field(default_factory=list)
    detected_frames: int = 0
    previous_bbox_frames: int = 0
    center_crop_frames: int = 0
    failed_frames: int = 0
    valid_frames: int = 0
    fallback_ratio: float = 0.0
    failure_ratio: float = 0.0
    status: str = "success"  # "success", "rejected_quality", "rejected_short", "failed_video"
    error_code: Optional[VideoErrorCode] = None
    error_message: Optional[str] = None

    def to_metadata_dict(self, config_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert result metrics and metadata into a serializable dictionary."""
        d = {
            "video_id": self.video_id,
            "group_id": self.group_id,
            "split": self.split,
            "status": self.status,
            "frame_indices": self.frame_indices,
            "counters": {
                "detected_frames": self.detected_frames,
                "previous_bbox_frames": self.previous_bbox_frames,
                "center_crop_frames": self.center_crop_frames,
                "failed_frames": self.failed_frames,
                "valid_frames": self.valid_frames,
            },
            "metrics": {
                "fallback_ratio": float(self.fallback_ratio),
                "failure_ratio": float(self.failure_ratio),
            },
            "error_code": self.error_code.value if self.error_code else None,
            "error_message": self.error_message,
        }
        if config_snapshot is not None:
            d["config"] = config_snapshot
        return d
