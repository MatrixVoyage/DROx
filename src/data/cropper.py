"""Face cropping, bounding box expansion, and fallback chain handling."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.utils.config import CropConfig
from src.data.detector import FaceDetector
from src.data.types import BoundingBox, CropResult, CropType


class FaceCropper:
    """Extracts cropped face regions with configurable margin and fallback chain."""

    def __init__(self, config: Optional[CropConfig] = None):
        self.config = config or CropConfig()

    def expand_bbox(
        self,
        bbox: BoundingBox,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by margin_ratio and clamp to frame boundaries.

        Args:
            bbox: Input bounding box.
            frame_width: Width of the frame in pixels.
            frame_height: Height of the frame in pixels.

        Returns:
            Tuple of clamped integer coordinates (x1, y1, x2, y2).
        """
        w = bbox.width
        h = bbox.height
        margin_x = w * self.config.margin_ratio
        margin_y = h * self.config.margin_ratio

        x1 = max(0, int(round(bbox.x1 - margin_x)))
        y1 = max(0, int(round(bbox.y1 - margin_y)))
        x2 = min(frame_width, int(round(bbox.x2 + margin_x)))
        y2 = min(frame_height, int(round(bbox.y2 + margin_y)))

        # Ensure non-zero width and height
        if x2 <= x1:
            x2 = min(frame_width, x1 + 1)
        if y2 <= y1:
            y2 = min(frame_height, y1 + 1)

        return x1, y1, x2, y2

    def get_center_crop(self, frame: np.ndarray) -> Tuple[np.ndarray, BoundingBox]:
        """Compute square center crop from frame."""
        h, w = frame.shape[:2]
        crop_size = min(h, w)
        x1 = (w - crop_size) // 2
        y1 = (h - crop_size) // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        crop_img = frame[y1:y2, x1:x2]
        bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=0.0)
        return crop_img, bbox

    def crop_sequence(
        self,
        frames: List[np.ndarray],
        detector: FaceDetector,
    ) -> Tuple[List[CropResult], Dict[str, Any]]:
        """Process a sequence of frames through detection and fallback chain.

        Fallback chain order:
        1. Detected bbox (if confidence >= threshold).
        2. Previous bbox (if enabled and previous detected bbox exists).
        3. Center crop (if enabled as last resort).
        4. Failed crop (if no fallback succeeds).

        Args:
            frames: List of RGB frames (HxWxC uint8).
            detector: FaceDetector instance.

        Returns:
            Tuple of (list of CropResult, summary dictionary with counters and metrics).
        """
        num_frames = len(frames)
        detected_frames = 0
        previous_bbox_frames = 0
        center_crop_frames = 0
        failed_frames = 0

        last_detected_bbox: Optional[BoundingBox] = None
        crop_results: List[CropResult] = []

        if hasattr(detector, "detect_sequence"):
            bboxes = detector.detect_sequence(frames)
        else:
            bboxes = [detector.detect_face(f) for f in frames]

        for frame, bbox in zip(frames, bboxes):
            h, w = frame.shape[:2]

            if bbox is not None:
                # 1. Successful detection
                x1, y1, x2, y2 = self.expand_bbox(bbox, w, h)
                crop_img = frame[y1:y2, x1:x2]
                last_detected_bbox = bbox
                detected_frames += 1
                crop_results.append(CropResult(image=crop_img, crop_type=CropType.DETECTED, bbox=bbox))
                continue

            # 2. Fallback: Previous BBox
            if self.config.fallback.use_previous_bbox and last_detected_bbox is not None:
                x1, y1, x2, y2 = self.expand_bbox(last_detected_bbox, w, h)
                crop_img = frame[y1:y2, x1:x2]
                previous_bbox_frames += 1
                crop_results.append(
                    CropResult(image=crop_img, crop_type=CropType.PREVIOUS_BBOX, bbox=last_detected_bbox)
                )
                continue

            # 3. Fallback: Center Crop
            if self.config.fallback.allow_center_crop:
                crop_img, center_bbox = self.get_center_crop(frame)
                center_crop_frames += 1
                crop_results.append(
                    CropResult(image=crop_img, crop_type=CropType.CENTER_CROP, bbox=center_bbox)
                )
                continue

            # 4. Failed
            failed_frames += 1
            crop_results.append(CropResult(image=None, crop_type=CropType.FAILED, bbox=None))

        valid_frames = detected_frames + previous_bbox_frames + center_crop_frames
        fallback_ratio = (previous_bbox_frames + center_crop_frames) / num_frames if num_frames > 0 else 0.0
        failure_ratio = failed_frames / num_frames if num_frames > 0 else 0.0

        # Quality control validation
        is_rejected = False
        reject_reason = None

        if failed_frames > 0:
            is_rejected = True
            reject_reason = f"Sequence has {failed_frames} failed frame crops out of {num_frames}"
        elif valid_frames < self.config.min_valid_frames:
            is_rejected = True
            reject_reason = (
                f"valid_frames ({valid_frames}) is less than min_valid_frames ({self.config.min_valid_frames})"
            )
        elif fallback_ratio > self.config.max_fallback_ratio:
            is_rejected = True
            reject_reason = (
                f"fallback_ratio ({fallback_ratio:.2f}) exceeds max_fallback_ratio "
                f"({self.config.max_fallback_ratio:.2f})"
            )

        summary = {
            "num_frames": num_frames,
            "detected_frames": detected_frames,
            "previous_bbox_frames": previous_bbox_frames,
            "center_crop_frames": center_crop_frames,
            "failed_frames": failed_frames,
            "valid_frames": valid_frames,
            "fallback_ratio": fallback_ratio,
            "failure_ratio": failure_ratio,
            "is_rejected": is_rejected,
            "reject_reason": reject_reason,
        }

        return crop_results, summary
