"""Face detection interface and MTCNN detector implementation."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

try:
    import torch
    from facenet_pytorch import MTCNN
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False

from src.utils.config import DetectionConfig
from src.data.types import BoundingBox


class FaceDetector(ABC):
    """Abstract interface for face detection."""

    @abstractmethod
    def detect_face(self, image: np.ndarray) -> Optional[BoundingBox]:
        """Detect the primary face in an RGB image.

        Args:
            image: RGB image as an HxWxC uint8 ndarray.

        Returns:
            BoundingBox if a valid face is detected, otherwise None.
        """
        pass


class MTCNNFaceDetector(FaceDetector):
    """MTCNN-based face detector with deterministic primary face selection.

    Reused across frames within a pipeline/process instance.
    Runs detection within torch.inference_mode().
    """

    def __init__(self, config: Optional[DetectionConfig] = None, device: str = "cpu"):
        if not HAS_FACENET:
            raise ImportError(
                "facenet-pytorch is required for MTCNNFaceDetector. "
                "Install it with: pip install facenet-pytorch"
            )

        self.config = config or DetectionConfig()
        self.device = device

        # Initialize MTCNN instance
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            post_process=False,
            min_face_size=20,
        )

    def detect_face(self, image: np.ndarray) -> Optional[BoundingBox]:
        """Detect faces, filter by confidence threshold, and deterministically pick primary face.

        Args:
            image: RGB image as an HxWxC uint8 ndarray.

        Returns:
            Primary BoundingBox or None.
        """
        with torch.inference_mode():
            # MTCNN accepts PIL Image, numpy array (HxWxC), or torch tensor
            boxes, probs = self.mtcnn.detect(image)

        if boxes is None or len(boxes) == 0:
            return None

        # Filter candidate detections by confidence_threshold
        candidates: List[BoundingBox] = []
        for box, prob in zip(boxes, probs):
            if prob is None or np.isnan(prob):
                continue
            prob_float = float(prob)
            if prob_float >= self.config.confidence_threshold:
                x1, y1, x2, y2 = [float(coord) for coord in box]
                candidates.append(
                    BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=prob_float,
                    )
                )

        if not candidates:
            return None

        # Deterministic primary-face selection
        if self.config.selection_strategy == "largest":
            # Primary: area descending; Secondary: confidence descending; Tertiary: coordinate
            candidates.sort(
                key=lambda b: (b.area, b.confidence, -b.x1, -b.y1),
                reverse=True,
            )
        elif self.config.selection_strategy == "highest_confidence":
            candidates.sort(
                key=lambda b: (b.confidence, b.area, -b.x1, -b.y1),
                reverse=True,
            )
        else:
            raise ValueError(f"Unknown selection strategy: {self.config.selection_strategy}")

        return candidates[0]

    def detect_sequence(self, frames: List[np.ndarray]) -> List[Optional[BoundingBox]]:
        """Batched face detection across all frames in a video sequence for accelerated inference.

        Args:
            frames: List of RGB images (HxWxC uint8).

        Returns:
            List of primary BoundingBoxes (or None) corresponding to each input frame.
        """
        if not frames:
            return []

        with torch.inference_mode():
            batch_boxes, batch_probs = self.mtcnn.detect(frames)

        if batch_boxes is None or len(batch_boxes) == 0:
            return [None] * len(frames)

        results: List[Optional[BoundingBox]] = []
        for boxes, probs in zip(batch_boxes, batch_probs):
            if boxes is None or len(boxes) == 0:
                results.append(None)
                continue

            candidates: List[BoundingBox] = []
            for box, prob in zip(boxes, probs):
                if prob is None or np.isnan(prob):
                    continue
                prob_float = float(prob)
                if prob_float >= self.config.confidence_threshold:
                    x1, y1, x2, y2 = [float(coord) for coord in box]
                    candidates.append(
                        BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=prob_float,
                        )
                    )

            if not candidates:
                results.append(None)
                continue

            if self.config.selection_strategy == "largest":
                candidates.sort(
                    key=lambda b: (b.area, b.confidence, -b.x1, -b.y1),
                    reverse=True,
                )
            elif self.config.selection_strategy == "highest_confidence":
                candidates.sort(
                    key=lambda b: (b.confidence, b.area, -b.x1, -b.y1),
                    reverse=True,
                )
            else:
                raise ValueError(f"Unknown selection strategy: {self.config.selection_strategy}")

            results.append(candidates[0])

        return results
