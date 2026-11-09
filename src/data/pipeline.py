"""End-to-end video-to-tensor preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.utils.config import AppConfig
from src.data.detector import FaceDetector, MTCNNFaceDetector
from src.data.cropper import FaceCropper
from src.data.preprocessor import TensorPreprocessor
from src.data.types import PreprocessResult, VideoErrorCode, VideoReadError
from src.data.video import VideoSampler


class VideoFacePipeline:
    """Orchestrates VideoSampler -> FaceDetector -> FaceCropper -> TensorPreprocessor."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        detector: Optional[FaceDetector] = None,
    ):
        self.config = config or AppConfig()

        self.sampler = VideoSampler(self.config.data.sampling)

        if detector is not None:
            self.detector = detector
        else:
            effective_device = self.config.get_effective_device()
            self.detector = MTCNNFaceDetector(
                config=self.config.data.detection,
                device=effective_device,
            )

        self.cropper = FaceCropper(self.config.data.crop)
        self.preprocessor = TensorPreprocessor(
            image_size=self.config.data.crop.image_size,
            normalization=self.config.data.normalization,
        )

    def process_video(
        self,
        video_path: Union[str, Path],
        video_id: Optional[str] = None,
        group_id: Optional[str] = None,
        split: str = "unassigned",
    ) -> PreprocessResult:
        """Process a single video into a verified facial sequence tensor and metadata.

        Args:
            video_path: Path to the video file.
            video_id: Optional unique video identifier.
            group_id: Optional group identity; defaults to video_id.
            split: Target split ("train", "val", "test", or "unassigned").

        Returns:
            PreprocessResult containing status, metrics, and tensor on success.
        """
        path = Path(video_path)
        vid = video_id if video_id is not None else path.stem
        gid = group_id if group_id is not None else vid

        # 1. Video Frame Sampling
        try:
            frames, indices = self.sampler.sample_frames(
                video_path=path,
                video_id=vid,
                base_seed=self.config.seed,
            )
        except VideoReadError as e:
            status = "rejected_short" if e.error_code == VideoErrorCode.TOO_SHORT else "failed_video"
            return PreprocessResult(
                video_id=vid,
                group_id=gid,
                split=split,
                status=status,
                error_code=e.error_code,
                error_message=e.message,
            )
        except Exception as e:
            return PreprocessResult(
                video_id=vid,
                group_id=gid,
                split=split,
                status="failed_video",
                error_code=VideoErrorCode.DECODER_ERROR,
                error_message=str(e),
            )

        # 2. Face Detection and Cropping Sequence
        crop_results, summary = self.cropper.crop_sequence(frames, self.detector)

        detected_frames = summary["detected_frames"]
        previous_bbox_frames = summary["previous_bbox_frames"]
        center_crop_frames = summary["center_crop_frames"]
        failed_frames = summary["failed_frames"]
        valid_frames = summary["valid_frames"]
        fallback_ratio = summary["fallback_ratio"]
        failure_ratio = summary["failure_ratio"]

        # 3. Quality Control Evaluation
        if summary["is_rejected"]:
            return PreprocessResult(
                video_id=vid,
                group_id=gid,
                split=split,
                frame_indices=indices,
                detected_frames=detected_frames,
                previous_bbox_frames=previous_bbox_frames,
                center_crop_frames=center_crop_frames,
                failed_frames=failed_frames,
                valid_frames=valid_frames,
                fallback_ratio=fallback_ratio,
                failure_ratio=failure_ratio,
                status="rejected_quality",
                error_message=summary["reject_reason"],
            )

        # 4. Tensor Preprocessing
        crop_images = [cr.image for cr in crop_results]
        tensor = self.preprocessor.preprocess_crops(crop_images)

        # 5. Assert Pipeline Success Invariants
        expected_t = self.config.data.sampling.num_frames
        expected_size = self.config.data.crop.image_size

        assert tensor is not None, "Pipeline invariant violated: tensor is None on success"
        assert tensor.shape == (expected_t, 3, expected_size, expected_size), (
            f"Pipeline invariant violated: expected shape {(expected_t, 3, expected_size, expected_size)}, "
            f"got {tensor.shape}"
        )
        assert tensor.dtype == torch.float32, (
            f"Pipeline invariant violated: expected dtype torch.float32, got {tensor.dtype}"
        )
        assert not torch.isnan(tensor).any(), "Pipeline invariant violated: tensor contains NaN"
        assert not torch.isinf(tensor).any(), "Pipeline invariant violated: tensor contains Inf"
        assert len(indices) == expected_t, (
            f"Pipeline invariant violated: expected {expected_t} frame_indices, got {len(indices)}"
        )

        return PreprocessResult(
            video_id=vid,
            group_id=gid,
            split=split,
            tensor=tensor,
            frame_indices=indices,
            detected_frames=detected_frames,
            previous_bbox_frames=previous_bbox_frames,
            center_crop_frames=center_crop_frames,
            failed_frames=failed_frames,
            valid_frames=valid_frames,
            fallback_ratio=fallback_ratio,
            failure_ratio=failure_ratio,
            status="success",
        )
