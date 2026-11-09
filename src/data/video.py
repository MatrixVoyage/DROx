"""Video frame sampling and decoding with robust error taxonomy."""

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.utils.config import SamplingConfig
from src.utils.seed import derive_video_seed
from src.data.types import VideoErrorCode, VideoReadError


class VideoSampler:
    """Extracts sampled video frames with strict error handling and deterministic seeding."""

    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()

    def sample_frames(
        self,
        video_path: Union[str, Path],
        video_id: Optional[str] = None,
        base_seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Sample exact N frames from a video file.

        Args:
            video_path: Absolute or relative path to the video file.
            video_id: Optional identifier; defaults to video file stem.
            base_seed: Base random seed for random sampling.

        Returns:
            Tuple of (list of RGB frames as uint8 ndarrays, list of frame indices).

        Raises:
            VideoReadError: With specific VideoErrorCode on validation or decode failure.
        """
        path = Path(video_path)
        if not path.exists():
            raise VideoReadError(
                VideoErrorCode.FILE_NOT_FOUND,
                f"Video file does not exist: {path}",
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoReadError(
                VideoErrorCode.UNREADABLE_VIDEO,
                f"Could not open video file: {path}",
            )

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))

            if total_frames <= 0:
                raise VideoReadError(
                    VideoErrorCode.ZERO_FRAMES,
                    f"Video has zero frames: {path}",
                )

            if fps <= 0.0 or math.isnan(fps):
                raise VideoReadError(
                    VideoErrorCode.INVALID_FPS,
                    f"Video has invalid FPS ({fps}): {path}",
                )

            num_frames = self.config.num_frames

            # Short video policy
            if total_frames < num_frames:
                if self.config.short_video_policy == "reject":
                    raise VideoReadError(
                        VideoErrorCode.TOO_SHORT,
                        f"Video has {total_frames} frames, fewer than requested {num_frames}.",
                    )
                raise VideoReadError(
                    VideoErrorCode.TOO_SHORT,
                    f"Video has {total_frames} frames < {num_frames}.",
                )

            # Determine frame indices
            effective_video_id = video_id if video_id is not None else path.stem

            if self.config.method == "uniform":
                # Uniform sampling is deterministic by definition; does not use seed
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
            elif self.config.method == "strided":
                # Alternative deterministic sampling: fixed stride step across frames
                step = max(1, (total_frames - 1) // (num_frames - 1)) if num_frames > 1 else 1
                indices = [min(total_frames - 1, i * step) for i in range(num_frames)]
            elif self.config.method == "random":
                # Paper-aligned default: derive per-video seed via SHA-256
                vseed = derive_video_seed(base_seed, effective_video_id)
                rng = np.random.RandomState(vseed)
                selected = rng.choice(total_frames, size=num_frames, replace=False)
                indices = sorted(selected.tolist())
            else:
                raise ValueError(f"Unknown sampling method: {self.config.method}")

            # Read and decode frames
            frames: List[np.ndarray] = []
            current_pos = 0
            for idx in indices:
                gap = idx - current_pos
                if 0 <= gap <= 30:
                    for _ in range(gap):
                        cap.grab()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

                ret, frame_bgr = cap.read()
                current_pos = idx + 1
                if not ret or frame_bgr is None:
                    raise VideoReadError(
                        VideoErrorCode.DECODER_ERROR,
                        f"Failed to decode frame at index {idx} in {path}",
                    )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            return frames, indices

        finally:
            cap.release()
