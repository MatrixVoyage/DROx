"""Tests for VideoSampler, frame extraction, determinism, and error taxonomy."""

from pathlib import Path
import pytest

from src.utils.config import SamplingConfig
from src.data.types import VideoErrorCode, VideoReadError
from src.data.video import VideoSampler


def test_video_sampler_file_not_found(tmp_path: Path):
    """Verify FILE_NOT_FOUND error code on non-existent file."""
    sampler = VideoSampler()
    fake_path = tmp_path / "does_not_exist.mp4"
    with pytest.raises(VideoReadError) as exc_info:
        sampler.sample_frames(fake_path)
    assert exc_info.value.error_code == VideoErrorCode.FILE_NOT_FOUND


def test_video_sampler_short_video_reject(create_synthetic_video):
    """Verify TOO_SHORT error code when total frames < num_frames."""
    # Create video with only 5 frames
    short_video = create_synthetic_video("short.mp4", num_frames=5)
    config = SamplingConfig(num_frames=16, short_video_policy="reject")
    sampler = VideoSampler(config)

    with pytest.raises(VideoReadError) as exc_info:
        sampler.sample_frames(short_video)
    assert exc_info.value.error_code == VideoErrorCode.TOO_SHORT


def test_video_sampler_uniform(create_synthetic_video):
    """Verify uniform sampling returns exact T frames with deterministic indices."""
    video_path = create_synthetic_video("uniform.mp4", num_frames=30)
    config = SamplingConfig(method="uniform", num_frames=8)
    sampler = VideoSampler(config)

    frames, indices = sampler.sample_frames(video_path)
    assert len(frames) == 8
    assert len(indices) == 8
    assert indices == sorted(indices)
    assert frames[0].shape == (120, 120, 3)

    # Re-run should give identical indices without requiring seed
    _, indices_2 = sampler.sample_frames(video_path)
    assert indices == indices_2


def test_video_sampler_random_determinism(create_synthetic_video):
    """Verify random sampling is deterministic for the same video_id and seed."""
    video_path = create_synthetic_video("random.mp4", num_frames=50)
    config = SamplingConfig(method="random", num_frames=16)
    sampler = VideoSampler(config)

    _, indices_run1 = sampler.sample_frames(video_path, video_id="vid_100", base_seed=42)
    _, indices_run2 = sampler.sample_frames(video_path, video_id="vid_100", base_seed=42)
    assert indices_run1 == indices_run2
    assert len(indices_run1) == 16
    assert indices_run1 == sorted(indices_run1)

    # Different video_id or seed should yield different indices
    _, indices_diff_id = sampler.sample_frames(video_path, video_id="vid_200", base_seed=42)
    assert indices_run1 != indices_diff_id

    _, indices_diff_seed = sampler.sample_frames(video_path, video_id="vid_100", base_seed=99)
    assert indices_run1 != indices_diff_seed
