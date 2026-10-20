"""Tests for VideoFacePipeline enforcing success invariants and determinism."""

import pytest
import torch

from src.utils.config import AppConfig
from src.data.pipeline import VideoFacePipeline
from tests.conftest import MockFaceDetector


def test_pipeline_success_invariants(create_synthetic_video):
    """Assert all strict pipeline success invariants on processed synthetic video."""
    video_path = create_synthetic_video("pipeline_test.mp4", num_frames=30)
    config = AppConfig()
    config.data.sampling.num_frames = 16
    config.data.crop.image_size = 112

    mock_detector = MockFaceDetector()
    pipeline = VideoFacePipeline(config=config, detector=mock_detector)

    result = pipeline.process_video(
        video_path=video_path,
        video_id="video_alpha",
        group_id="group_alpha",
        split="train",
    )

    assert result.status == "success"
    assert result.video_id == "video_alpha"
    assert result.group_id == "group_alpha"
    assert result.split == "train"

    # Pipeline Invariant Checks
    assert result.tensor is not None
    assert result.tensor.shape == (16, 3, 112, 112)
    assert result.tensor.dtype == torch.float32
    assert not torch.isnan(result.tensor).any()
    assert not torch.isinf(result.tensor).any()
    assert len(result.frame_indices) == 16
    assert result.frame_indices == sorted(result.frame_indices)


def test_pipeline_bitwise_reproducibility(create_synthetic_video):
    """Verify identical configuration, video, and seed produces bitwise identical tensors."""
    video_path = create_synthetic_video("repro_test.mp4", num_frames=40)
    config = AppConfig(seed=42)
    config.data.sampling.method = "random"
    config.data.sampling.num_frames = 16

    mock_detector1 = MockFaceDetector()
    pipeline1 = VideoFacePipeline(config=config, detector=mock_detector1)
    res1 = pipeline1.process_video(video_path, video_id="vid_sample")

    mock_detector2 = MockFaceDetector()
    pipeline2 = VideoFacePipeline(config=config, detector=mock_detector2)
    res2 = pipeline2.process_video(video_path, video_id="vid_sample")

    assert res1.frame_indices == res2.frame_indices
    assert torch.equal(res1.tensor, res2.tensor)


def test_pipeline_normalization_modes(create_synthetic_video):
    """Verify normalization options ('none' vs 'imagenet')."""
    video_path = create_synthetic_video("norm_test.mp4", num_frames=20)

    # 1. Normalization: none -> values in [0.0, 1.0]
    cfg_none = AppConfig()
    cfg_none.data.normalization = "none"
    cfg_none.data.sampling.num_frames = 8
    cfg_none.data.crop.min_valid_frames = 6
    cfg_none.validate()
    p_none = VideoFacePipeline(config=cfg_none, detector=MockFaceDetector())
    res_none = p_none.process_video(video_path)

    assert res_none.status == "success"
    assert res_none.tensor is not None
    assert res_none.tensor.min() >= 0.0
    assert res_none.tensor.max() <= 1.0

    # 2. Normalization: imagenet -> normalized values can be negative
    cfg_imgnet = AppConfig()
    cfg_imgnet.data.normalization = "imagenet"
    cfg_imgnet.data.sampling.num_frames = 8
    cfg_imgnet.data.crop.min_valid_frames = 6
    cfg_imgnet.validate()
    p_imgnet = VideoFacePipeline(config=cfg_imgnet, detector=MockFaceDetector())
    res_imgnet = p_imgnet.process_video(video_path)

    assert res_imgnet.status == "success"
    assert res_imgnet.tensor is not None
    # Standardized with ImageNet mean/std typically has negative minimum values
    assert res_imgnet.tensor.min() < 0.0
