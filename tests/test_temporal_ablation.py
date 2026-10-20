"""Unit tests for Milestone 9 temporal ablation."""

import pytest

from src.data.video import VideoSampler
from src.evaluation.temporal_ablation import run_temporal_ablation_suite
from src.utils.config import SamplingConfig


def test_video_sampler_methods():
    cfg_uni = SamplingConfig(method="uniform", num_frames=16)
    s_uni = VideoSampler(config=cfg_uni)
    assert s_uni.config.method == "uniform"

    cfg_strided = SamplingConfig(method="strided", num_frames=16)
    s_strided = VideoSampler(config=cfg_strided)
    assert s_strided.config.method == "strided"


def test_temporal_ablation_incompatibilities():
    res = run_temporal_ablation_suite(eval_videos=[], model=None, device="cpu")
    incomp = res["architectural_incompatibilities"]

    assert "T8_uniform" in incomp
    assert incomp["T8_uniform"]["status"] == "NOT EVALUATED"
    assert "architecturally incompatible" in incomp["T8_uniform"]["reason"]

    assert "T32_uniform" in incomp
    assert incomp["T32_uniform"]["status"] == "NOT EVALUATED"
