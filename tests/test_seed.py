"""Tests for reproducibility and seed derivation utilities."""

import random
import numpy as np

from src.utils.seed import seed_everything, derive_video_seed


def test_seed_everything_determinism():
    """Verify seed_everything resets random and numpy generators consistently."""
    seed_everything(12345)
    r1 = [random.random() for _ in range(5)]
    n1 = np.random.rand(5).tolist()

    seed_everything(12345)
    r2 = [random.random() for _ in range(5)]
    n2 = np.random.rand(5).tolist()

    assert r1 == r2
    assert n1 == n2


def test_derive_video_seed_sha256_stability():
    """Verify derive_video_seed is deterministic across multiple evaluations."""
    s1 = derive_video_seed(42, "video_001.mp4")
    s2 = derive_video_seed(42, "video_001.mp4")
    assert s1 == s2
    assert isinstance(s1, int)
    assert 0 <= s1 < (2 ** 31)


def test_derive_video_seed_distinct_ids():
    """Verify distinct video IDs produce different seeds."""
    s1 = derive_video_seed(42, "video_001.mp4")
    s2 = derive_video_seed(42, "video_002.mp4")
    s3 = derive_video_seed(43, "video_001.mp4")

    assert s1 != s2
    assert s1 != s3
