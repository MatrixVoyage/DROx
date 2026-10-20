"""Unit tests for Milestone 9 score analysis and distribution shift metrics."""

import numpy as np
import pytest

from src.evaluation.score_analysis import (
    compute_cliffs_delta,
    compute_cohens_d,
    compute_distribution_distance,
    compute_group_statistics,
)


def test_group_statistics():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = compute_group_statistics(scores)

    assert stats["count"] == 9
    assert np.isclose(stats["mean"], 0.5, atol=1e-3)
    assert np.isclose(stats["median"], 0.5, atol=1e-3)
    assert stats["min"] == 0.1
    assert stats["max"] == 0.9
    assert stats["q25"] == 0.3
    assert stats["q75"] == 0.7
    assert np.isclose(stats["iqr"], 0.4, atol=1e-3)


def test_cliffs_delta():
    # If all x are strictly greater than all y, delta = 1.0
    x = [0.8, 0.9, 0.95]
    y = [0.1, 0.2, 0.3]
    delta = compute_cliffs_delta(x, y)
    assert np.isclose(delta, 1.0, atol=1e-4)

    # Identical sets -> delta = 0.0
    assert np.isclose(compute_cliffs_delta(x, x), 0.0, atol=1e-4)


def test_cohens_d():
    x = [10.0, 11.0, 12.0]
    y = [1.0, 2.0, 3.0]
    d = compute_cohens_d(x, y)
    assert d > 0.0


def test_distribution_distance():
    x = [0.2, 0.3, 0.25, 0.22, 0.28]
    y = [0.7, 0.8, 0.75, 0.72, 0.78]

    res = compute_distribution_distance(x, y)
    assert "ks_statistic" in res
    assert "wasserstein_distance" in res
    assert "jensen_shannon_distance" in res
    assert "interpretation" in res
    assert res["wasserstein_distance"] > 0.4
    assert res["ks_statistic"] > 0.5
    # Verify non-causal language in interpretation
    assert "consistent with a domain-distribution difference" in res["interpretation"]
