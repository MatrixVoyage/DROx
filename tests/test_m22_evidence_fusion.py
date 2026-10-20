"""Unit tests for Milestone 22: Evidence Fusion & Decision Calibration Modules.

Verifies:
1. ConstrainedLinearFusion maintains convex weights (w >= 0, sum w = 1) and monotonicity.
2. CalibratedLogOddsFusion outputs valid probabilities in [0, 1].
3. RankAverageFusion produces normalized rank scores in [0, 1] immune to scale shifts.
4. HierarchicalGatedFusion routes high-confidence decisions and uses fallback appropriately.
5. derive_threshold_policies derives exact validation percentile thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evidence_fusion import (
    ConstrainedLinearFusion,
    CalibratedLogOddsFusion,
    RankAverageFusion,
    HierarchicalGatedFusion,
    derive_threshold_policies,
)


def test_constrained_linear_fusion_convexity_and_monotonicity():
    """Verify weights remain non-negative, sum to 1, and preserve ranking monotonicity."""
    # Synthetic validation logits: Model 1 and Model 2
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, size=n)
    # Model 1 has AUC ~0.8, Model 2 has AUC ~0.85
    z1 = labels * 1.5 + np.random.randn(n)
    z2 = labels * 2.0 + np.random.randn(n)
    logits = np.column_stack([z1, z2])

    fusion = ConstrainedLinearFusion(num_models=2)
    fusion.fit(logits, labels)

    assert len(fusion.weights) == 2
    assert all(w >= 0.0 for w in fusion.weights)
    assert np.isclose(np.sum(fusion.weights), 1.0)

    # Monotonicity check
    z_fused = fusion.predict_logits(logits)
    p_fused = fusion.predict_proba(logits)
    assert np.all(p_fused >= 0.0) and np.all(p_fused <= 1.0)
    # Higher logit must produce strictly higher probability
    assert np.all(np.diff(p_fused[np.argsort(z_fused)]) >= 0.0)


def test_calibrated_log_odds_fusion():
    """Verify log-odds combination fits and outputs calibrated probabilities."""
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, size=n)
    p1 = np.clip(labels * 0.7 + np.random.uniform(0.1, 0.3, size=n), 0.01, 0.99)
    p2 = np.clip(labels * 0.8 + np.random.uniform(0.05, 0.2, size=n), 0.01, 0.99)
    probas = np.column_stack([p1, p2])

    fusion = CalibratedLogOddsFusion(num_models=2)
    fusion.fit(probas, labels)

    p_fused = fusion.predict_proba(probas)
    assert p_fused.shape == (n,)
    assert np.all(p_fused >= 0.0) and np.all(p_fused <= 1.0)


def test_rank_average_fusion_scale_invariance():
    """Verify rank average fusion is scale-free and invariant to linear offsets."""
    np.random.seed(42)
    n = 100
    z1 = np.random.randn(n) * 2.0 + 5.0  # Large positive offset
    z2 = np.random.randn(n) * 0.5 - 3.0  # Negative offset

    fusion = RankAverageFusion(num_models=2)
    fusion.fit(np.column_stack([z1, z2]))

    ranks = fusion.predict_rank_score(np.column_stack([z1, z2]))
    assert ranks.shape == (n,)
    assert np.all(ranks >= 0.0) and np.all(ranks <= 1.0)

    # Shifting z1 by +100 should not change rank score
    z1_shifted = z1 + 100.0
    fusion_shifted = RankAverageFusion(num_models=2)
    fusion_shifted.fit(np.column_stack([z1_shifted, z2]))
    ranks_shifted = fusion_shifted.predict_rank_score(np.column_stack([z1_shifted, z2]))
    assert np.allclose(ranks, ranks_shifted)


def test_hierarchical_gated_fusion():
    """Verify hierarchical gating routes high-confidence and falls back appropriately."""
    fusion = HierarchicalGatedFusion(tau_high=0.90, tau_low=0.10)
    probas = np.array([
        [0.95, 0.50],  # Model 1 high confidence fake -> routed
        [0.05, 0.08],  # Both clean real -> routed
        [0.60, 0.70],  # Ambiguous -> fallback linear
    ])
    logits = np.array([[2.0, 0.0], [-2.5, -2.0], [0.4, 0.8]])

    p = fusion.predict_proba(probas, logits)
    assert p[0] == 0.95
    assert p[1] == 0.05
    assert 0.10 < p[2] < 0.90


def test_derive_threshold_policies():
    """Verify operational decision policies tau_95, tau_99, and tau_youden."""
    np.random.seed(42)
    scores = np.concatenate([
        np.random.uniform(0.0, 0.4, size=100),  # Reals
        np.random.uniform(0.6, 1.0, size=100),  # Fakes
    ])
    labels = np.array([0] * 100 + [1] * 100)

    policies = derive_threshold_policies(scores, labels)
    reals = scores[labels == 0]
    expected_tau_95 = float(np.percentile(reals, 95.0))
    expected_tau_99 = float(np.percentile(reals, 99.0))

    assert pytest.approx(policies.tau_95) == expected_tau_95
    assert pytest.approx(policies.tau_99) == expected_tau_99
    assert policies.tau_99 >= policies.tau_95
    assert 0.35 <= policies.tau_youden <= 0.65
