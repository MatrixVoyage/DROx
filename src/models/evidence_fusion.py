"""Milestone 22: Decision-Level Evidence Fusion & Calibration Modules.

Zero Neural Network Retraining:
Combines frozen predictions from multiple forensic models (e.g., Champion M13.5
and Baseline M18 Dual-Stream) via lightweight, monotonic mathematical operations.

Fusion Strategies:
1. ConstrainedLinearFusion: w_1 * z_1 + w_2 * z_2 + b (w_i >= 0, sum w_i = 1)
2. CalibratedLogOddsFusion: sigma(w_1 * logit(p_1) + w_2 * logit(p_2) + b)
3. RankAverageFusion: 0.5 * F_val(z_1) + 0.5 * F_val(z_2) (non-parametric, scale-free)
4. HierarchicalGatedFusion: High-specificity threshold gating with fallback linear fusion
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch


@dataclass
class FusionThresholdPolicy:
    """Decision threshold policy derived on validation data."""
    tau_95: float = 0.50
    tau_99: float = 0.50
    tau_youden: float = 0.50
    tau_default: float = 0.50

    def to_dict(self) -> Dict[str, float]:
        return {
            "tau_95": float(self.tau_95),
            "tau_99": float(self.tau_99),
            "tau_youden": float(self.tau_youden),
            "tau_default": float(self.tau_default),
        }


class ConstrainedLinearFusion:
    """Monotonic convex combination of raw decision logits with learnable bias.

    z_fused = w_1 * z_1 + w_2 * z_2 + ... + w_K * z_K + b
    Subject to: w_k >= 0, sum(w_k) = 1.
    """

    def __init__(self, num_models: int = 2, weights: Optional[List[float]] = None, bias: float = 0.0):
        self.num_models = num_models
        if weights is not None:
            assert len(weights) == num_models
            w_arr = np.array(weights, dtype=np.float64)
            self.weights = w_arr / np.sum(w_arr)
        else:
            self.weights = np.ones(num_models, dtype=np.float64) / num_models
        self.bias = float(bias)

    def fit(self, logits_matrix: np.ndarray, labels: np.ndarray) -> "ConstrainedLinearFusion":
        """Fit weights and bias on validation set to minimize binary cross-entropy."""
        n_samples, n_models = logits_matrix.shape
        assert n_models == self.num_models

        # Initial parameters: [w_1, ..., w_K, b]
        # Reparameterize weights via softmax to strictly enforce w_k >= 0, sum w_k = 1
        def objective(params: np.ndarray) -> float:
            raw_w = params[:n_models]
            b = params[n_models]
            # Softmax weights
            w = np.exp(raw_w - np.max(raw_w))
            w = w / np.sum(w)

            z_fused = np.dot(logits_matrix, w) + b
            p = expit(z_fused)
            p = np.clip(p, 1e-7, 1.0 - 1e-7)

            bce = -np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
            return float(bce)

        init_params = np.zeros(n_models + 1, dtype=np.float64)
        init_params[:n_models] = 0.0  # Equal uniform initial weights
        init_params[n_models] = 0.0

        res = minimize(objective, init_params, method="L-BFGS-B")
        best_w = np.exp(res.x[:n_models] - np.max(res.x[:n_models]))
        self.weights = best_w / np.sum(best_w)
        self.bias = float(res.x[n_models])
        return self

    def predict_logits(self, logits_matrix: np.ndarray) -> np.ndarray:
        """Compute fused logits."""
        return np.dot(logits_matrix, self.weights) + self.bias

    def predict_proba(self, logits_matrix: np.ndarray) -> np.ndarray:
        """Compute fused probabilities via sigmoid."""
        z = self.predict_logits(logits_matrix)
        return expit(z)


class CalibratedLogOddsFusion:
    """Combines calibrated probabilities in log-odds space with Platt parameters.

    p_fused = sigma(w_1 * logit(p_1) + w_2 * logit(p_2) + b)
    """

    def __init__(self, num_models: int = 2, weights: Optional[List[float]] = None, bias: float = 0.0):
        self.num_models = num_models
        if weights is not None:
            assert len(weights) == num_models
            self.weights = np.array(weights, dtype=np.float64)
        else:
            self.weights = np.ones(num_models, dtype=np.float64) / num_models
        self.bias = float(bias)

    def fit(self, probas_matrix: np.ndarray, labels: np.ndarray) -> "CalibratedLogOddsFusion":
        """Fit log-odds combination weights and bias on validation probabilities."""
        n_samples, n_models = probas_matrix.shape
        clipped_p = np.clip(probas_matrix, 1e-5, 1.0 - 1e-5)
        log_odds = logit(clipped_p)

        def objective(params: np.ndarray) -> float:
            w = params[:n_models]
            b = params[n_models]
            z_fused = np.dot(log_odds, w) + b
            p = expit(z_fused)
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            bce = -np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
            # Ridge regularization on weights to prevent overfitting
            reg = 0.01 * np.sum(w ** 2)
            return float(bce + reg)

        init_params = np.zeros(n_models + 1, dtype=np.float64)
        init_params[:n_models] = 1.0 / n_models
        init_params[n_models] = 0.0

        res = minimize(objective, init_params, method="L-BFGS-B")
        self.weights = np.array(res.x[:n_models], dtype=np.float64)
        self.bias = float(res.x[n_models])
        return self

    def predict_proba(self, probas_matrix: np.ndarray) -> np.ndarray:
        clipped_p = np.clip(probas_matrix, 1e-5, 1.0 - 1e-5)
        log_odds = logit(clipped_p)
        z = np.dot(log_odds, self.weights) + self.bias
        return expit(z)


class RankAverageFusion:
    """Non-parametric rank average fusion using empirical CDFs fitted on validation.

    Immune to logit scaling differences, score offsets, and outlier distortions.
    """

    def __init__(self, num_models: int = 2):
        self.num_models = num_models
        self.ecdfs: List[np.ndarray] = []

    def fit(self, logits_matrix: np.ndarray) -> "RankAverageFusion":
        """Record sorted validation logits to evaluate empirical CDF."""
        n_samples, n_models = logits_matrix.shape
        assert n_models == self.num_models
        self.ecdfs = []
        for m in range(n_models):
            self.ecdfs.append(np.sort(logits_matrix[:, m]))
        return self

    def predict_rank_score(self, logits_matrix: np.ndarray) -> np.ndarray:
        """Compute average empirical percentile rank across models."""
        n_samples, n_models = logits_matrix.shape
        ranks = np.zeros((n_samples, n_models), dtype=np.float64)

        for m in range(n_models):
            sorted_ref = self.ecdfs[m]
            # searchsorted returns index where x would be inserted
            pos = np.searchsorted(sorted_ref, logits_matrix[:, m], side="right")
            ranks[:, m] = pos / len(sorted_ref)

        return np.mean(ranks, axis=1)


class HierarchicalGatedFusion:
    """High-Specificity Gated Fusion routing.

    Leverages Champion M13.5's high clean specificity for confident decisions,
    falling back to M18 kinematic integration for ambiguous cases.
    """

    def __init__(
        self,
        tau_high: float = 0.95,
        tau_low: float = 0.05,
        fallback_fusion: Optional[ConstrainedLinearFusion] = None,
    ):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.fallback = fallback_fusion or ConstrainedLinearFusion(num_models=2)

    def predict_proba(self, probas_matrix: np.ndarray, logits_matrix: np.ndarray) -> np.ndarray:
        p_m13 = probas_matrix[:, 0]
        p_m18 = probas_matrix[:, 1]
        n = len(p_m13)
        final_p = np.zeros(n, dtype=np.float64)

        fallback_p = self.fallback.predict_proba(logits_matrix)

        for i in range(n):
            if p_m13[i] >= self.tau_high:
                # High confidence fake from champion
                final_p[i] = p_m13[i]
            elif p_m13[i] <= self.tau_low and p_m18[i] <= self.tau_low:
                # Both models agree clean real
                final_p[i] = min(p_m13[i], p_m18[i])
            else:
                # Ambiguous region: use calibrated linear fusion
                final_p[i] = fallback_p[i]

        return final_p


def derive_threshold_policies(scores: np.ndarray, labels: np.ndarray) -> FusionThresholdPolicy:
    """Derive operational decision thresholds strictly on validation data."""
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    # Specificity = 95% threshold: 95th percentile of real scores
    tau_95 = float(np.percentile(real_scores, 95.0))

    # Specificity = 99% threshold: 99th percentile of real scores
    tau_99 = float(np.percentile(real_scores, 99.0))

    # Youden's J optimal threshold
    all_thresholds = np.unique(scores)
    if len(all_thresholds) > 200:
        all_thresholds = np.percentile(scores, np.linspace(0, 100, 200))

    best_j = -1.0
    best_tau_y = 0.50
    for tau in all_thresholds:
        sens = np.mean(fake_scores >= tau)
        spec = np.mean(real_scores < tau)
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_tau_y = float(tau)

    return FusionThresholdPolicy(
        tau_95=tau_95,
        tau_99=tau_99,
        tau_youden=best_tau_y,
        tau_default=0.50,
    )
