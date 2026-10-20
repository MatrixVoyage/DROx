"""Milestone 9 Gate 7: Statistical Significance, Video-Level Bootstrap & Paired Tests."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger("m9_statistics")


def bootstrap_video_level_metrics(
    probabilities: List[float],
    labels: List[int],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute non-parametric bootstrap confidence intervals at the VIDEO level."""
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64)
    n = len(probs)

    if n == 0 or len(set(y_true)) < 2:
        return {}

    rng = np.random.RandomState(random_seed)

    boot_acc = []
    boot_prec = []
    boot_rec = []
    boot_f1 = []
    boot_auc = []

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    for _ in range(n_bootstraps):
        # Resample video indices WITH REPLACEMENT
        idx = rng.randint(0, n, size=n)
        sample_probs = probs[idx]
        sample_true = y_true[idx]

        # Require both classes to compute valid AUC and classification metrics
        if len(set(sample_true)) < 2:
            continue

        sample_pred = (sample_probs >= threshold).astype(int)

        boot_acc.append(accuracy_score(sample_true, sample_pred))
        boot_prec.append(precision_score(sample_true, sample_pred, zero_division=0))
        boot_rec.append(recall_score(sample_true, sample_pred, zero_division=0))
        boot_f1.append(f1_score(sample_true, sample_pred, zero_division=0))

        try:
            boot_auc.append(roc_auc_score(sample_true, sample_probs))
        except ValueError:
            pass

    def get_ci_dict(data: List[float], point_val: float) -> Dict[str, float]:
        if not data:
            return {"point_estimate": point_val, "ci_lower": point_val, "ci_upper": point_val, "std_error": 0.0}
        low = float(np.percentile(data, lower_pct))
        high = float(np.percentile(data, upper_pct))
        se = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
        return {
            "point_estimate": round(float(point_val), 4),
            "ci_lower": round(low, 4),
            "ci_upper": round(high, 4),
            "std_error": round(se, 4),
        }

    # Point estimates on full dataset
    full_pred = (probs >= threshold).astype(int)
    p_acc = accuracy_score(y_true, full_pred)
    p_prec = precision_score(y_true, full_pred, zero_division=0)
    p_rec = recall_score(y_true, full_pred, zero_division=0)
    p_f1 = f1_score(y_true, full_pred, zero_division=0)
    p_auc = roc_auc_score(y_true, probs)

    return {
        "bootstrap_iterations": n_bootstraps,
        "valid_iterations": len(boot_acc),
        "confidence_level": confidence_level,
        "random_seed": random_seed,
        "resampling_unit": "video",
        "sample_size": n,
        "metrics": {
            "accuracy": get_ci_dict(boot_acc, p_acc),
            "precision": get_ci_dict(boot_prec, p_prec),
            "recall": get_ci_dict(boot_rec, p_rec),
            "f1": get_ci_dict(boot_f1, p_f1),
            "roc_auc": get_ci_dict(boot_auc, p_auc),
        },
    }


def compute_paired_mcnemar_test(
    baseline_predictions: List[int],
    alternative_predictions: List[int],
    ground_truth: List[int],
    baseline_name: str = "baseline",
    alternative_name: str = "alternative",
) -> Dict[str, Any]:
    """Execute paired McNemar test on identical sample set."""
    b_preds = np.asarray(baseline_predictions)
    a_preds = np.asarray(alternative_predictions)
    truth = np.asarray(ground_truth)

    if len(b_preds) != len(a_preds) or len(b_preds) != len(truth):
        raise ValueError("Sample sizes must match for paired McNemar test.")

    b_correct = (b_preds == truth)
    a_correct = (a_preds == truth)

    # Contingency Table:
    # a: both correct, b: baseline correct only, c: alternative correct only, d: both incorrect
    n_a = int(np.sum(b_correct & a_correct))
    n_b = int(np.sum(b_correct & (~a_correct)))
    n_c = int(np.sum((~b_correct) & a_correct))
    n_d = int(np.sum((~b_correct) & (~a_correct)))

    # McNemar test with continuity correction
    discordant = n_b + n_c
    if discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        # If discordant pairs are small (<25), use exact binomial distribution
        if discordant < 25:
            # Exact two-sided binomial test with p=0.5
            p_val = float(stats.binomtest(min(n_b, n_c), discordant, p=0.5).pvalue)
            stat = float((abs(n_b - n_c) - 1.0) ** 2 / discordant)
        else:
            stat = float((abs(n_b - n_c) - 1.0) ** 2 / discordant)
            p_val = float(stats.chi2.sf(stat, df=1))

    return {
        "comparison": f"{baseline_name} vs {alternative_name}",
        "sample_size": len(b_preds),
        "contingency_table": {
            "both_correct": n_a,
            "baseline_only_correct": n_b,
            "alternative_only_correct": n_c,
            "both_incorrect": n_d,
        },
        "test_type": "exact_binomial" if discordant < 25 else "chi_square_continuity_corrected",
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "is_significant_at_05": p_val < 0.05,
        "interpretation": (
            f"The paired classification difference between {baseline_name} and {alternative_name} "
            f"is {'statistically significant' if p_val < 0.05 else 'not statistically significant'} (p = {p_val:.4f})."
        ),
    }
