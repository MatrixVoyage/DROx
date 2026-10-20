"""Milestone 9 Gate 2: Score Distribution, Dataset Shift & Effect-Size Analysis."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger("m9_score_analysis")


def compute_group_statistics(scores: List[float]) -> Dict[str, float]:
    """Compute comprehensive descriptive summary statistics for a group of scores."""
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
        "median": round(float(np.median(arr)), 4),
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "iqr": round(q75 - q25, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def compute_cliffs_delta(x: List[float], y: List[float]) -> float:
    """Compute Cliff's delta non-parametric effect size between two score distributions.

    d in [-1, 1], where 0 indicates identical stochastic dominance.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n_x, n_y = len(x_arr), len(y_arr)
    if n_x == 0 or n_y == 0:
        return 0.0

    # Vectorized comparison
    diff = x_arr[:, None] - y_arr[None, :]
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)
    return float((greater - less) / (n_x * n_y))


def compute_cohens_d(x: List[float], y: List[float]) -> float:
    """Compute Cohen's d parametric effect size between two score distributions."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_x, n_y = len(x_arr), len(y_arr)
    if n_x < 2 or n_y < 2:
        return 0.0

    var_x = np.var(x_arr, ddof=1)
    var_y = np.var(y_arr, ddof=1)
    pooled_sd = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(x_arr) - np.mean(y_arr)) / pooled_sd)


def compute_distribution_distance(x: List[float], y: List[float]) -> Dict[str, Any]:
    """Compute distribution comparison metrics between two 1D score sets."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if len(x_arr) == 0 or len(y_arr) == 0:
        return {
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "wasserstein_distance": 0.0,
            "jensen_shannon_distance": 0.0,
            "cliffs_delta": 0.0,
            "cohens_d": 0.0,
            "interpretation": "Insufficient sample size for comparison.",
        }

    # Two-sample Kolmogorov-Smirnov test
    ks_res = stats.ks_2samp(x_arr, y_arr)

    # Wasserstein distance (Earth Mover's Distance)
    w_dist = stats.wasserstein_distance(x_arr, y_arr)

    # Jensen-Shannon distance on 20-bin histogram density
    bins = np.linspace(0.0, 1.0, 21)
    p_hist, _ = np.histogram(x_arr, bins=bins, density=True)
    q_hist, _ = np.histogram(y_arr, bins=bins, density=True)
    # Add epsilon to prevent log-zero
    p_prob = (p_hist + 1e-7) / np.sum(p_hist + 1e-7)
    q_prob = (q_hist + 1e-7) / np.sum(q_hist + 1e-7)
    from scipy.spatial.distance import jensenshannon
    js_dist = jensenshannon(p_prob, q_prob)

    c_delta = compute_cliffs_delta(x, y)
    c_d = compute_cohens_d(x, y)

    # Non-causal scientific interpretation
    interp = (
        f"The observed score distributions differ (Wasserstein={w_dist:.4f}, KS={ks_res.statistic:.4f}, "
        f"Cliff's delta={c_delta:.4f}), consistent with a domain-distribution difference across evaluated groups."
    )

    return {
        "ks_statistic": round(float(ks_res.statistic), 4),
        "ks_pvalue": float(ks_res.pvalue),
        "wasserstein_distance": round(float(w_dist), 4),
        "jensen_shannon_distance": round(float(js_dist), 4),
        "cliffs_delta": round(float(c_delta), 4),
        "cohens_d": round(float(c_d), 4),
        "interpretation": interp,
    }


def analyze_all_score_groups(
    ffpp_test_records: List[Dict[str, Any]],
    celebdf_test_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze score distributions and distribution shift metrics across datasets and families."""
    ffpp_real = [r["probability"] for r in ffpp_test_records if r["label"] == 0]
    ffpp_fake = [r["probability"] for r in ffpp_test_records if r["label"] == 1]

    celeb_real = [r["probability"] for r in celebdf_test_records if r["label"] == 0]
    celeb_fake = [r["probability"] for r in celebdf_test_records if r["label"] == 1]

    families = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    fam_scores = {
        fam: [r["probability"] for r in ffpp_test_records if r.get("manipulation_type") == fam]
        for fam in families
    }

    group_stats = {
        "ffpp_real": compute_group_statistics(ffpp_real),
        "ffpp_fake": compute_group_statistics(ffpp_fake),
        "celeb_real": compute_group_statistics(celeb_real),
        "celeb_fake": compute_group_statistics(celeb_fake),
    }
    for fam, sc in fam_scores.items():
        group_stats[f"ffpp_family_{fam.lower()}"] = compute_group_statistics(sc)

    comparisons = {
        "ffpp_real_vs_fake": compute_distribution_distance(ffpp_real, ffpp_fake),
        "celeb_real_vs_fake": compute_distribution_distance(celeb_real, celeb_fake),
        "ffpp_real_vs_celeb_real": compute_distribution_distance(ffpp_real, celeb_real),
        "ffpp_fake_vs_celeb_fake": compute_distribution_distance(ffpp_fake, celeb_fake),
    }

    return {
        "group_statistics": group_stats,
        "distribution_comparisons": comparisons,
    }


def plot_score_distributions(
    ffpp_test_records: List[Dict[str, Any]],
    celebdf_test_records: List[Dict[str, Any]],
    output_png: Path,
) -> None:
    """Generate publication-grade score distribution histogram and boxplots."""
    ffpp_real = [r["probability"] for r in ffpp_test_records if r["label"] == 0]
    ffpp_fake = [r["probability"] for r in ffpp_test_records if r["label"] == 1]
    celeb_real = [r["probability"] for r in celebdf_test_records if r["label"] == 0]
    celeb_fake = [r["probability"] for r in celebdf_test_records if r["label"] == 1]

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(14, 6))

    bins = np.linspace(0.0, 1.0, 26)
    ax_hist.hist(ffpp_real, bins=bins, alpha=0.6, density=True, label=f"FF++ Real (N={len(ffpp_real)})", color="#10b981")
    ax_hist.hist(ffpp_fake, bins=bins, alpha=0.6, density=True, label=f"FF++ Fake (N={len(ffpp_fake)})", color="#ef4444")
    ax_hist.hist(celeb_real, bins=bins, alpha=0.5, density=True, label=f"Celeb Real (N={len(celeb_real)})", color="#3b82f6", histtype="step", linewidth=2)
    ax_hist.hist(celeb_fake, bins=bins, alpha=0.5, density=True, label=f"Celeb Fake (N={len(celeb_fake)})", color="#f59e0b", histtype="step", linewidth=2)

    ax_hist.axvline(0.5, color="black", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
    ax_hist.set_xlabel("Predicted Probability (Fake Score)")
    ax_hist.set_ylabel("Empirical Density")
    ax_hist.set_title("Probability Score Distributions Across Benchmarks")
    ax_hist.legend(loc="upper left", fontsize=9)
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    # Boxplot
    box_data = [ffpp_real, ffpp_fake, celeb_real, celeb_fake]
    box_labels = ["FF++ Real", "FF++ Fake", "Celeb Real", "Celeb Fake"]
    bplot = ax_box.boxplot(box_data, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    ax_box.set_xticks(range(1, len(box_labels) + 1))
    ax_box.set_xticklabels(box_labels)
    colors = ["#a7f3d0", "#fecaca", "#bfdbfe", "#fde68a"]
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)

    ax_box.axhline(0.5, color="black", linestyle="--", alpha=0.7)
    ax_box.set_ylabel("Predicted Probability")
    ax_box.set_title("Score Quartiles & Outliers")
    ax_box.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_roc_curves(
    ffpp_test_records: List[Dict[str, Any]],
    celebdf_test_records: List[Dict[str, Any]],
    output_png: Path,
) -> None:
    """Generate ROC curves comparison across FF++ in-domain and Celeb-DF v2 cross-dataset."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # FF++
    ffpp_probs = [r["probability"] for r in ffpp_test_records]
    ffpp_labels = [r["label"] for r in ffpp_test_records]
    fpr_ff, tpr_ff, _ = roc_curve(ffpp_labels, ffpp_probs)
    auc_ff = stats.rankdata(ffpp_probs)  # or sklearn auc
    from sklearn.metrics import auc as sk_auc
    ax.plot(fpr_ff, tpr_ff, color="#ef4444", lw=2, label=f"FF++ C23 Test (AUC = {sk_auc(fpr_ff, tpr_ff):.4f})")

    # Celeb-DF
    celeb_probs = [r["probability"] for r in celebdf_test_records]
    celeb_labels = [r["label"] for r in celebdf_test_records]
    fpr_cel, tpr_cel, _ = roc_curve(celeb_labels, celeb_probs)
    ax.plot(fpr_cel, tpr_cel, color="#3b82f6", lw=2, label=f"Celeb-DF v2 Test (AUC = {sk_auc(fpr_cel, tpr_cel):.4f})")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess (AUC = 0.5000)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_pr_curves(
    ffpp_test_records: List[Dict[str, Any]],
    celebdf_test_records: List[Dict[str, Any]],
    output_png: Path,
) -> None:
    """Generate Precision-Recall curves comparison with prevalence baselines."""
    fig, ax = plt.subplots(figsize=(8, 7))
    from sklearn.metrics import auc as sk_auc

    # FF++
    ffpp_probs = [r["probability"] for r in ffpp_test_records]
    ffpp_labels = [r["label"] for r in ffpp_test_records]
    prec_ff, rec_ff, _ = precision_recall_curve(ffpp_labels, ffpp_probs)
    prev_ff = sum(ffpp_labels) / len(ffpp_labels)
    ax.plot(rec_ff, prec_ff, color="#ef4444", lw=2, label=f"FF++ C23 Test (PR-AUC = {sk_auc(rec_ff, prec_ff):.4f})")
    ax.axhline(prev_ff, color="#ef4444", linestyle=":", alpha=0.5, label=f"FF++ Prevalence ({prev_ff:.2f})")

    # Celeb-DF
    celeb_probs = [r["probability"] for r in celebdf_test_records]
    celeb_labels = [r["label"] for r in celebdf_test_records]
    prec_cel, rec_cel, _ = precision_recall_curve(celeb_labels, celeb_probs)
    prev_cel = sum(celeb_labels) / len(celeb_labels)
    ax.plot(rec_cel, prec_cel, color="#3b82f6", lw=2, label=f"Celeb-DF v2 Test (PR-AUC = {sk_auc(rec_cel, prec_cel):.4f})")
    ax.axhline(prev_cel, color="#3b82f6", linestyle=":", alpha=0.5, label=f"Celeb-DF Prevalence ({prev_cel:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (PR) Curves Comparison")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()
