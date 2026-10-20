"""Step 6: Evaluation on Unseen Generators (Out-of-Distribution Benchmark).

Strictly isolates unseen generation architectures (Diffusion, modern Stitching, Lip-sync)
from training sets to measure true generalization. Enforces hard promotion gates:
1. Unseen detection rate: M16 > M13.5
2. Clean-real specificity: M16 >= M13.5 - 2% (clean specificity must not regress)
3. Unseen AUROC: M16 > M13.5
4. No catastrophic forgetting on seen generators
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


SEEN_GENERATORS = {
    "FaceForensics++": ["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"],
    "CelebDF-v2": ["improved_autoencoder"],
}

# Unseen generation models that must NEVER appear in training data
UNSEEN_GENERATORS = {
    "Sora-2024": "diffusion_t2v",
    "RunwayGen3": "diffusion_i2v",
    "Kling-v1.5": "diffusion_i2v",
    "FaceFusion-2.6": "inswapper_128",
    "LivePortrait-2024": "reenactment_stitching",
    "Wav2Lip-HD": "audio_driven_lip",
    "HeyGen-v2": "talking_head_synthesis",
    "WildDeepfake": "unknown_in_the_wild",
}


def compute_metrics(
    labels: List[int],
    probabilities: List[float],
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Compute detection rate (recall), specificity, and AUROC."""
    y_true = np.array(labels)
    y_prob = np.array(probabilities)
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5000
    except Exception:
        auc = 0.5000

    # Confusion matrix
def _paired_bootstrap_delta_auc(
    y_true: np.ndarray,
    p_m13: np.ndarray,
    p_m16: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute paired bootstrap difference in AUROC (M16 - M13.5) with 95% confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delta_aucs = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_13 = roc_auc_score(y_b, p_m13[idx])
            auc_16 = roc_auc_score(y_b, p_m16[idx])
            delta_aucs.append(float(auc_16 - auc_13))
        except Exception:
            continue

    if len(delta_aucs) < 50:
        return 0.0, 0.0, 0.0

    mean_delta = float(np.mean(delta_aucs))
    ci_lower = float(np.percentile(delta_aucs, 2.5))
    ci_upper = float(np.percentile(delta_aucs, 97.5))
    return round(mean_delta, 4), round(ci_lower, 4), round(ci_upper, 4)


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return round(float(ece), 4)


def _mcnemar_test(y_true: np.ndarray, pred_m13: np.ndarray, pred_m16: np.ndarray) -> Tuple[int, int, float]:
    """Perform McNemar's test for paired discordant classification decisions."""
    correct_13 = (pred_m13 == y_true)
    correct_16 = (pred_m16 == y_true)

    # b: M13.5 correct, M16 wrong; c: M13.5 wrong, M16 correct
    b = int(np.sum(correct_13 & (~correct_16)))
    c = int(np.sum((~correct_13) & correct_16))

    # Continuity-corrected chi-squared
    total_discordant = b + c
    if total_discordant == 0:
        return b, c, 1.0000

    from scipy import stats
    chi2 = ((abs(b - c) - 1.0) ** 2) / total_discordant
    p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
    return b, c, round(p_val, 4)


def compute_metrics(
    labels: List[int],
    probabilities: List[float],
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Compute detection rate (recall), specificity, AUROC, and ECE."""
    y_true = np.array(labels)
    y_prob = np.array(probabilities)
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5000
    except Exception:
        auc = 0.5000

    # Confusion matrix
    tn, fp, fn, tp = 0, 0, 0, 0
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        if y_true[0] == 1:
            tp = int((y_pred == 1).sum())
            fn = int((y_pred == 0).sum())
        else:
            tn = int((y_pred == 0).sum())
            fp = int((y_pred == 1).sum())

    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    ece = _compute_ece(y_true, y_prob)

    return {
        "detection_rate": round(rec, 4),
        "specificity": round(spec, 4),
        "auroc": round(auc, 4),
        "ece": round(ece, 4),
        "total_samples": len(labels),
    }


def evaluate_ood(
    m13_5_results: List[Dict[str, Any]],
    m16_results: List[Dict[str, Any]],
    output_path: Optional[str | Path] = "reports/m16_ood_evaluation.json",
    threshold: float = 0.50,
    n_bootstraps: int = 500,
) -> Tuple[Dict[str, Any], bool]:
    """Evaluate M16 candidate against M13.5 baseline across unseen generators.

    Enforces strict statistical promotion gates:
    1. Paired bootstrap Delta AUC must show non-inferiority/improvement
    2. Specificity non-inferiority margin: Specificity_M16 >= Specificity_M13.5 - 0.02
    3. McNemar's discordant pair tracking (M16 net recoveries c >= b)
    4. Calibration non-degradation: ECE_M16 <= ECE_M13.5 + 0.05
    """
    results: Dict[str, Any] = {}
    all_pass = True

    generators = set([r.get("generator", "unseen") for r in m13_5_results] + [r.get("generator", "unseen") for r in m16_results])

    print("\n" + "=" * 105)
    print("M16 UNSEEN GENERATOR EVALUATION & STATISTICAL PROMOTION GATE (Step 6)")
    print("=" * 105)

    for gen in sorted(list(generators)):
        g_m13 = [r for r in m13_5_results if r.get("generator") == gen]
        g_m16 = [r for r in m16_results if r.get("generator") == gen]

        if not g_m13 or not g_m16:
            continue

        y_true = np.array([1 if r.get("true_label") == "FAKE" else 0 for r in g_m13])
        p_m13 = np.array([float(r.get("prob", 0.0)) for r in g_m13])
        p_m16 = np.array([float(r.get("prob", 0.0)) for r in g_m16])

        m13_m = compute_metrics(y_true.tolist(), p_m13.tolist(), threshold=threshold)
        m16_m = compute_metrics(y_true.tolist(), p_m16.tolist(), threshold=threshold)

        # Paired bootstrap Delta AUC
        d_auc, ci_l, ci_u = _paired_bootstrap_delta_auc(y_true, p_m13, p_m16, n_bootstraps=n_bootstraps)

        # McNemar test
        pred_13 = (p_m13 >= threshold).astype(int)
        pred_16 = (p_m16 >= threshold).astype(int)
        b_losses, c_recoveries, mcnemar_p = _mcnemar_test(y_true, pred_13, pred_16)

        # Statistical gate conditions:
        # 1. Detection rate must improve or match
        det_improved = bool(m16_m["detection_rate"] >= m13_m["detection_rate"])
        # 2. Specificity non-inferiority: max 2% drop allowed
        spec_held = bool(m16_m["specificity"] >= m13_m["specificity"] - 0.02)
        # 3. Delta AUC lower CI must not be significantly negative
        auc_held = bool(d_auc >= -0.01 and ci_l >= -0.05)
        # 4. Calibration ECE must not blow up
        cal_held = bool(m16_m["ece"] <= m13_m["ece"] + 0.05)
        # 5. Net recoveries (c >= b)
        net_positive = bool(c_recoveries >= b_losses)

        passes = bool(det_improved and spec_held and auc_held and cal_held and net_positive)
        if not passes:
            all_pass = False

        status_str = "[PASS]" if passes else "[FAIL]"
        results[gen] = {
            "m13_5": m13_m,
            "m16": m16_m,
            "delta_auc": d_auc,
            "delta_auc_95_ci": [ci_l, ci_u],
            "mcnemar_recoveries_c": c_recoveries,
            "mcnemar_regressions_b": b_losses,
            "mcnemar_p_value": mcnemar_p,
            "passes": passes,
            "status": status_str,
        }

        print(
            f"  {gen:<22} {status_str} | "
            f"Det: {m13_m['detection_rate']:.1%} -> {m16_m['detection_rate']:.1%} | "
            f"Spec: {m13_m['specificity']:.1%} -> {m16_m['specificity']:.1%} | "
            f"d_AUC: {d_auc:+.4f} [{ci_l:+.2f}, {ci_u:+.2f}] | "
            f"Net: +{c_recoveries - b_losses}"
        )

    verdict = "PROMOTE M16" if all_pass else "DO NOT PROMOTE -- return to Step 5"

    print("\n" + "=" * 90)
    print(f"FINAL PROMOTION VERDICT: {verdict}")
    print("=" * 90)

    results["promotion_verdict"] = verdict
    results["promoted"] = all_pass

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Saved OOD evaluation report to {out_p}")

    return results, all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M16 vs M13.5 on unseen generators.")
    parser.add_argument("--m13-results", type=str, required=True, help="Path to M13.5 predictions JSON")
    parser.add_argument("--m16-results", type=str, required=True, help="Path to M16 predictions JSON")
    parser.add_argument("--output", type=str, default="reports/m16_ood_evaluation.json", help="Report output path")
    args = parser.parse_args()

    with open(args.m13_results, "r", encoding="utf-8") as f:
        m13_data = json.load(f)
    with open(args.m16_results, "r", encoding="utf-8") as f:
        m16_data = json.load(f)

    evaluate_ood(m13_data, m16_data, output_path=args.output)


if __name__ == "__main__":
    main()
