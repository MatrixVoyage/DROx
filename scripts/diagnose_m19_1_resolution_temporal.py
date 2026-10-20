"""Milestone 19.1: Resolution & Temporal Sampling Attribution Suite.

Directly tests Hypotheses H2 (Spatial Localization), H3 (Temporal Context),
and H7 (Pixel-Resolution Ceiling) on the frozen M18 Dual-Stream checkpoint
without any model retraining.

Evaluates the exact same Face2Face locked holdout (116 videos: 58 pristine reals, 58 F2F fakes)
across the 3x3 core matrix + intermediate resolution points:
- Spatial Resolution (R): 224, 160, 112 (baseline), 96, 64
- Temporal Sampling (T): 8, 16 (baseline), 32 frames
- Paired bootstrap statistical comparison against the 112x16 baseline.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.m18_detector import M18ReenactmentDetector
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("diagnose_m19_1_resolution_temporal")

# Checkpoints
M17_TRACK_A_CKPT = PROJECT_ROOT / "checkpoints" / "m17_track_a_seed42" / "best.pth"
M18_DUAL_STREAM_CKPT = PROJECT_ROOT / "checkpoints" / "m18_dual_stream_seed42" / "best.pth"

# Manifests
MANIFEST_MIXED = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"

# Preprocessed Tensor Directories
TENSOR_DIRS = [
    PROJECT_ROOT / "processed" / "m12_full",
    PROJECT_ROOT / "processed" / "m13_3_lofo",
    PROJECT_ROOT / "processed" / "celebdf_v2" / "test" / "tensors",
]

OUTPUT_JSON = PROJECT_ROOT / "reports" / "m19_1_resolution_temporal_attribution.json"
OUTPUT_MD = PROJECT_ROOT / "reports" / "m19_1_resolution_temporal_attribution.md"


def find_tensor(video_id: str, split: str = "test") -> Optional[Path]:
    """Find preprocessed tensor on disk."""
    candidates = [
        TENSOR_DIRS[0] / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[2] / f"{video_id}.pt",
        TENSOR_DIRS[0] / "val" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[0] / "train" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / "val" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / "train" / "tensors" / f"{video_id}.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def modulate_temporal_sampling(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """Modulate temporal frame count of sequence tensor.
    
    Args:
        x: Tensor of shape (B, T_in, C, H, W).
        target_t: Target number of frames (8, 16, or 32).
        
    Returns:
        Tensor of shape (B, target_t, C, H, W).
    """
    B, T_in, C, H, W = x.shape
    if target_t == T_in:
        return x
    
    if target_t == 8:
        # Uniform 8-frame sub-sampling (halving temporal frame rate)
        idx = [0, 2, 4, 6, 8, 10, 12, 14]
        return x[:, idx]
    elif target_t == 32:
        # Trilinear temporal super-sampling (doubling temporal sampling density)
        # Permute to (B, C, T, H, W) for 3D interpolation
        x_5d = x.permute(0, 2, 1, 3, 4)
        x_interp = F.interpolate(
            x_5d, size=(32, H, W), mode="trilinear", align_corners=False
        )
        return x_interp.permute(0, 2, 1, 3, 4)
    else:
        # General linear temporal resampling
        x_5d = x.permute(0, 2, 1, 3, 4)
        x_interp = F.interpolate(
            x_5d, size=(target_t, H, W), mode="trilinear", align_corners=False
        )
        return x_interp.permute(0, 2, 1, 3, 4)


def modulate_spatial_resolution(x: torch.Tensor, target_r: int) -> torch.Tensor:
    """Modulate spatial resolution bandwidth via downsampling and restoration.
    
    Args:
        x: Tensor of shape (B, T, C, H, W) where H=W=112.
        target_r: Target spatial resolution (e.g. 64, 96, 112, 160, 224).
        
    Returns:
        Bandwidth-modulated tensor of shape (B, T, C, 112, 112).
    """
    if target_r == 112:
        return x
    
    B, T, C, H, W = x.shape
    flat = x.view(B * T, C, H, W)
    down = F.interpolate(flat, size=(target_r, target_r), mode="bilinear", align_corners=False)
    back = F.interpolate(down, size=(112, 112), mode="bilinear", align_corners=False)
    return back.view(B, T, C, 112, 112)


def compute_paired_bootstrap(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_ref: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired bootstrap difference in ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_cand = roc_auc_score(y_b, scores_cand[idx])
        auc_ref = roc_auc_score(y_b, scores_ref[idx])
        deltas.append(auc_cand - auc_ref)
    deltas = np.array(deltas)
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
        "p_value": float(np.mean(deltas <= 0.0)),
    }


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's delta non-parametric effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for val_x in x:
        greater += int(np.sum(val_x > y))
        less += int(np.sum(val_x < y))
    return float((greater - less) / (n_x * n_y))


def load_m18_model(device: torch.device) -> M18ReenactmentDetector:
    """Load the frozen M18 Dual-Stream model."""
    cfg = load_config("configs/experiments/m12_full.yaml")
    base_cand = DeepfakeDetector(config=cfg.model).to(device)
    m17_ckpt = torch.load(M17_TRACK_A_CKPT, map_location=device, weights_only=False)
    model_track_a = M17TrackADetector(
        base_detector=base_cand,
        num_families=5,
        res_dim=128,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        gate_bottleneck=128,
    ).to(device)
    model_track_a.load_state_dict(m17_ckpt["model_state_dict"])
    model_track_a.eval()

    m18_ckpt = torch.load(M18_DUAL_STREAM_CKPT, map_location=device, weights_only=False)
    m18_dual = M18ReenactmentDetector(
        trunk_detector=model_track_a,
        kin_dim=256,
        aux_hidden_dim=512,
        gate_bottleneck=128,
        freeze_trunk=True,
    ).to(device)
    m18_dual.load_state_dict(m18_ckpt["model_state_dict"])
    m18_dual.eval()
    return m18_dual


def evaluate_f2f_under_condition(
    f2f_items: List[Tuple[Path, int]],
    model: M18ReenactmentDetector,
    device: torch.device,
    target_r: int,
    target_t: int,
) -> Dict[str, Any]:
    """Evaluate all 116 Face2Face holdout clips under (R, T) condition."""
    logits_list = []
    probs_list = []
    gates_list = []
    labels_list = []

    with torch.no_grad():
        for tp, label in f2f_items:
            t = torch.load(tp, map_location=device, weights_only=False)
            if t.dim() == 4:
                t = t.unsqueeze(0)  # (1, 16, 3, 112, 112)
            
            # 1. Modulate temporal frames
            t_mod = modulate_temporal_sampling(t, target_t)
            
            # 2. Modulate spatial resolution
            t_mod = modulate_spatial_resolution(t_mod, target_r)
            
            # 3. Model forward pass
            with torch.amp.autocast("cuda"):
                out = model(t_mod, mode="dual_stream")
                
            logits_list.append(float(out.primary_logits.item()))
            probs_list.append(float(out.p_fused.item()))
            gates_list.append(float(out.gate_values.item()))
            labels_list.append(int(label))
            del t, t_mod, out

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    z_arr = np.array(logits_list)
    p_arr = np.array(probs_list)
    g_arr = np.array(gates_list)
    y_arr = np.array(labels_list)

    reals_mask = y_arr == 0
    fakes_mask = y_arr == 1

    auc = float(roc_auc_score(y_arr, z_arr))
    prauc = float(average_precision_score(y_arr, z_arr))

    r_z_med = float(np.median(z_arr[reals_mask]))
    f_z_med = float(np.median(z_arr[fakes_mask]))
    logit_margin = f_z_med - r_z_med

    r_g_med = float(np.median(g_arr[reals_mask]))
    f_g_med = float(np.median(g_arr[fakes_mask]))

    # Operational metrics at tau=0.50 (z=0)
    spec_05 = float(np.mean(p_arr[reals_mask] < 0.50))
    sens_05 = float(np.mean(p_arr[fakes_mask] >= 0.50))

    return {
        "target_r": target_r,
        "target_t": target_t,
        "auc": auc,
        "prauc": prauc,
        "real_median_logit": r_z_med,
        "fake_median_logit": f_z_med,
        "logit_margin": logit_margin,
        "real_median_gate": r_g_med,
        "fake_median_gate": f_g_med,
        "spec_at_tau05": spec_05,
        "sens_at_tau05": sens_05,
        "scores": z_arr,
        "labels": y_arr,
    }


def main():
    logger.info("Initializing Milestone 19.1: Resolution & Temporal Sampling Attribution Suite...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    m18_dual = load_m18_model(device)
    logger.info("Frozen M18 Dual-Stream model loaded cleanly on %s", device)

    # 2. Load Locked Face2Face Holdout (116 clips: 58 pristine reals, 58 F2F fakes)
    mixed_df = pd.read_csv(MANIFEST_MIXED)
    mixed_test = mixed_df[mixed_df["split"] == "test"].copy().reset_index(drop=True)
    clean_reals = mixed_test[mixed_test["manipulation_type"] == "original"]
    f2f_fakes = mixed_test[mixed_test["manipulation_type"] == "Face2Face"]

    f2f_items: List[Tuple[Path, int]] = []
    for _, r in clean_reals.iterrows():
        tp = find_tensor(str(r["video_id"]), split="test")
        if tp is not None:
            f2f_items.append((tp, 0))

    for _, r in f2f_fakes.iterrows():
        tp = find_tensor(str(r["video_id"]), split="test")
        if tp is not None:
            f2f_items.append((tp, 1))

    logger.info("Loaded %d Face2Face holdout clips (58 Reals, 58 F2F Fakes)", len(f2f_items))
    assert len(f2f_items) == 116, f"Expected 116 clips, got {len(f2f_items)}"

    # 3. Define the Evaluation Matrix
    # Core 3x3 matrix:
    core_matrix = [
        (224, 8), (224, 16), (224, 32),
        (112, 8), (112, 16), (112, 32),
        (64, 8),  (64, 16),  (64, 32),
    ]
    # Intermediate resolution points at baseline T=16:
    extra_points = [(160, 16), (96, 16)]
    all_conditions = core_matrix + extra_points

    # 4. Run Baseline Condition (112x112, 16 frames)
    logger.info("\n=== Evaluating Baseline Condition: R=112, T=16 ===")
    baseline_res = evaluate_f2f_under_condition(f2f_items, m18_dual, device, target_r=112, target_t=16)
    logger.info("  Baseline AUC: %.4f | Real Logit: %.3f | Fake Logit: %.3f | Margin: %+.3f",
                baseline_res["auc"], baseline_res["real_median_logit"],
                baseline_res["fake_median_logit"], baseline_res["logit_margin"])

    # 5. Run All Matrix Conditions
    results: Dict[str, Any] = {}
    for r, t in all_conditions:
        cond_key = f"R{r}_T{t}"
        logger.info("\n--- Evaluating Condition: R=%d, T=%d ---", r, t)
        t0 = time.perf_counter()
        res = evaluate_f2f_under_condition(f2f_items, m18_dual, device, target_r=r, target_t=t)
        elapsed = time.perf_counter() - t0

        # Paired bootstrap vs baseline (112x16)
        boot = compute_paired_bootstrap(
            baseline_res["labels"],
            res["scores"],
            baseline_res["scores"],
            n_resamples=1000,
            seed=42,
        )

        c_delta = compute_cliffs_delta(res["scores"], baseline_res["scores"])

        logger.info("  AUC: %.4f | Delta AUC: %+.4f [95%% CI: %.4f, %.4f] (p=%.4f) | Elapsed: %.2fs",
                    res["auc"], boot["delta_mean"], boot["ci_lower"], boot["ci_upper"],
                    boot["p_value"], elapsed)

        results[cond_key] = {
            "resolution": r,
            "frames": t,
            "auc": res["auc"],
            "prauc": res["prauc"],
            "delta_auc_vs_baseline": boot["delta_mean"],
            "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"],
            "p_val_vs_baseline": boot["p_value"],
            "cliffs_delta_vs_baseline": c_delta,
            "real_median_logit": res["real_median_logit"],
            "fake_median_logit": res["fake_median_logit"],
            "logit_margin": res["logit_margin"],
            "real_median_gate": res["real_median_gate"],
            "fake_median_gate": res["fake_median_gate"],
            "spec_at_tau05": res["spec_at_tau05"],
            "sens_at_tau05": res["sens_at_tau05"],
        }
        torch.cuda.empty_cache()
        gc.collect()

    # 6. Synthesize Hypotheses & Scenarios
    # Scenario A: Resolution dominates (e.g. 64 -> 112 -> 224 spans > +0.08 AUC)
    auc_64_16 = results["R64_T16"]["auc"]
    auc_112_16 = results["R112_T16"]["auc"]
    auc_224_16 = results["R224_T16"]["auc"]
    res_spread = auc_224_16 - auc_64_16

    # Scenario B: Temporal context dominates (e.g. 8 -> 16 -> 32 spans > +0.05 AUC)
    auc_112_8 = results["R112_T8"]["auc"]
    auc_112_32 = results["R112_T32"]["auc"]
    temp_spread = auc_112_32 - auc_112_8

    logger.info("\n=== Attribution Synthesis ===")
    logger.info("  Resolution Gradient (64 -> 112 -> 224 @ T=16): %.4f -> %.4f -> %.4f (Spread: %+.4f)",
                auc_64_16, auc_112_16, auc_224_16, res_spread)
    logger.info("  Temporal Gradient   (8 -> 16 -> 32 @ R=112):   %.4f -> %.4f -> %.4f (Spread: %+.4f)",
                auc_112_8, auc_112_16, auc_112_32, temp_spread)

    if res_spread >= 0.05 and temp_spread < 0.03:
        scenario = "Scenario A: Resolution Dominates"
        synthesis = "Resolution is the primary bottleneck. Micro-deformation cues are attenuated by low spatial resolution. M20 should focus on high-resolution localized representation."
    elif temp_spread >= 0.05 and res_spread < 0.03:
        scenario = "Scenario B: Temporal Context Dominates"
        synthesis = "Temporal context is the primary bottleneck. Motion duration captures subtle phase shifts. M20 should focus on longer-context temporal dynamics."
    elif res_spread >= 0.05 and temp_spread >= 0.05:
        scenario = "Scenario C: Both Spatial Resolution and Temporal Context Matter"
        synthesis = "Both spatial bandwidth and temporal duration contribute complementary gains. M20 should combine high-resolution localized facial regions + longer temporal hierarchy."
    else:
        scenario = "Scenario D: Neither Spatial Resolution nor Temporal Sampling Move AUC Significantly"
        synthesis = "Neither resolution upscaling nor frame expansion overcomes the reenactment ceiling. Confirms H1: First-order kinematic frame differencing inherently lacks non-rigid optical warp representation, and pixel residuals in compressed C23 video are depleted. M20 must NOT add further RGB differencing, but formulate localized deformable patch representations or multimodal evidence expansion."

    final_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": "M19.1",
        "title": "Resolution and Temporal Sampling Attribution Analysis",
        "model_analyzed": "M18 Dual-Stream Detector (Frozen, Seed 42)",
        "dataset_analyzed": "Face2Face Locked Holdout (N=116: 58 Reals, 58 Fakes)",
        "baseline_auc": baseline_res["auc"],
        "results": results,
        "synthesis": {
            "scenario": scenario,
            "resolution_spread_at_t16": res_spread,
            "temporal_spread_at_r112": temp_spread,
            "interpretation": synthesis,
            "h2_status": "Supported" if res_spread > 0.03 else "Weak / Inconsequential",
            "h3_status": "Supported" if temp_spread > 0.03 else "Weak / Inconsequential",
            "h7_status": "Refuted as Primary Ceiling (Degradation is modest)" if auc_64_16 >= 0.70 else "Strongly Supported (Resolution collapse observed)",
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    logger.info("Saved complete M19.1 JSON to %s", OUTPUT_JSON)

    write_m19_1_markdown(final_payload)
    logger.info("Saved complete M19.1 Markdown report to %s", OUTPUT_MD)


def write_m19_1_markdown(data: Dict[str, Any]) -> None:
    """Format and write structured M19.1 Markdown report."""
    res = data["results"]
    syn = data["synthesis"]

    lines = [
        "# Milestone 19.1 — Resolution & Temporal Sampling Attribution Report",
        "",
        f"**Date**: {data['timestamp'][:10]}  ",
        f"**Target Model**: Frozen M18 Dual-Stream Detector (`checkpoints/m18_dual_stream_seed42/best.pth`)  ",
        f"**Dataset**: Face2Face Locked Holdout (58 Pristine Reals, 58 Face2Face Fakes, Total N=116)  ",
        "**Methodology**: Zero-retraining evaluation across 3x3 core resolution-temporal matrix + intermediate resolution steps. Paired bootstrap test against the 112×16 baseline.  ",
        "",
        "---",
        "",
        "## 1. Core 3×3 Resolution-Temporal Matrix Results",
        "",
        "| Resolution ($R$) | Temporal Frames ($T$) | ROC-AUC | PR-AUC | $\\Delta$ AUC vs. 112×16 | 95% Bootstrap CI | Paired $p$-value | Cliff's $\\delta$ | Logit Margin |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in [224, 112, 64]:
        for t in [8, 16, 32]:
            k = f"R{r}_T{t}"
            entry = res[k]
            is_base = (r == 112 and t == 16)
            delta_str = "Baseline Reference" if is_base else f"{entry['delta_auc_vs_baseline']:+.4f}"
            ci_str = "—" if is_base else f"[{entry['ci_lower']:+.4f}, {entry['ci_upper']:+.4f}]"
            p_str = "—" if is_base else f"{entry['p_val_vs_baseline']:.4f}"
            delta_cd = "—" if is_base else f"{entry['cliffs_delta_vs_baseline']:+.3f}"

            lines.append(
                f"| **{r}×{r}** | **{t} frames** | **{entry['auc']:.4f}** | {entry['prauc']:.4f} | "
                f"{delta_str} | {ci_str} | {p_str} | {delta_cd} | {entry['logit_margin']:+.3f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Complete Resolution Gradient at Baseline $T = 16$ Frames",
        "",
        "| Spatial Resolution ($R$) | ROC-AUC | $\\Delta$ AUC vs. 112×16 | 95% Bootstrap CI | Real Median Logit | Fake Median Logit | Logit Margin |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in [224, 160, 112, 96, 64]:
        k = f"R{r}_T16"
        entry = res[k]
        is_base = (r == 112)
        delta_str = "Baseline (112×16)" if is_base else f"{entry['delta_auc_vs_baseline']:+.4f}"
        ci_str = "—" if is_base else f"[{entry['ci_lower']:+.4f}, {entry['ci_upper']:+.4f}]"

        lines.append(
            f"| **{r}×{r}** | **{entry['auc']:.4f}** | {delta_str} | {ci_str} | "
            f"{entry['real_median_logit']:.3f} | {entry['fake_median_logit']:.3f} | {entry['logit_margin']:+.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Complete Temporal Gradient at Baseline Resolution $R = 112$",
        "",
        "| Temporal Sampling ($T$) | ROC-AUC | $\\Delta$ AUC vs. 112×16 | 95% Bootstrap CI | Real Median Logit | Fake Median Logit | Logit Margin |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for t in [8, 16, 32]:
        k = f"R112_T{t}"
        entry = res[k]
        is_base = (t == 16)
        delta_str = "Baseline (112×16)" if is_base else f"{entry['delta_auc_vs_baseline']:+.4f}"
        ci_str = "—" if is_base else f"[{entry['ci_lower']:+.4f}, {entry['ci_upper']:+.4f}]"

        lines.append(
            f"| **{t} frames** | **{entry['auc']:.4f}** | {delta_str} | {ci_str} | "
            f"{entry['real_median_logit']:.3f} | {entry['fake_median_logit']:.3f} | {entry['logit_margin']:+.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Scientific Hypothesis Attribution Verdict",
        "",
        f"- **Empirical Scenario**: **`{syn['scenario']}`**",
        f"- **Resolution AUC Spread ($224 - 64$ at $T=16$)**: **`{syn['resolution_spread_at_t16']:+.4f}`**",
        f"- **Temporal AUC Spread ($32 - 8$ at $R=112$)**: **`{syn['temporal_spread_at_r112']:+.4f}`**",
        f"- **Hypothesis H2 (Spatial Localization)**: `{syn['h2_status']}`",
        f"- **Hypothesis H3 (Temporal Context)**: `{syn['h3_status']}`",
        f"- **Hypothesis H7 (Pixel-Resolution Ceiling)**: `{syn['h7_status']}`",
        "",
        "### Interpretation & Architectural Guidance for M20",
        f"> {syn['interpretation']}",
        "",
        "---",
        "",
        "## 5. Frozen Governance Declarations",
        "",
        "- **M19.1 Resolution/Temporal Attribution**: **COMPLETED & FROZEN**",
        "- **M13.5 (`v5-family-invariant`)**: **ACTIVE PRODUCTION CHAMPION** (Undefeated)",
        "- **M17 Track A (`Sensor/Residual`)**: **FROZEN RESEARCH BASELINE**",
        "- **M18 Dual-Stream (`Sensor + Kinematic`)**: **FROZEN RESEARCH CANDIDATE**",
    ])

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
