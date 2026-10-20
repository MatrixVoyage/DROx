"""Comprehensive Evaluation Script for Milestone 13.4: Calibration, Threshold & Multi-Seed Stability.

Executes:
- Phase A: Multi-Seed Stability Evaluation (seeds 42, 1337, 2026) with mean +/- SD/SE.
- Phase B: Validation-Constrained Threshold Policy (tau search on Val -> evaluated on Test).
- Phase C: Validation-Only Calibration (Temperature Scaling & Isotonic Calibration, ECE, Brier, Score Percentiles).
- Phase D: 7-Way Branch Ablation (Temporal, Freq, Texture permutations).
- Integrity: Verification of frozen M12 baseline hash before and after execution.

Usage:
    .venv\\Scripts\\python.exe scripts/evaluate_m13_4_stability.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProcessedVideoDataset
from src.models.detector import DeepfakeDetector
from src.models.multi_head_detector import MultiHeadDeepfakeDetector
from src.models.family_invariant_detector import (
    FamilyInvariantDeepfakeDetector,
    FamilyInvariantOutput,
)
from src.utils.config import ModelConfig, _dict_to_config
from src.utils.logger import setup_logger

logger = setup_logger("m13_4_stability")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
FREEZE_MANIFEST_PATH = PROJECT_ROOT / "reports" / "m12_baseline_freeze.json"
M13_2B_STAGE5_CKPT = PROJECT_ROOT / "checkpoints" / "m13_2b_stage5" / "best.pth"
PROCESSED_DIR = PROJECT_ROOT / "processed" / "m12_full"

REPORT_JSON = PROJECT_ROOT / "reports" / "m13_4_stability_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m13_4_stability_report.md"

SEEDS = [42, 1337, 2026]


def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_m12_integrity() -> str:
    """Verify that M12 baseline best.pth has not been modified."""
    with open(FREEZE_MANIFEST_PATH, "r") as f:
        freeze_data = json.load(f)
    expected_hash = freeze_data["fingerprints"]["m12_checkpoint"]["sha256"]
    current_hash = sha256_file(M12_CHECKPOINT_PATH)
    assert current_hash == expected_hash, (
        f"CRITICAL: M12 checkpoint modified! {current_hash} != {expected_hash}"
    )
    logger.info(f"M12 integrity verified. Hash: {current_hash[:16]}...")
    return current_hash


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        mask = (y_prob > b_low) & (y_prob <= b_high) if i > 0 else (y_prob >= b_low) & (y_prob <= b_high)
        b_n = np.sum(mask)
        if b_n > 0:
            b_acc = np.mean(y_true[mask])
            b_conf = np.mean(y_prob[mask])
            ece += (b_n / n) * abs(b_acc - b_conf)
    return float(ece)


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a temperature scaling parameter T > 0 on validation logits using BCE."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    log_temp = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.01, max_iter=100)
    criterion = nn.BCEWithLogitsLoss()

    def eval_loss():
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        loss = criterion(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    fitted_t = float(torch.exp(log_temp).item())
    return max(0.01, fitted_t)


def build_datasets(manifest_path: Path, processed_dir: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load validation manifest and test benchmark slices."""
    df = pd.read_csv(manifest_path)

    # Validation DataFrame (filtered to existing tensors)
    val_raw = df[df["split"] == "val"].copy()
    val_keep = []
    for _, row in val_raw.iterrows():
        p = processed_dir / "val" / "tensors" / f"{row['video_id']}.pt"
        if p.exists():
            val_keep.append(row)
    val_df = pd.DataFrame(val_keep).reset_index(drop=True)

    # Test Benchmark Slices
    test_df = df[df["split"] == "test"].copy()
    def filter_test(subset: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for _, row in subset.iterrows():
            p = processed_dir / "test" / "tensors" / f"{row['video_id']}.pt"
            if p.exists():
                keep.append(row)
        return pd.DataFrame(keep).reset_index(drop=True)

    clean_real = filter_test(test_df[test_df["label"] == 0])
    seen_fake = filter_test(test_df[test_df["manipulation_type"].isin(["Deepfakes", "FaceSwap", "NeuralTextures"])])
    celeb_df = filter_test(test_df[test_df["dataset"] == "celebdf"])
    f2f = filter_test(test_df[test_df["manipulation_type"] == "Face2Face"])

    test_slices = {
        "clean_real": clean_real,
        "seen_fake": seen_fake,
        "celeb_df": celeb_df,
        "face2face": f2f,
    }

    logger.info(
        f"Loaded datasets: Val={len(val_df)} | Test Slices: Clean={len(clean_real)}, "
        f"Seen={len(seen_fake)}, CelebDF={len(celeb_df)}, F2F={len(f2f)}"
    )
    return val_df, test_slices


def run_inference_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str = "invariant",
) -> Dict[str, np.ndarray]:
    """Run forward pass and return probabilities, logits, targets, and head breakdown."""
    model.eval()
    all_probs = []
    all_logits = []
    all_targets = []
    all_p_prim = []
    all_p_freq = []
    all_p_tex = []

    with torch.no_grad():
        for batch in loader:
            tensors, targets, vids = batch
            tensors = tensors.to(device, dtype=torch.float32)

            if model_type == "m12":
                logits = model(tensors).cpu().numpy().flatten()
                probs = 1.0 / (1.0 + np.exp(-logits))
                p_prim = probs
                p_freq = probs
                p_tex = probs
            elif model_type == "multi_head":
                out = model(tensors, return_dict=True)
                probs = out.p_fused.cpu().numpy().flatten()
                logits = out.primary_logits.cpu().numpy().flatten()
                p_prim = torch.sigmoid(out.primary_logits).cpu().numpy().flatten()
                p_freq = probs
                p_tex = probs
            elif model_type == "invariant":
                out: FamilyInvariantOutput = model(tensors, return_dict=True)
                probs = out.p_fused.cpu().numpy().flatten()
                logits = out.primary_logits.cpu().numpy().flatten()
                p_prim = torch.sigmoid(out.primary_logits).cpu().numpy().flatten()
                p_freq = out.p_freq.cpu().numpy().flatten()
                p_tex = out.p_texture.cpu().numpy().flatten()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            all_probs.extend(probs.tolist())
            all_logits.extend(logits.tolist())
            all_targets.extend(targets.numpy().flatten().tolist())
            all_p_prim.extend(p_prim.tolist())
            all_p_freq.extend(p_freq.tolist())
            all_p_tex.extend(p_tex.tolist())

    return {
        "probs": np.array(all_probs),
        "logits": np.array(all_logits),
        "targets": np.array(all_targets),
        "p_primary": np.array(all_p_prim),
        "p_freq": np.array(all_p_freq),
        "p_texture": np.array(all_p_tex),
    }


def evaluate_benchmark_slices(
    model: nn.Module,
    model_type: str,
    test_slices: Dict[str, pd.DataFrame],
    processed_dir: Path,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate a model on the 4 test benchmark slices and compute metrics."""
    out = {}
    for s_name, s_df in test_slices.items():
        ds = ProcessedVideoDataset(manifest=s_df, split="test", processed_dir=processed_dir)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        out[s_name] = run_inference_on_loader(model, loader, device, model_type=model_type)

    # Clean Real Metrics
    cr_probs = out["clean_real"]["probs"]
    cr_n = len(cr_probs)
    cr_false_alarms = int(np.sum(cr_probs >= 0.5))
    cr_spec = float(np.mean(cr_probs < 0.5)) if cr_n > 0 else 0.0

    # Held-Out Face2Face Metrics
    f2f_probs = out["face2face"]["probs"]
    f2f_n = len(f2f_probs)
    f2f_det = int(np.sum(f2f_probs >= 0.5))
    f2f_det_rate = float(np.mean(f2f_probs >= 0.5)) if f2f_n > 0 else 0.0

    # Face2Face Ranking against Clean Real
    y_eval = np.concatenate([np.zeros(cr_n), np.ones(f2f_n)])
    p_eval = np.concatenate([cr_probs, f2f_probs])
    f2f_roc_auc = float(roc_auc_score(y_eval, p_eval))
    f2f_pr_auc = float(average_precision_score(y_eval, p_eval))

    # Celeb-DF Cross-Domain AUC
    celeb_y = out["celeb_df"]["targets"]
    celeb_p = out["celeb_df"]["probs"]
    celeb_auc = float(roc_auc_score(celeb_y, celeb_p)) if len(np.unique(celeb_y)) > 1 else 0.5

    return {
        "clean_specificity": cr_spec,
        "clean_false_alarms": cr_false_alarms,
        "clean_total": cr_n,
        "f2f_det_rate": f2f_det_rate,
        "f2f_detected": f2f_det,
        "f2f_total": f2f_n,
        "f2f_roc_auc": f2f_roc_auc,
        "f2f_pr_auc": f2f_pr_auc,
        "celeb_df_auc": celeb_auc,
        "raw_outputs": out,
        "y_eval": y_eval,
        "p_eval": p_eval,
    }


def main():
    logger.info("=" * 80)
    logger.info("Milestone 13.4: Calibration, Threshold & Multi-Seed Stability Audit")
    logger.info("=" * 80)

    # 1. Pre-execution M12 integrity check
    verify_m12_integrity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Load datasets
    val_df, test_slices = build_datasets(MANIFEST_PATH, PROCESSED_DIR)

    # 2. Load Frozen M12 baseline
    ckpt_m12 = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    m12_cfg = _dict_to_config(ckpt_m12["config"]) if isinstance(ckpt_m12["config"], dict) else ckpt_m12["config"]
    m12_model = DeepfakeDetector(config=m12_cfg.model).to(device)
    m12_model.load_state_dict(ckpt_m12["model_state_dict"])
    m12_res = evaluate_benchmark_slices(m12_model, "m12", test_slices, PROCESSED_DIR, device)
    logger.info(
        f"M12 Baseline: Clean Spec={m12_res['clean_specificity']*100:.2f}%, "
        f"F2F Det={m12_res['f2f_detected']:02d}/{m12_res['f2f_total']:02d} ({m12_res['f2f_det_rate']*100:.2f}%), "
        f"F2F AUC={m12_res['f2f_roc_auc']:.4f}, Celeb-DF={m12_res['celeb_df_auc']:.4f}"
    )

    # 3. Load Principal Champion Baseline (M13.2b-Stage5)
    ckpt_13_2b = torch.load(M13_2B_STAGE5_CKPT, map_location=device, weights_only=False)
    m13_2b_model = MultiHeadDeepfakeDetector(
        base_detector=DeepfakeDetector(config=m12_cfg.model).to(device),
        num_families=5,
        aux_hidden_dim=512,
        enable_mid_head=True,
        mid_in_channels=112,
        mid_hidden_dim=128,
    ).to(device)
    m13_2b_model.load_state_dict(ckpt_13_2b["model_state_dict"])
    m13_2b_res = evaluate_benchmark_slices(m13_2b_model, "multi_head", test_slices, PROCESSED_DIR, device)
    logger.info(
        f"M13.2b Champion: Clean Spec={m13_2b_res['clean_specificity']*100:.2f}%, "
        f"F2F Det={m13_2b_res['f2f_detected']:02d}/{m13_2b_res['f2f_total']:02d} ({m13_2b_res['f2f_det_rate']*100:.2f}%), "
        f"F2F AUC={m13_2b_res['f2f_roc_auc']:.4f}, Celeb-DF={m13_2b_res['celeb_df_auc']:.4f}"
    )

    # =========================================================================
    # PHASE A: Multi-Seed Stability (Seeds 42, 1337, 2026)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE A: Multi-Seed Stability Evaluation")
    logger.info("=" * 60)

    seed_results = {}
    for seed in SEEDS:
        ckpt_path = PROJECT_ROOT / "checkpoints" / f"m13_3_f2f_seed{seed}" / "best.pth"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint for seed {seed} at {ckpt_path} not found. Skipping.")
            continue
        logger.info(f"Loading candidate checkpoint: seed {seed} from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cand_model = FamilyInvariantDeepfakeDetector(
            base_detector=DeepfakeDetector(config=m12_cfg.model).to(device),
            num_families=5,
            aux_hidden_dim=512,
            freq_dim=64,
            texture_dim=64,
            fusion_dim=512,
            enable_mid_head=True,
            mid_in_channels=112,
            mid_hidden_dim=128,
        ).to(device)
        cand_model.load_state_dict(ckpt["model_state_dict"])
        s_res = evaluate_benchmark_slices(cand_model, "invariant", test_slices, PROCESSED_DIR, device)
        seed_results[seed] = {
            "model": cand_model,
            "metrics": s_res,
            "ckpt_hash": sha256_file(ckpt_path),
        }
        logger.info(
            f"Seed {seed}: Clean Spec={s_res['clean_specificity']*100:.2f}%, "
            f"F2F Det={s_res['f2f_detected']:02d}/{s_res['f2f_total']:02d} ({s_res['f2f_det_rate']*100:.2f}%), "
            f"F2F AUC={s_res['f2f_roc_auc']:.4f}, PR-AUC={s_res['f2f_pr_auc']:.4f}, Celeb-DF={s_res['celeb_df_auc']:.4f}"
        )

    # Compute multi-seed mean +/- SD/SE
    completed_seeds = list(seed_results.keys())
    multi_seed_stats = {}
    if len(completed_seeds) > 0:
        for metric_key in ["clean_specificity", "f2f_det_rate", "f2f_roc_auc", "f2f_pr_auc", "celeb_df_auc"]:
            vals = [seed_results[s]["metrics"][metric_key] for s in completed_seeds]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_v = float(sd_v / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            multi_seed_stats[metric_key] = {
                "mean": mean_v,
                "sd": sd_v,
                "se": se_v,
                "values": vals,
            }

    # =========================================================================
    # PHASE B: Threshold Analysis (Validation-Constrained)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE B: Validation-Constrained Threshold Policy")
    logger.info("=" * 60)

    # Use seed 42 (or first available seed) as the representative model
    primary_seed = completed_seeds[0] if len(completed_seeds) > 0 else 42
    rep_model = seed_results[primary_seed]["model"] if primary_seed in seed_results else None

    threshold_policies = {}
    if rep_model is not None:
        # Run inference on validation dataset
        val_ds = ProcessedVideoDataset(manifest=val_df, split="val", processed_dir=PROCESSED_DIR)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
        val_out = run_inference_on_loader(rep_model, val_loader, device, model_type="invariant")

        val_probs = val_out["probs"]
        val_targets = val_out["targets"]
        val_clean_mask = (val_targets == 0)

        # Threshold grid search
        tau_grid = np.linspace(0.01, 0.99, 99)
        # Policy 1: Constrained Spec >= 90% on Validation Clean Reals
        tau_90 = 0.5
        for t in tau_grid:
            spec_t = np.mean(val_probs[val_clean_mask] < t)
            if spec_t >= 0.90:
                tau_90 = float(t)
                break

        # Policy 2: Constrained Spec >= 95% on Validation Clean Reals
        tau_95 = 0.5
        for t in tau_grid:
            spec_t = np.mean(val_probs[val_clean_mask] < t)
            if spec_t >= 0.95:
                tau_95 = float(t)
                break

        # Policy 3: Youden's J on Validation Split
        fpr_v, tpr_v, th_v = roc_curve(val_targets, val_probs)
        j_scores = tpr_v - fpr_v
        best_j_idx = np.argmax(j_scores)
        tau_youden = float(th_v[best_j_idx])

        logger.info(
            f"Validation-Derived Thresholds: Tau@Spec90% = {tau_90:.4f}, "
            f"Tau@Spec95% = {tau_95:.4f}, Tau@Youden = {tau_youden:.4f}"
        )

        # Apply thresholds to untouched Test Benchmark
        test_cr_probs = seed_results[primary_seed]["metrics"]["raw_outputs"]["clean_real"]["probs"]
        test_f2f_probs = seed_results[primary_seed]["metrics"]["raw_outputs"]["face2face"]["probs"]

        for pol_name, tau_val in [
            ("Standard (tau=0.50)", 0.50),
            ("Spec-Constrained >= 90%", tau_90),
            ("Spec-Constrained >= 95%", tau_95),
            ("Validation Youden J", tau_youden),
        ]:
            cr_spec_test = float(np.mean(test_cr_probs < tau_val))
            cr_fa_test = int(np.sum(test_cr_probs >= tau_val))
            f2f_det_test = int(np.sum(test_f2f_probs >= tau_val))
            f2f_rate_test = float(np.mean(test_f2f_probs >= tau_val))

            threshold_policies[pol_name] = {
                "tau": float(tau_val),
                "test_clean_spec": cr_spec_test,
                "test_clean_false_alarms": cr_fa_test,
                "test_f2f_detected": f2f_det_test,
                "test_f2f_det_rate": f2f_rate_test,
            }
            logger.info(
                f"  Policy [{pol_name:<24} | tau={tau_val:.4f}]: "
                f"Clean Spec={cr_spec_test*100:.2f}% ({len(test_cr_probs)-cr_fa_test:02d}/{len(test_cr_probs):02d}), "
                f"F2F Det={f2f_rate_test*100:.2f}% ({f2f_det_test:02d}/{len(test_f2f_probs):02d})"
            )

    # =========================================================================
    # PHASE C: Validation-Only Calibration
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE C: Validation-Only Calibration & Score Distribution")
    logger.info("=" * 60)

    calibration_results = {}
    if rep_model is not None:
        # Fit calibration strictly on validation data
        val_logits = val_out["logits"]
        fitted_temp = fit_temperature_scaling(val_logits, val_targets)
        logger.info(f"Fitted Validation Temperature: T = {fitted_temp:.4f}")

        # Isotonic Regression on validation
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(val_probs, val_targets)

        # Apply to Test Benchmark
        y_test_f2f = seed_results[primary_seed]["metrics"]["y_eval"]
        p_raw = seed_results[primary_seed]["metrics"]["p_eval"]

        # 1. Raw uncalibrated metrics
        ece_raw = compute_ece(y_test_f2f, p_raw)
        brier_raw = float(brier_score_loss(y_test_f2f, p_raw))

        # 2. Temperature scaled
        # Scale test primary logits by T
        test_cr_raw = seed_results[primary_seed]["metrics"]["raw_outputs"]["clean_real"]
        test_f2f_raw = seed_results[primary_seed]["metrics"]["raw_outputs"]["face2face"]
        cr_logits_t = test_cr_raw["logits"] / fitted_temp
        f2f_logits_t = test_f2f_raw["logits"] / fitted_temp
        p_cr_temp = 0.5 * (1.0 / (1.0 + np.exp(-cr_logits_t))) + 0.5 * test_cr_raw["p_freq"]
        p_f2f_temp = 0.5 * (1.0 / (1.0 + np.exp(-f2f_logits_t))) + 0.5 * test_f2f_raw["p_freq"]
        p_temp = np.concatenate([p_cr_temp, p_f2f_temp])

        ece_temp = compute_ece(y_test_f2f, p_temp)
        brier_temp = float(brier_score_loss(y_test_f2f, p_temp))

        # 3. Isotonic calibrated
        p_iso = iso_reg.predict(p_raw)
        ece_iso = compute_ece(y_test_f2f, p_iso)
        brier_iso = float(brier_score_loss(y_test_f2f, p_iso))

        # Probability percentiles for slices
        def compute_percentiles(arr: np.ndarray) -> Dict[str, float]:
            return {
                "p10": float(np.percentile(arr, 10)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "mean": float(np.mean(arr)),
            }

        seen_probs = seed_results[primary_seed]["metrics"]["raw_outputs"]["seen_fake"]["probs"]
        dist_cr = compute_percentiles(test_cr_probs)
        dist_seen = compute_percentiles(seen_probs)
        dist_f2f = compute_percentiles(test_f2f_probs)

        calibration_results = {
            "fitted_temperature": fitted_temp,
            "raw": {"ece": ece_raw, "brier": brier_raw},
            "temp_scaled": {"ece": ece_temp, "brier": brier_temp},
            "isotonic": {"ece": ece_iso, "brier": brier_iso},
            "score_distributions": {
                "clean_real": dist_cr,
                "seen_fake": dist_seen,
                "held_out_f2f": dist_f2f,
            },
        }

        logger.info(f"Calibration on Test Benchmark: Raw ECE={ece_raw:.4f}, Temp-Scaled ECE={ece_temp:.4f}, Isotonic ECE={ece_iso:.4f}")
        logger.info(f"Brier Scores: Raw={brier_raw:.4f}, Temp-Scaled={brier_temp:.4f}, Isotonic={brier_iso:.4f}")
        logger.info(f"Distributions (Median p50): Clean Real={dist_cr['p50']:.4f} | Seen Fake={dist_seen['p50']:.4f} | F2F={dist_f2f['p50']:.4f}")

    # =========================================================================
    # PHASE D: Systematic 7-Way Branch Ablation
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE D: Systematic 7-Way Branch Ablation")
    logger.info("=" * 60)

    branch_ablation_results = {}
    if rep_model is not None:
        cr_raw = seed_results[primary_seed]["metrics"]["raw_outputs"]["clean_real"]
        f2f_raw = seed_results[primary_seed]["metrics"]["raw_outputs"]["face2face"]

        y_eval_abl = np.concatenate([np.zeros(len(cr_raw["probs"])), np.ones(len(f2f_raw["probs"]))])

        # 7 Representation Permutations
        permutations = {
            "1. Temporal Only": (cr_raw["p_primary"], f2f_raw["p_primary"]),
            "2. Frequency Only": (cr_raw["p_freq"], f2f_raw["p_freq"]),
            "3. Texture Only": (cr_raw["p_texture"], f2f_raw["p_texture"]),
            "4. Temporal + Frequency": (
                0.5 * cr_raw["p_primary"] + 0.5 * cr_raw["p_freq"],
                0.5 * f2f_raw["p_primary"] + 0.5 * f2f_raw["p_freq"],
            ),
            "5. Temporal + Texture": (
                0.5 * cr_raw["p_primary"] + 0.5 * cr_raw["p_texture"],
                0.5 * f2f_raw["p_primary"] + 0.5 * f2f_raw["p_texture"],
            ),
            "6. Frequency + Texture": (
                0.5 * cr_raw["p_freq"] + 0.5 * cr_raw["p_texture"],
                0.5 * f2f_raw["p_freq"] + 0.5 * f2f_raw["p_texture"],
            ),
            "7. Full Multi-Domain Fusion": (cr_raw["probs"], f2f_raw["probs"]),
        }

        print(f"{'Branch Permutation':<28} | {'Clean Spec':<10} | {'F2F Det (tau=0.5)':<18} | {'F2F ROC-AUC':<12} | {'F2F PR-AUC':<11}")
        print("-" * 88)
        for name, (cr_p, f2f_p) in permutations.items():
            p_comb = np.concatenate([cr_p, f2f_p])
            spec_val = float(np.mean(cr_p < 0.5))
            det_cnt = int(np.sum(f2f_p >= 0.5))
            det_rate = float(np.mean(f2f_p >= 0.5))
            auc_val = float(roc_auc_score(y_eval_abl, p_comb))
            prauc_val = float(average_precision_score(y_eval_abl, p_comb))

            branch_ablation_results[name] = {
                "clean_specificity": spec_val,
                "f2f_detected": det_cnt,
                "f2f_total": len(f2f_p),
                "f2f_det_rate": det_rate,
                "f2f_roc_auc": auc_val,
                "f2f_pr_auc": prauc_val,
            }
            print(
                f"{name:<28} | {spec_val*100:6.2f}%    | "
                f"{det_rate*100:5.2f}% ({det_cnt:02d}/{len(f2f_p):02d})     | "
                f"{auc_val:.4f}       | {prauc_val:.4f}"
            )
        print("-" * 88)

    # Post-execution integrity check
    verify_m12_integrity()

    # Save JSON Report
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "m12_hash": sha256_file(M12_CHECKPOINT_PATH),
        "m13_2b_hash": sha256_file(M13_2B_STAGE5_CKPT),
        "completed_seeds": completed_seeds,
        "m12_baseline": {
            "clean_specificity": m12_res["clean_specificity"],
            "f2f_det_rate": m12_res["f2f_det_rate"],
            "f2f_detected": m12_res["f2f_detected"],
            "f2f_total": m12_res["f2f_total"],
            "f2f_roc_auc": m12_res["f2f_roc_auc"],
            "f2f_pr_auc": m12_res["f2f_pr_auc"],
            "celeb_df_auc": m12_res["celeb_df_auc"],
        },
        "m13_2b_champion": {
            "clean_specificity": m13_2b_res["clean_specificity"],
            "f2f_det_rate": m13_2b_res["f2f_det_rate"],
            "f2f_detected": m13_2b_res["f2f_detected"],
            "f2f_total": m13_2b_res["f2f_total"],
            "f2f_roc_auc": m13_2b_res["f2f_roc_auc"],
            "f2f_pr_auc": m13_2b_res["f2f_pr_auc"],
            "celeb_df_auc": m13_2b_res["celeb_df_auc"],
        },
        "phase_a_multi_seed": {
            "seeds": {s: seed_results[s]["metrics"] for s in completed_seeds},
            "summary_stats": multi_seed_stats,
        },
        "phase_b_threshold_policies": threshold_policies,
        "phase_c_calibration": calibration_results,
        "phase_d_branch_ablation": branch_ablation_results,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    logger.info(f"Saved M13.4 JSON report to {REPORT_JSON}")

    # Save Markdown Report
    md_content = f"""# Milestone 13.4: Calibration, Threshold & Multi-Seed Stability Report

**Evaluation Timestamp**: `{report_data['timestamp']}`  
**M12 Baseline Hash**: `{report_data['m12_hash']}` (Immutable Reference)  
**M13.2b Champion Hash**: `{report_data['m13_2b_hash']}` (Principal Comparison Baseline)  

---

## 1. Phase A: Multi-Seed Stability (LOFO-F2F Benchmark)

| Model / Seed | Clean Specificity | Face2Face Det (tau=0.5) | Face2Face ROC-AUC | Face2Face PR-AUC | Celeb-DF Cross-Domain AUC |
|---|---|---|---|---|---|
| **M12-Robust (Frozen)** | {m12_res['clean_specificity']*100:.2f}% ({m12_res['clean_total']-m12_res['clean_false_alarms']:02d}/{m12_res['clean_total']:02d}) | {m12_res['f2f_det_rate']*100:.2f}% ({m12_res['f2f_detected']:02d}/{m12_res['f2f_total']:02d}) | {m12_res['f2f_roc_auc']:.4f} | {m12_res['f2f_pr_auc']:.4f} | {m12_res['celeb_df_auc']:.4f} |
| **M13.2b-Stage5 (Champion)** | {m13_2b_res['clean_specificity']*100:.2f}% ({m13_2b_res['clean_total']-m13_2b_res['clean_false_alarms']:02d}/{m13_2b_res['clean_total']:02d}) | {m13_2b_res['f2f_det_rate']*100:.2f}% ({m13_2b_res['f2f_detected']:02d}/{m13_2b_res['f2f_total']:02d}) | {m13_2b_res['f2f_roc_auc']:.4f} | {m13_2b_res['f2f_pr_auc']:.4f} | {m13_2b_res['celeb_df_auc']:.4f} |
"""
    for s in completed_seeds:
        sm = seed_results[s]["metrics"]
        md_content += f"| **M13.3-F2F-Seed{s}** | {sm['clean_specificity']*100:.2f}% ({sm['clean_total']-sm['clean_false_alarms']:02d}/{sm['clean_total']:02d}) | {sm['f2f_det_rate']*100:.2f}% ({sm['f2f_detected']:02d}/{sm['f2f_total']:02d}) | {sm['f2f_roc_auc']:.4f} | {sm['f2f_pr_auc']:.4f} | {sm['celeb_df_auc']:.4f} |\n"

    if len(completed_seeds) > 1 and "f2f_roc_auc" in multi_seed_stats:
        md_content += f"""
### Cross-Seed Summary (N={len(completed_seeds)} Seeds):
- **Clean Specificity**: `{multi_seed_stats['clean_specificity']['mean']*100:.2f}% ± {multi_seed_stats['clean_specificity']['sd']*100:.2f}%` (SE: `{multi_seed_stats['clean_specificity']['se']*100:.2f}%`)
- **Face2Face Det Rate**: `{multi_seed_stats['f2f_det_rate']['mean']*100:.2f}% ± {multi_seed_stats['f2f_det_rate']['sd']*100:.2f}%` (SE: `{multi_seed_stats['f2f_det_rate']['se']*100:.2f}%`)
- **Face2Face ROC-AUC**: `{multi_seed_stats['f2f_roc_auc']['mean']:.4f} ± {multi_seed_stats['f2f_roc_auc']['sd']:.4f}` (SE: `{multi_seed_stats['f2f_roc_auc']['se']:.4f}`)
- **Face2Face PR-AUC**: `{multi_seed_stats['f2f_pr_auc']['mean']:.4f} ± {multi_seed_stats['f2f_pr_auc']['sd']:.4f}` (SE: `{multi_seed_stats['f2f_pr_auc']['se']:.4f}`)
- **Celeb-DF AUC**: `{multi_seed_stats['celeb_df_auc']['mean']:.4f} ± {multi_seed_stats['celeb_df_auc']['sd']:.4f}` (SE: `{multi_seed_stats['celeb_df_auc']['se']:.4f}`)
"""

    md_content += "\n---\n\n## 2. Phase B: Validation-Constrained Threshold Policy\n\n"
    md_content += "| Threshold Policy | Operating Threshold (tau) | Test Clean Specificity | Test Face2Face Detection |\n"
    md_content += "|---|---|---|---|\n"
    for pol_name, p_data in threshold_policies.items():
        md_content += f"| **{pol_name}** | `tau = {p_data['tau']:.4f}` | {p_data['test_clean_spec']*100:.2f}% (False Alarms: {p_data['test_clean_false_alarms']:02d}) | {p_data['test_f2f_det_rate']*100:.2f}% ({p_data['test_f2f_detected']:02d}/58) |\n"

    md_content += "\n---\n\n## 3. Phase C: Calibration Diagnostics & Probability Distributions\n\n"
    if calibration_results:
        md_content += f"- **Fitted Validation Temperature**: `T = {calibration_results['fitted_temperature']:.4f}`\n"
        md_content += f"- **Raw Test ECE / Brier**: `{calibration_results['raw']['ece']:.4f}` / `{calibration_results['raw']['brier']:.4f}`\n"
        md_content += f"- **Temperature Scaled ECE / Brier**: `{calibration_results['temp_scaled']['ece']:.4f}` / `{calibration_results['temp_scaled']['brier']:.4f}`\n"
        md_content += f"- **Isotonic Calibrated ECE / Brier**: `{calibration_results['isotonic']['ece']:.4f}` / `{calibration_results['isotonic']['brier']:.4f}`\n\n"
        md_content += "### Score Distributions (Percentiles: p10, p50 median, p90):\n"
        for sl_name, sl_dist in calibration_results["score_distributions"].items():
            md_content += f"- **{sl_name.replace('_', ' ').title()}**: Median=`{sl_dist['p50']:.4f}`, Mean=`{sl_dist['mean']:.4f}`, IQR=`[{sl_dist['p25']:.4f}, {sl_dist['p75']:.4f}]` (p10=`{sl_dist['p10']:.4f}`, p90=`{sl_dist['p90']:.4f}`)\n"

    md_content += "\n---\n\n## 4. Phase D: Systematic 7-Way Branch Ablation\n\n"
    md_content += "| Branch Permutation | Test Clean Specificity | Face2Face Det (tau=0.5) | Face2Face ROC-AUC | Face2Face PR-AUC |\n"
    md_content += "|---|---|---|---|---|\n"
    for perm_name, perm_data in branch_ablation_results.items():
        md_content += f"| **{perm_name}** | {perm_data['clean_specificity']*100:.2f}% | {perm_data['f2f_det_rate']*100:.2f}% ({perm_data['f2f_detected']:02d}/{perm_data['f2f_total']:02d}) | {perm_data['f2f_roc_auc']:.4f} | {perm_data['f2f_pr_auc']:.4f} |\n"

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved M13.4 Markdown report to {REPORT_MD}")


if __name__ == "__main__":
    main()
