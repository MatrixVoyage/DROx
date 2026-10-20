"""Evaluation Script for Milestone 17 Track A (Sensor/Residual Representation).

Evaluates M17 Track A against the pre-registered promotion gates:
- Gate 1: Unseen OOD Generalization (Celeb-DF-v2 holdout, Delta AUC_Celeb > 0.0 vs frozen M13.5).
- Gate 2: Clean Real Specificity Floor (FF++ >= 93.0% and Celeb-DF >= 95.0% at tau = 0.50).
- Gate 3: Compression Robustness Stress Test (ROC-AUC >= 0.6500 under JPEG Q=50).
- Gate 4: Cross-Family Multi-Manipulation (Macro AUC >= 0.9000 across F2F, NT, FS, DF).
- Gate 5: Hardware Boundaries (P50 latency <= 35.0ms, Peak VRAM <= 5.2GB).
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

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector
from src.models.m17_detector import M17TrackADetector, M17TrackAOutput
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("m17_track_a_eval")

MANIFEST_MIXED = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
MANIFEST_CELEBDF = PROJECT_ROOT / "manifests" / "celebdf_v2.csv"
MANIFEST_LOFO = PROJECT_ROOT / "manifests" / "lofo_f2f.csv"

M17_TRACK_A_CKPT = PROJECT_ROOT / "checkpoints" / "m17_track_a_seed42" / "best.pth"
V5_CHAMPION_CKPT = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
V5_CALIB_CONFIG = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"

TENSOR_DIR_M12 = PROJECT_ROOT / "processed" / "m12_full"
TENSOR_DIR_LOFO = PROJECT_ROOT / "processed" / "m13_3_lofo"
TENSOR_DIR_CELEBDF = PROJECT_ROOT / "processed" / "celebdf_v2" / "test" / "tensors"

REPORT_JSON = PROJECT_ROOT / "reports" / "m17_track_a_evaluation_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m17_track_a_evaluation_report.md"


def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_tensor(video_id: str, split: str = "test") -> Optional[Path]:
    candidates = [
        TENSOR_DIR_M12 / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIR_LOFO / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIR_CELEBDF / f"{video_id}.pt",
        TENSOR_DIR_M12 / "val" / "tensors" / f"{video_id}.pt",
        TENSOR_DIR_M12 / "train" / "tensors" / f"{video_id}.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def apply_simulated_jpeg(tensor: torch.Tensor, quality: int = 50) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = tensor.cpu().numpy()
    out_arr = np.zeros_like(arr)

    for t in range(arr.shape[0]):
        frame = arr[t].transpose(1, 2, 0)
        frame = (frame * std + mean) * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", frame, encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR) if success else frame

        norm_frame = dec.astype(np.float32) / 255.0
        norm_frame = (norm_frame - mean) / std
        out_arr[t] = norm_frame.transpose(2, 0, 1)

    return torch.from_numpy(out_arr).to(tensor.device, dtype=tensor.dtype)


def compute_paired_bootstrap(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_base: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_cand = roc_auc_score(y_b, scores_cand[idx])
        auc_base = roc_auc_score(y_b, scores_base[idx])
        deltas.append(auc_cand - auc_base)

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = float(np.mean(deltas <= 0.0))
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    y = val_labels.astype(np.float64)
    z = val_logits.astype(np.float64)

    def nll(T: float) -> float:
        T = max(1e-4, float(T))
        p = 1.0 / (1.0 + np.exp(-z / T))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def main():
    parser = argparse.ArgumentParser(description="Evaluate M17 Track A against Promotion Gates")
    parser.add_argument("--checkpoint", type=Path, default=M17_TRACK_A_CKPT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Evaluating M17 Track A on device: {device}")

    # Load frozen v5 champion
    logger.info(f"Loading frozen v5 champion from: {V5_CHAMPION_CKPT}")
    cfg = load_config("configs/experiments/m12_full.yaml")
    base_v5 = DeepfakeDetector(config=cfg.model).to(device)
    v5_ckpt = torch.load(V5_CHAMPION_CKPT, map_location=device, weights_only=False)

    model_v5 = FamilyInvariantDeepfakeDetector(
        base_detector=base_v5,
        num_families=5,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        freq_dim=64,
        texture_dim=64,
        fusion_dim=512,
        enable_mid_head=True,
    ).to(device)
    model_v5.load_state_dict(v5_ckpt["model_state_dict"])
    model_v5.eval()

    with open(V5_CALIB_CONFIG, "r", encoding="utf-8") as f:
        v5_calib = json.load(f)
    v5_T = float(v5_calib.get("temperature", 2.0502))
    logger.info(f"v5 Champion ready with T={v5_T:.4f}")

    # Load M17 Track A Candidate
    logger.info(f"Loading M17 Track A candidate from: {args.checkpoint}")
    m17_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_cand = DeepfakeDetector(config=cfg.model).to(device)

    model_cand = M17TrackADetector(
        base_detector=base_cand,
        num_families=5,
        res_dim=128,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        gate_bottleneck=128,
    ).to(device)
    model_cand.load_state_dict(m17_ckpt["model_state_dict"])
    model_cand.eval()

    # Temperature scaling on validation set
    val_df = pd.read_csv(MANIFEST_LOFO)
    val_df = val_df[val_df["split"] == "val"].reset_index(drop=True)
    val_items = []
    for _, row in val_df.iterrows():
        tp = find_tensor(str(row["video_id"]), split="val")
        if tp is not None:
            val_items.append((tp, int(row["label"])))

    val_logits_list = []
    val_labels_list = []
    with torch.no_grad():
        for tp, lbl in val_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            with torch.amp.autocast("cuda"):
                out = model_cand(tensor)
            val_logits_list.append(out.primary_logits.item())
            val_labels_list.append(lbl)

    T_cand = fit_temperature_scaling(np.array(val_logits_list), np.array(val_labels_list))
    logger.info(f"Fitted M17 Track A temperature scaling T={T_cand:.4f}")

    # =========================================================================
    # Phase 1: Gate 1 & Gate 2 — Celeb-DF v2 External Generalization Benchmark
    # =========================================================================
    logger.info("Evaluating Gate 1 (Celeb-DF Generalization) & Gate 2 (Clean Specificity)...")
    celeb_df = pd.read_csv(MANIFEST_CELEBDF)
    celeb_test = celeb_df[celeb_df["split"] == "test"].copy().reset_index(drop=True)

    celeb_items = []
    for _, row in celeb_test.iterrows():
        tp = find_tensor(str(row["video_id"]), split="test")
        if tp is not None:
            celeb_items.append((tp, int(row["label"])))

    logger.info(f"Loaded {len(celeb_items)} Celeb-DF test items.")

    celeb_labels = []
    celeb_scores_cand = []
    celeb_scores_v5 = []

    with torch.no_grad():
        for tp, lbl in celeb_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)

            with torch.amp.autocast("cuda"):
                out_cand = model_cand(tensor)
                p_cand = torch.sigmoid(out_cand.primary_logits / T_cand).item()

                out_v5 = model_v5(tensor, return_dict=True)
                p_v5 = torch.sigmoid(out_v5.primary_logits / v5_T).item()

            celeb_labels.append(lbl)
            celeb_scores_cand.append(p_cand)
            celeb_scores_v5.append(p_v5)

    celeb_labels_arr = np.array(celeb_labels)
    celeb_cand_arr = np.array(celeb_scores_cand)
    celeb_v5_arr = np.array(celeb_scores_v5)

    auc_cand_celeb = float(roc_auc_score(celeb_labels_arr, celeb_cand_arr))
    auc_v5_celeb = float(roc_auc_score(celeb_labels_arr, celeb_v5_arr))
    delta_auc_celeb = auc_cand_celeb - auc_v5_celeb

    celeb_clean_mask = celeb_labels_arr == 0
    n_celeb_clean = int(np.sum(celeb_clean_mask))
    celeb_clean_correct = int(np.sum(celeb_cand_arr[celeb_clean_mask] < 0.50))
    celeb_spec = celeb_clean_correct / max(1, n_celeb_clean)

    boot_celeb = compute_paired_bootstrap(celeb_labels_arr, celeb_cand_arr, celeb_v5_arr)

    logger.info(f"Celeb-DF AUC: Cand={auc_cand_celeb:.4f} | v5={auc_v5_celeb:.4f} | Delta={delta_auc_celeb:+.4f} (95% CI: [{boot_celeb['ci_lower']:+.4f}, {boot_celeb['ci_upper']:+.4f}])")
    logger.info(f"Celeb-DF Clean Specificity: {celeb_spec*100:.2f}% ({celeb_clean_correct}/{n_celeb_clean})")

    # =========================================================================
    # Phase 2: Gate 3 — Compression Stress Testing (JPEG Q=50)
    # =========================================================================
    logger.info("Evaluating Gate 3 (Compression Robustness under JPEG Q=50)...")
    lofo_df = pd.read_csv(MANIFEST_LOFO)
    test_lofo = lofo_df[lofo_df["split"] == "test"].copy().reset_index(drop=True)

    test_items = []
    for _, row in test_lofo.iterrows():
        tp = find_tensor(str(row["video_id"]), split="test")
        if tp is not None:
            test_items.append((tp, int(row["label"]), str(row.get("manipulation_type", ""))))

    q50_labels = []
    q50_scores = []
    with torch.no_grad():
        for tp, lbl, _ in test_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            degraded = apply_simulated_jpeg(tensor.squeeze(0), quality=50).unsqueeze(0)
            with torch.amp.autocast("cuda"):
                out = model_cand(degraded)
                p = torch.sigmoid(out.primary_logits / T_cand).item()
            q50_labels.append(lbl)
            q50_scores.append(p)

    q50_auc = float(roc_auc_score(np.array(q50_labels), np.array(q50_scores)))
    logger.info(f"JPEG Q=50 Stress AUC: {q50_auc:.4f} (Threshold: >= 0.6500)")

    # =========================================================================
    # Phase 3: Gate 4 — Cross-Family Multi-Manipulation Performance
    # =========================================================================
    logger.info("Evaluating Gate 4 (Cross-Family Performance across FF++ Families)...")
    mixed_df = pd.read_csv(MANIFEST_MIXED)
    mixed_test = mixed_df[mixed_df["split"] == "test"].copy().reset_index(drop=True)

    family_results = {}
    clean_reals = mixed_test[mixed_test["manipulation_type"] == "original"]
    clean_tensors = [(find_tensor(str(r["video_id"]), split="test"), 0) for _, r in clean_reals.iterrows()]
    clean_tensors = [(tp, lbl) for tp, lbl in clean_tensors if tp is not None]

    # Clean FF++ Specificity (Gate 2)
    ffpp_clean_correct = 0
    with torch.no_grad():
        for tp, _ in clean_tensors:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            with torch.amp.autocast("cuda"):
                out = model_cand(tensor)
                p = torch.sigmoid(out.primary_logits / T_cand).item()
            if p < 0.50:
                ffpp_clean_correct += 1

    clean_ffpp_spec = ffpp_clean_correct / max(1, len(clean_tensors))
    logger.info(f"Clean FF++ Specificity: {clean_ffpp_spec*100:.2f}% ({ffpp_clean_correct}/{len(clean_tensors)})")

    target_families = ["Face2Face", "NeuralTextures", "FaceSwap", "Deepfakes"]
    family_aucs = []

    for fam in target_families:
        fam_df = mixed_test[mixed_test["manipulation_type"] == fam]
        fam_tensors = [(find_tensor(str(r["video_id"]), split="test"), 1) for _, r in fam_df.iterrows()]
        fam_tensors = [(tp, lbl) for tp, lbl in fam_tensors if tp is not None]

        eval_set = clean_tensors + fam_tensors
        y_fam, s_fam = [], []
        with torch.no_grad():
            for tp, lbl in eval_set:
                tensor = torch.load(tp, map_location=device, weights_only=False)
                if tensor.dim() == 4:
                    tensor = tensor.unsqueeze(0)
                with torch.amp.autocast("cuda"):
                    out = model_cand(tensor)
                    p = torch.sigmoid(out.primary_logits / T_cand).item()
                y_fam.append(lbl)
                s_fam.append(p)

        fam_auc = float(roc_auc_score(y_fam, s_fam)) if len(np.unique(y_fam)) > 1 else 0.5
        det_rate = float(np.mean(np.array(s_fam)[np.array(y_fam) == 1] >= 0.50))
        family_results[fam] = {"auc": fam_auc, "det_rate": det_rate}
        family_aucs.append(fam_auc)
        logger.info(f"  [{fam}] AUC: {fam_auc:.4f} | Det Rate: {det_rate*100:.1f}%")

    macro_auc = float(np.mean(family_aucs))
    logger.info(f"Cross-Family Macro AUC: {macro_auc:.4f} (Threshold: >= 0.9000)")

    # =========================================================================
    # Phase 4: Gate 5 — Hardware Latency & Peak VRAM
    # =========================================================================
    logger.info("Evaluating Gate 5 (Hardware Latency & Peak VRAM)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    latencies = []
    dummy_input = torch.randn(1, 16, 3, 112, 112, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = model_cand(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(30):
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = model_cand(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_latency = float(np.percentile(latencies, 50))
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    logger.info(f"P50 Latency: {p50_latency:.2f} ms | Peak VRAM: {peak_vram_gb:.2f} GB")

    # =========================================================================
    # Promotion Gate Decision Verification
    # =========================================================================
    gate1_pass = delta_auc_celeb > 0.0
    gate2_pass = clean_ffpp_spec >= 0.930 and celeb_spec >= 0.950
    gate3_pass = q50_auc >= 0.6500
    gate4_pass = macro_auc >= 0.9000
    gate5_pass = p50_latency <= 35.0 and peak_vram_gb <= 5.2

    overall_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass

    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track": "Track A (Sensor/Residual Representation)",
        "checkpoint": str(args.checkpoint),
        "temperature": T_cand,
        "gates": {
            "gate1_unseen_ood_celebdf": {
                "metric": "Delta AUC on Celeb-DF v2 vs v5 Champion",
                "auc_cand": auc_cand_celeb,
                "auc_v5": auc_v5_celeb,
                "delta_auc": delta_auc_celeb,
                "ci_lower": boot_celeb["ci_lower"],
                "ci_upper": boot_celeb["ci_upper"],
                "threshold": "> 0.0",
                "passed": gate1_pass,
            },
            "gate2_clean_real_specificity": {
                "metric": "Clean Real Specificity",
                "ffpp_clean_spec": clean_ffpp_spec,
                "celeb_clean_spec": celeb_spec,
                "threshold": "FF++ >= 93.0% and Celeb-DF >= 95.0%",
                "passed": gate2_pass,
            },
            "gate3_compression_stress": {
                "metric": "JPEG Q=50 Stress Test ROC-AUC",
                "q50_auc": q50_auc,
                "threshold": ">= 0.6500",
                "passed": gate3_pass,
            },
            "gate4_cross_family": {
                "metric": "4-Family Evaluation (F2F, NT, FS, DF)",
                "family_metrics": family_results,
                "macro_auc": macro_auc,
                "threshold": "Macro AUC >= 0.9000",
                "passed": gate4_pass,
            },
            "gate5_hardware": {
                "metric": "P50 Latency and Peak VRAM on RTX 3050",
                "p50_latency_ms": p50_latency,
                "peak_vram_gb": peak_vram_gb,
                "threshold": "Latency <= 35.0ms, VRAM <= 5.2GB",
                "passed": gate5_pass,
            },
        },
        "verdict": "PROMOTE_TRACK_A" if overall_pass else "RETAIN_M13_5_CONTROL",
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Saved evaluation report JSON to {REPORT_JSON}")
    logger.info(f"Final Promotion Verdict for Track A: {results_data['verdict']}")
    return results_data


if __name__ == "__main__":
    main()
