"""Milestone 20.2: Comprehensive 7-Gate Multi-Benchmark Promotion Evaluation.

Evaluates the candidate M20 Pyramidal Dual-Stream Detector against:
1. Active Production Champion M13.5 (v5-family-invariant)
2. Research Candidate Baseline M18 Dual-Stream
3. Research Reference M17 Track A

Pre-Registered 7 Promotion Gates:
- Gate 1 (Primary Gain): Face2Face locked holdout (paired bootstrap vs M18 baseline 0.7854, p < 0.05)
- Gate 1 (Promotion Veto): Non-inferiority vs M13.5 champion (0.7426, Delta >= -0.0150)
- Gate 2 (Cross-Domain Retention): Celeb-DF v2 holdout (Delta >= -0.0150 vs M13.5 0.7823 and M18 0.8308)
- Gate 3 (Compression Robustness): JPEG Q50 holdout (AUC >= 0.7500)
- Gate 4 (Video Codec Non-Inferiority): Uniform H.264 CRF 23 holdout (AUC >= 0.4895)
- Gate 5 (Clean Specificity): Specificity >= 95.0% under validation-derived decision policy tau_95*
- Gate 6 & 7 (Hardware Constraints): P50 forward latency <= 35 ms; Peak VRAM <= 5.5 GB on RTX 3050

Governance Rule:
M13.5 remains the undefeated production champion unless M20.2 satisfies all 7 gates simultaneously.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.m18_detector import M18ReenactmentDetector
from src.models.m20_pyramidal_detector import (
    M20PyramidalDualStreamDetector,
    PyramidalFusionStem,
)
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_m20_promotions")

# Checkpoint paths
V5_CHAMPION_CKPT = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
V5_CALIB_CONFIG = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"
M17_TRACK_A_CKPT = PROJECT_ROOT / "checkpoints" / "m17_track_a_seed42" / "best.pth"
M18_DUAL_STREAM_CKPT = PROJECT_ROOT / "checkpoints" / "m18_dual_stream_seed42" / "best.pth"
M20_CKPT_PATH = PROJECT_ROOT / "checkpoints" / "m20_pyramidal_seed42" / "best.pth"
M20_CALIB_CONFIG = PROJECT_ROOT / "checkpoints" / "m20_pyramidal_seed42" / "calibration_config.json"

# Manifest paths
MANIFEST_CELEBDF = PROJECT_ROOT / "manifests" / "celebdf_v2.csv"
MANIFEST_LOFO = PROJECT_ROOT / "manifests" / "lofo_f2f.csv"
MANIFEST_MIXED = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"

# Preprocessed Tensor Directories
TENSOR_DIRS = [
    PROJECT_ROOT / "processed" / "m12_full",
    PROJECT_ROOT / "processed" / "m13_3_lofo",
    PROJECT_ROOT / "processed" / "celebdf_v2" / "test" / "tensors",
]

OUTPUT_JSON = PROJECT_ROOT / "reports" / "m20_2_evaluation_report.json"
OUTPUT_MD = PROJECT_ROOT / "reports" / "m20_2_evaluation_report.md"

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def apply_simulated_jpeg(tensor: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """Simulate lossy JPEG intra-frame compression."""
    arr = tensor.cpu().numpy()
    out_arr = np.zeros_like(arr)
    for t in range(arr.shape[0]):
        frame = arr[t].transpose(1, 2, 0)
        frame = (frame * NORM_STD + NORM_MEAN) * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if success:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out_arr[t] = ((dec - NORM_MEAN) / NORM_STD).transpose(2, 0, 1)
        else:
            out_arr[t] = arr[t]
    return torch.from_numpy(out_arr).to(tensor.device)


def reencode_tensor_h264_crf23(tensor: torch.Tensor, crf: int = 23) -> torch.Tensor:
    """Uniformly re-encode 16 face frames with x264 CRF 23 using a managed temporary directory."""
    arr = tensor.cpu().numpy()
    T, C, H, W = arr.shape
    uint8_frames = []
    for t in range(T):
        f = arr[t].transpose(1, 2, 0)
        f = (f * NORM_STD + NORM_MEAN) * 255.0
        uint8_frames.append(np.clip(f, 0, 255).astype(np.uint8))

    with tempfile.TemporaryDirectory() as tmpdir:
        in_p = os.path.join(tmpdir, "in.avi")
        out_p = os.path.join(tmpdir, "out.mp4")

        writer = cv2.VideoWriter(in_p, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
        for f in uint8_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", in_p, "-c:v", "libx264", "-crf", str(crf), "-preset", "ultrafast", out_p]
        subprocess.run(cmd, check=True)

        cap = cv2.VideoCapture(out_p)
        dec_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            norm = (rgb - NORM_MEAN) / NORM_STD
            dec_frames.append(norm.transpose(2, 0, 1))
        cap.release()

    if len(dec_frames) < T:
        while len(dec_frames) < T:
            dec_frames.append(dec_frames[-1])
    elif len(dec_frames) > T:
        dec_frames = dec_frames[:T]

    return torch.from_numpy(np.array(dec_frames, dtype=np.float32)).to(tensor.device)


def compute_paired_bootstrap(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_ref: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap delta AUC test with confidence intervals and p-value."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_cand = roc_auc_score(y_b, scores_cand[idx])
            auc_ref = roc_auc_score(y_b, scores_ref[idx])
            deltas.append(auc_cand - auc_ref)
        except ValueError:
            continue

    deltas = np.array(deltas)
    delta_mean = float(np.mean(deltas))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = float(np.mean(deltas <= 0.0))

    return {
        "delta_mean": delta_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


def evaluate_benchmark_quad(
    items: List[Any],
    model_m20: M20PyramidalDualStreamDetector,
    model_m18: M18ReenactmentDetector,
    model_m17: M17TrackADetector,
    model_v5: FamilyInvariantDeepfakeDetector,
    device: torch.device,
    v5_temp: float,
    m20_tau_95: float = 0.50,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate candidate and comparator models on identical benchmark items."""
    y_true = []
    p_m20_list, p_m18_list, p_m17_list, p_v5_list = [], [], [], []
    m20_gates = []

    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            tensors = []
            for it in chunk:
                data = it[0]
                if isinstance(data, (str, Path)):
                    t = torch.load(data, map_location="cpu", weights_only=False)
                elif isinstance(data, tuple) and len(data) == 2:
                    path, transform = data
                    t = torch.load(path, map_location="cpu", weights_only=False)
                    if transform == "jpeg_50":
                        t = apply_simulated_jpeg(t, quality=50)
                    elif transform == "h264_crf23":
                        t = reencode_tensor_h264_crf23(t, crf=23)
                elif isinstance(data, torch.Tensor):
                    t = data
                else:
                    raise TypeError(f"Unknown data type: {type(data)}")

                if t.dim() == 5:
                    t = t.squeeze(0)
                tensors.append(t)

            labels = [it[1] for it in chunk]
            batch_tensor = torch.stack(tensors, dim=0).to(device)

            with torch.amp.autocast("cuda"):
                # 1. M20 Candidate
                out_m20 = model_m20(batch_tensor, mode="dual_stream")
                p_m20 = out_m20.p_fused.squeeze(-1).tolist()
                gates = out_m20.gate_values.squeeze(-1).tolist()

                # 2. M18 Baseline
                out_m18 = model_m18(batch_tensor, mode="dual_stream")
                p_m18 = out_m18.p_fused.squeeze(-1).tolist()

                # 3. M17 Track A Reference
                out_m17 = model_m17(batch_tensor, active_scales=(True, True, False))
                p_m17 = torch.sigmoid(out_m17.primary_logits).squeeze(-1).tolist()

                # 4. M13.5 Champion Control
                out_v5 = model_v5(batch_tensor)
                p_v5 = torch.sigmoid(out_v5.primary_logits / v5_temp).squeeze(-1).tolist()

            if isinstance(p_m20, float): p_m20 = [p_m20]
            if isinstance(p_m18, float): p_m18 = [p_m18]
            if isinstance(p_m17, float): p_m17 = [p_m17]
            if isinstance(p_v5, float): p_v5 = [p_v5]
            if isinstance(gates, float): gates = [gates]

            y_true.extend(labels)
            p_m20_list.extend(p_m20)
            p_m18_list.extend(p_m18)
            p_m17_list.extend(p_m17)
            p_v5_list.extend(p_v5)
            m20_gates.extend(gates)

            del batch_tensor, tensors

    y_arr = np.array(y_true)
    m20_arr = np.array(p_m20_list)
    m18_arr = np.array(p_m18_list)
    m17_arr = np.array(p_m17_list)
    v5_arr = np.array(p_v5_list)

    def calc_metrics(scores: np.ndarray, tau: float = 0.50) -> Dict[str, float]:
        auc = float(roc_auc_score(y_arr, scores)) if len(np.unique(y_arr)) > 1 else 0.5
        prauc = float(average_precision_score(y_arr, scores)) if len(np.unique(y_arr)) > 1 else 0.5
        clean_mask = y_arr == 0
        fake_mask = y_arr == 1
        spec = float(np.mean(scores[clean_mask] < tau)) if np.any(clean_mask) else 0.0
        det = float(np.mean(scores[fake_mask] >= tau)) if np.any(fake_mask) else 0.0
        return {"auc": auc, "prauc": prauc, "spec": spec, "det_rate": det}

    res_m20 = calc_metrics(m20_arr, tau=m20_tau_95)
    res_m18 = calc_metrics(m18_arr, tau=0.50)
    res_m17 = calc_metrics(m17_arr, tau=0.50)
    res_v5 = calc_metrics(v5_arr, tau=0.50)

    # Paired bootstraps
    bs_m20_vs_m18 = compute_paired_bootstrap(y_arr, m20_arr, m18_arr)
    bs_m20_vs_v5 = compute_paired_bootstrap(y_arr, m20_arr, v5_arr)

    return {
        "m20_candidate": {
            **res_m20,
            "gate_mean": float(np.mean(m20_gates)),
            "delta_vs_m18": bs_m20_vs_m18["delta_mean"],
            "ci_vs_m18": [bs_m20_vs_m18["ci_lower"], bs_m20_vs_m18["ci_upper"]],
            "p_val_vs_m18": bs_m20_vs_m18["p_value"],
            "delta_vs_champion": bs_m20_vs_v5["delta_mean"],
            "ci_vs_champion": [bs_m20_vs_v5["ci_lower"], bs_m20_vs_v5["ci_upper"]],
            "p_val_vs_champion": bs_m20_vs_v5["p_value"],
        },
        "m18_baseline": res_m18,
        "m17_track_a": res_m17,
        "m13_5_champion": res_v5,
        "n_samples": len(y_arr),
        "n_reals": int(sum(1 for y in y_arr if y == 0)),
        "n_fakes": int(sum(1 for y in y_arr if y == 1)),
    }


def profile_hardware(model: M20PyramidalDualStreamDetector, device: torch.device) -> Dict[str, float]:
    """Profile P50/P90/P99 latency and peak VRAM on RTX 3050."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    dummy = torch.randn(1, 16, 3, 112, 112, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            with torch.amp.autocast("cuda"):
                _ = model(dummy)
    torch.cuda.synchronize()

    # Benchmark 100 runs
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda"):
                _ = model(dummy)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
        "peak_vram_mb": peak_vram_mb,
        "peak_vram_gb": peak_vram_mb / 1024.0,
    }


def main():
    logger.info("Initializing Milestone 20.2 Promotion Evaluation Suite...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load M13.5 Production Champion
    cfg = load_config("configs/experiments/m12_full.yaml")
    base_v5 = DeepfakeDetector(config=cfg.model).to(device)
    v5_ckpt = torch.load(V5_CHAMPION_CKPT, map_location=device, weights_only=False)
    model_v5 = FamilyInvariantDeepfakeDetector(
        base_detector=base_v5, num_families=5, aux_hidden_dim=512, aux_dropout=0.5,
        freq_dim=64, texture_dim=64, fusion_dim=512, enable_mid_head=True,
    ).to(device)
    model_v5.load_state_dict(v5_ckpt["model_state_dict"])
    model_v5.eval()

    with open(V5_CALIB_CONFIG, "r", encoding="utf-8") as f:
        v5_calib = json.load(f)
    v5_temp = float(v5_calib.get("temperature", 2.0502))
    logger.info("M13.5 Champion loaded (T=%.4f)", v5_temp)

    # 2. Load M17 Track A Reference
    base_m17 = DeepfakeDetector(config=cfg.model).to(device)
    m17_ckpt = torch.load(M17_TRACK_A_CKPT, map_location=device, weights_only=False)
    model_m17 = M17TrackADetector(
        base_detector=base_m17, num_families=5, res_dim=128, aux_hidden_dim=512,
        aux_dropout=0.5, gate_bottleneck=128,
    ).to(device)
    model_m17.load_state_dict(m17_ckpt["model_state_dict"])
    model_m17.eval()

    # 3. Load M18 Dual-Stream Baseline
    m18_ckpt = torch.load(M18_DUAL_STREAM_CKPT, map_location=device, weights_only=False)
    model_m18 = M18ReenactmentDetector(
        trunk_detector=model_m17, kin_dim=256, aux_hidden_dim=512,
        gate_bottleneck=128, freeze_trunk=True,
    ).to(device)
    model_m18.load_state_dict(m18_ckpt["model_state_dict"])
    model_m18.eval()

    # 4. Load M20 Pyramidal Candidate
    assert M20_CKPT_PATH.exists(), f"M20 Checkpoint missing: {M20_CKPT_PATH}"
    m20_ckpt = torch.load(M20_CKPT_PATH, map_location=device, weights_only=False)
    model_m20 = M20PyramidalDualStreamDetector(
        trunk_detector=model_m17, kin_dim=256, aux_hidden_dim=512,
        aux_dropout=0.5, gate_bottleneck=128, init_pyramid_alpha=0.50,
        freeze_trunk=True,
    ).to(device)
    model_m20.load_state_dict(m20_ckpt["model_state_dict"])
    model_m20.eval()

    # Read M20 calibration config if available
    m20_tau_95 = 0.50
    if M20_CALIB_CONFIG.exists():
        with open(M20_CALIB_CONFIG, "r", encoding="utf-8") as f:
            m20_calib_data = json.load(f)
        m20_tau_95 = float(m20_calib_data.get("policies", {}).get("uncalibrated", {}).get("tau_95", 0.50))
    logger.info("M20 Pyramidal Candidate loaded (Alpha=%.4f, tau_95*=%.4f)",
                float(model_m20.pyramidal_stem.alpha.item()), m20_tau_95)

    # Prepare Benchmark Datasets
    # 1. Gate 1: Face2Face Locked Holdout (116 clips)
    mixed_df = pd.read_csv(MANIFEST_MIXED)
    mixed_test = mixed_df[mixed_df["split"] == "test"].reset_index(drop=True)
    clean_reals = mixed_test[mixed_test["manipulation_type"] == "original"]
    f2f_fakes = mixed_test[mixed_test["manipulation_type"] == "Face2Face"]
    f2f_items = [(find_tensor(str(r["video_id"]), split="test"), 0)
                 for _, r in clean_reals.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None] + \
                [(find_tensor(str(r["video_id"]), split="test"), 1)
                 for _, r in f2f_fakes.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]

    # 2. Gate 2: Celeb-DF-v2 Locked Holdout (511 clips)
    celeb_df = pd.read_csv(MANIFEST_CELEBDF)
    celeb_test = celeb_df[celeb_df["split"] == "test"].reset_index(drop=True)
    celeb_items = [(find_tensor(str(r["video_id"]), split="test"), int(r["label"]))
                   for _, r in celeb_test.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]

    # 3. Gate 3: JPEG Q=50 Compression Stress (801 clips)
    lofo_df = pd.read_csv(MANIFEST_LOFO)
    test_lofo = lofo_df[lofo_df["split"] == "test"].reset_index(drop=True)
    q50_items = [((find_tensor(str(r["video_id"]), split="test"), "jpeg_50"), int(r["label"]))
                 for _, r in test_lofo.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]

    # 4. Gate 4: Uniform H.264 CRF 23 Re-Encoding Stress (511 Celeb-DF clips)
    reenc_items = [((tp, "h264_crf23"), lbl) for tp, lbl in celeb_items]

    # 5. Gate 5: Clean Specificity Set (FF++ Reals + Celeb-DF Reals)
    clean_items = [(find_tensor(str(r["video_id"]), split="test"), 0)
                   for _, r in clean_reals.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]

    logger.info("Evaluating Gate 1: Face2Face Locked Holdout (N=%d)...", len(f2f_items))
    res_g1 = evaluate_benchmark_quad(f2f_items, model_m20, model_m18, model_m17, model_v5, device, v5_temp, m20_tau_95)

    logger.info("Evaluating Gate 2: Celeb-DF v2 Holdout (N=%d)...", len(celeb_items))
    res_g2 = evaluate_benchmark_quad(celeb_items, model_m20, model_m18, model_m17, model_v5, device, v5_temp, m20_tau_95)

    logger.info("Evaluating Gate 3: JPEG Q50 Compression (N=%d)...", len(q50_items))
    res_g3 = evaluate_benchmark_quad(q50_items, model_m20, model_m18, model_m17, model_v5, device, v5_temp, m20_tau_95)

    logger.info("Evaluating Gate 4: H.264 CRF 23 Re-Encoding (N=%d)...", len(reenc_items))
    res_g4 = evaluate_benchmark_quad(reenc_items, model_m20, model_m18, model_m17, model_v5, device, v5_temp, m20_tau_95)

    logger.info("Evaluating Gate 6 & 7: Production Hardware Profiling...")
    hw_profile = profile_hardware(model_m20, device)

    # 7-Gate Verification Logic
    # Gate 1: F2F AUC vs M18 baseline (p < 0.05) & Non-inferiority vs Champion (Delta >= -0.0150)
    g1_auc = res_g1["m20_candidate"]["auc"]
    g1_p_val = res_g1["m20_candidate"]["p_val_vs_m18"]
    g1_delta_v5 = res_g1["m20_candidate"]["delta_vs_champion"]
    g1_pass = (g1_auc > res_g1["m18_baseline"]["auc"] and g1_p_val < 0.05) and (g1_delta_v5 >= -0.0150)

    # Gate 2: Celeb-DF retention (Delta >= -0.0150 vs M13.5 and M18)
    g2_auc = res_g2["m20_candidate"]["auc"]
    g2_delta_v5 = res_g2["m20_candidate"]["delta_vs_champion"]
    g2_delta_m18 = res_g2["m20_candidate"]["delta_vs_m18"]
    g2_pass = (g2_delta_v5 >= -0.0150) and (g2_delta_m18 >= -0.0150)

    # Gate 3: JPEG Q50 robustness (AUC >= 0.7500)
    g3_auc = res_g3["m20_candidate"]["auc"]
    g3_pass = g3_auc >= 0.7500

    # Gate 4: H.264 CRF 23 Non-Inferiority (AUC >= 0.4895)
    g4_auc = res_g4["m20_candidate"]["auc"]
    g4_pass = g4_auc >= 0.4895

    # Gate 5: Clean Specificity under tau_95* >= 95.0%
    g5_spec = res_g1["m20_candidate"]["spec"]
    g5_pass = g5_spec >= 0.950

    # Gate 6 & 7: Latency <= 35 ms & Peak VRAM <= 5.5 GB
    g6_pass = hw_profile["p50_ms"] <= 35.0
    g7_pass = hw_profile["peak_vram_gb"] <= 5.5

    all_gates_pass = all([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass, g6_pass, g7_pass])
    verdict = "PROMOTE_TO_CHAMPION" if all_gates_pass else "RETAIN_M13_5_CHAMPION"

    logger.info("=== 7-Gate Promotion Audit Verdict: %s ===", verdict)
    logger.info("Gate 1 (Face2Face Primary): %s (AUC=%.4f, Delta_M18=%+.4f, p=%.4f, Delta_v5=%+.4f)",
                "PASS" if g1_pass else "FAIL", g1_auc, res_g1["m20_candidate"]["delta_vs_m18"], g1_p_val, g1_delta_v5)
    logger.info("Gate 2 (Cross-Domain): %s (AUC=%.4f, Delta_v5=%+.4f, Delta_M18=%+.4f)",
                "PASS" if g2_pass else "FAIL", g2_auc, g2_delta_v5, g2_delta_m18)
    logger.info("Gate 3 (JPEG Q50): %s (AUC=%.4f, Req >= 0.7500)", "PASS" if g3_pass else "FAIL", g3_auc)
    logger.info("Gate 4 (H.264 CRF23): %s (AUC=%.4f, Req >= 0.4895)", "PASS" if g4_pass else "FAIL", g4_auc)
    logger.info("Gate 5 (Clean Spec): %s (Spec=%.1f%% under tau_95*=%.4f)", "PASS" if g5_pass else "FAIL", g5_spec * 100, m20_tau_95)
    logger.info("Gate 6 (P50 Latency): %s (%.2f ms <= 35 ms)", "PASS" if g6_pass else "FAIL", hw_profile["p50_ms"])
    logger.info("Gate 7 (Peak VRAM): %s (%.2f GB <= 5.5 GB)", "PASS" if g7_pass else "FAIL", hw_profile["peak_vram_gb"])

    report_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": "M20.2",
        "title": "Milestone 20.2: Pyramidal Dual-Stream Detector 7-Gate Promotion Audit",
        "verdict": verdict,
        "gates": {
            "Gate1_Face2Face": {
                "pass": bool(g1_pass),
                "auc": g1_auc,
                "delta_vs_m18": res_g1["m20_candidate"]["delta_vs_m18"],
                "p_val_vs_m18": g1_p_val,
                "delta_vs_champion": g1_delta_v5,
            },
            "Gate2_CrossDomain": {
                "pass": bool(g2_pass),
                "auc": g2_auc,
                "delta_vs_champion": g2_delta_v5,
                "delta_vs_m18": g2_delta_m18,
            },
            "Gate3_JPEG_Compression": {
                "pass": bool(g3_pass),
                "auc": g3_auc,
                "threshold": 0.7500,
            },
            "Gate4_VideoCodec": {
                "pass": bool(g4_pass),
                "auc": g4_auc,
                "threshold": 0.4895,
            },
            "Gate5_CleanSpecificity": {
                "pass": bool(g5_pass),
                "specificity": g5_spec,
                "tau_95": m20_tau_95,
            },
            "Gate6_ProductionLatency": {
                "pass": bool(g6_pass),
                "p50_ms": hw_profile["p50_ms"],
                "threshold_ms": 35.0,
            },
            "Gate7_VRAMEnvelope": {
                "pass": bool(g7_pass),
                "peak_vram_gb": hw_profile["peak_vram_gb"],
                "threshold_gb": 5.5,
            },
        },
        "benchmarks": {
            "Gate1_Face2Face": res_g1,
            "Gate2_CelebDF": res_g2,
            "Gate3_JPEG_Q50": res_g3,
            "Gate4_H264_CRF23": res_g4,
        },
        "hardware": hw_profile,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    logger.info("Evaluation JSON report saved to: %s", OUTPUT_JSON)

    # Format Markdown Report
    lines = [
        "# Milestone 20.2 — Pyramidal Dual-Stream Detector: 7-Gate Promotion Report",
        "",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Audit Verdict**: `{verdict}`  ",
        "**Candidate Model**: `M20PyramidalDualStreamDetector` (`checkpoints/m20_pyramidal_seed42/best.pth`)  ",
        "**Production Champion**: `M13.5 (v5-family-invariant)` (`checkpoints/m13_3_f2f_seed42/best.pth`)  ",
        "**Research Baseline**: `M18 Dual-Stream` (`checkpoints/m18_dual_stream_seed42/best.pth`)  ",
        "",
        "---",
        "",
        "## 1. Pre-Registered 7-Gate Audit Scorecard",
        "",
        "| Gate | Benchmark / Metric | Candidate M20 | Baseline M18 | Champion M13.5 | Requirement | Status |",
        "|---|---|---:|---:|---:|---|:---:|",
        f"| **Gate 1** | F2F ROC-AUC | {g1_auc:.4f} | {res_g1['m18_baseline']['auc']:.4f} | {res_g1['m13_5_champion']['auc']:.4f} | Paired Gain vs M18 (p<0.05) & Non-inferior vs M13.5 | {'PASS' if g1_pass else 'FAIL'} |",
        f"| **Gate 2** | Celeb-DF v2 ROC-AUC | {g2_auc:.4f} | {res_g2['m18_baseline']['auc']:.4f} | {res_g2['m13_5_champion']['auc']:.4f} | Delta >= -0.0150 vs M13.5 & M18 | {'PASS' if g2_pass else 'FAIL'} |",
        f"| **Gate 3** | JPEG Q50 ROC-AUC | {g3_auc:.4f} | {res_g3['m18_baseline']['auc']:.4f} | {res_g3['m13_5_champion']['auc']:.4f} | AUC >= 0.7500 | {'PASS' if g3_pass else 'FAIL'} |",
        f"| **Gate 4** | H.264 CRF 23 ROC-AUC | {g4_auc:.4f} | {res_g4['m18_baseline']['auc']:.4f} | {res_g4['m13_5_champion']['auc']:.4f} | AUC >= 0.4895 | {'PASS' if g4_pass else 'FAIL'} |",
        f"| **Gate 5** | Clean Specificity (tau_95*) | {g5_spec*100:.1f}% | {res_g1['m18_baseline']['spec']*100:.1f}% | {res_g1['m13_5_champion']['spec']*100:.1f}% | Specificity >= 95.0% | {'PASS' if g5_pass else 'FAIL'} |",
        f"| **Gate 6** | P50 Forward Latency | {hw_profile['p50_ms']:.2f} ms | — | — | <= 35.0 ms | {'PASS' if g6_pass else 'FAIL'} |",
        f"| **Gate 7** | Peak GPU VRAM | {hw_profile['peak_vram_gb']:.2f} GB | — | — | <= 5.50 GB | {'PASS' if g7_pass else 'FAIL'} |",
        "",
        "---",
        "",
        "## 2. Statistical Attribution & Significance",
        "",
        f"- **Gate 1 Delta vs M18 Baseline**: {res_g1['m20_candidate']['delta_vs_m18']:+.4f} (95% CI: [{res_g1['m20_candidate']['ci_vs_m18'][0]:+.4f}, {res_g1['m20_candidate']['ci_vs_m18'][1]:+.4f}], p = {g1_p_val:.4f})",
        f"- **Gate 1 Delta vs Production Champion**: {g1_delta_v5:+.4f} (95% CI: [{res_g1['m20_candidate']['ci_vs_champion'][0]:+.4f}, {res_g1['m20_candidate']['ci_vs_champion'][1]:+.4f}], p = {res_g1['m20_candidate']['p_val_vs_champion']:.4f})",
        f"- **Gate 2 Delta vs M18 Baseline**: {g2_delta_m18:+.4f} (95% CI: [{res_g2['m20_candidate']['ci_vs_m18'][0]:+.4f}, {res_g2['m20_candidate']['ci_vs_m18'][1]:+.4f}], p = {res_g2['m20_candidate']['p_val_vs_m18']:.4f})",
        f"- **Gate 2 Delta vs Production Champion**: {g2_delta_v5:+.4f} (95% CI: [{res_g2['m20_candidate']['ci_vs_champion'][0]:+.4f}, {res_g2['m20_candidate']['ci_vs_champion'][1]:+.4f}], p = {res_g2['m20_candidate']['p_val_vs_champion']:.4f})",
        "",
        "---",
        "",
        "## 3. Governance Verdict & Next Steps",
        "",
        f"**Audit Verdict**: `{verdict}`",
        "",
        "- If `PROMOTE_TO_CHAMPION`: M20.2 satisfies all 7 promotion gates, improving reenactment discrimination without regression on cross-domain, compression, or latency budgets.",
        "- If `RETAIN_M13_5_CHAMPION`: M13.5 remains the undefeated active production champion. M20.2 is archived as a valuable empirical research milestone.",
        "",
    ]

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Evaluation Markdown report saved to: %s", OUTPUT_MD)


if __name__ == "__main__":
    main()
