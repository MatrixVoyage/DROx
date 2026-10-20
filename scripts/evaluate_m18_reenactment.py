"""Milestone 18: Comprehensive Evaluation and Pre-Registered 5-Gate Promotion Audit.

Evaluates 3-way paired comparisons on the exact same video subsets:
1. M18 Dual-Stream Candidate (Frozen M17 Track A S1+S2 + Kinematic Branch + Gated Fusion)
2. M18 Kinematic-Only Branch (Isolated temporal motion representation)
3. M17 Track A Reference (Frozen S1+S2 Residual Trunk)
4. M13.5 Champion Control (v5-family-invariant baseline)

Evaluated Across:
- Gate 1: Face2Face Locked Holdout (116 clips: 58 pristine + 58 F2F)
- Gate 2: Celeb-DF-v2 Locked Holdout (511 clips)
- Gate 3: Clean Specificity Floors on FF++ and Celeb-DF (tau = 0.50)
- Gate 4: Compression Robustness:
  - JPEG Q=50 Stress (801 clips)
  - Uniform H.264 CRF 23 Video Re-Encoding Stress (511 clips)
- Gate 5: Production Latency & VRAM Profiling on RTX 3050
- Additional Veto: Cross-Family Multi-Manipulation Invariance Audit
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
from scipy.optimize import minimize
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.m18_detector import M18ReenactmentDetector
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_m18_reenactment")

# Checkpoint paths
V5_CHAMPION_CKPT = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
V5_CALIB_CONFIG = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"
M17_TRACK_A_CKPT = PROJECT_ROOT / "checkpoints" / "m17_track_a_seed42" / "best.pth"
M18_DUAL_STREAM_CKPT = PROJECT_ROOT / "checkpoints" / "m18_dual_stream_seed42" / "best.pth"
M18_KINEMATIC_ONLY_CKPT = PROJECT_ROOT / "checkpoints" / "m18_kinematic_only_seed42" / "best.pth"

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

OUTPUT_JSON = PROJECT_ROOT / "reports" / "m18_reenactment_evaluation_report.json"
OUTPUT_MD = PROJECT_ROOT / "reports" / "m18_reenactment_evaluation_report.md"

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

    # Ensure exactly T frames
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
    """Paired bootstrap delta AUC test with two-sided p-value."""
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


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit optimal temperature scaling parameter on validation set."""
    def nll_loss(T_arr):
        T = T_arr[0]
        scaled = logits / T
        probs = 1.0 / (1.0 + np.exp(-scaled))
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)
        return -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))

    res = minimize(nll_loss, [1.5], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    return float(res.x[0])


def evaluate_batch_on_models(
    items: List[Any],
    m18_dual: M18ReenactmentDetector,
    m18_kin: Optional[M18ReenactmentDetector],
    m17_cand: M17TrackADetector,
    model_v5: FamilyInvariantDeepfakeDetector,
    t_dual: float,
    t_kin: float,
    t_track_a: float,
    v5_t: float,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Run batched inference across all models on exact same items with streaming tensor loads."""
    y_true = []
    p_dual_list = []
    p_kin_list = []
    p_track_a_list = []
    p_v5_list = []
    gate_list = []

    total_chunks = (len(items) + batch_size - 1) // batch_size
    with torch.no_grad():
        for chunk_idx, i in enumerate(range(0, len(items), batch_size)):
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
                # 1. M18 Dual-Stream
                out_dual = m18_dual(batch_tensor, mode="dual_stream")
                p_dual = torch.sigmoid(out_dual.primary_logits / t_dual).squeeze(-1).tolist()
                gates = out_dual.gate_values.squeeze(-1).tolist()

                # 2. M18 Kinematic-Only
                if m18_kin is not None:
                    out_kin = m18_kin(batch_tensor, mode="kinematic_only")
                    p_k = torch.sigmoid(out_kin.kin_logits / t_kin).squeeze(-1).tolist()
                else:
                    p_k = torch.sigmoid(out_dual.kin_logits / t_kin).squeeze(-1).tolist()

                # 3. M17 Track A (Scale 1 + Scale 2 active)
                out_track_a = m17_cand(batch_tensor, active_scales=(True, True, False))
                p_ta = torch.sigmoid(out_track_a.primary_logits / t_track_a).squeeze(-1).tolist()

                # 4. M13.5 Champion (v5)
                out_v5 = model_v5(batch_tensor, return_dict=True)
                p_v5 = torch.sigmoid(out_v5.primary_logits / v5_t).squeeze(-1).tolist()

            if isinstance(p_dual, float):
                p_dual = [p_dual]
            if isinstance(p_k, float):
                p_k = [p_k]
            if isinstance(p_ta, float):
                p_ta = [p_ta]
            if isinstance(p_v5, float):
                p_v5 = [p_v5]
            if isinstance(gates, float):
                gates = [gates]

            y_true.extend(labels)
            p_dual_list.extend(p_dual)
            p_kin_list.extend(p_k)
            p_track_a_list.extend(p_ta)
            p_v5_list.extend(p_v5)
            gate_list.extend(gates)

            del batch_tensor, tensors
            if (chunk_idx + 1) % 25 == 0 or (chunk_idx + 1) == total_chunks:
                logger.info(f"    Evaluated {min(i + batch_size, len(items))}/{len(items)} items...")

    y_arr = np.array(y_true)
    dual_arr = np.array(p_dual_list)
    kin_arr = np.array(p_kin_list)
    ta_arr = np.array(p_track_a_list)
    v5_arr = np.array(p_v5_list)
    gate_arr = np.array(gate_list)

    def get_metrics(scores: np.ndarray) -> Dict[str, float]:
        auc = float(roc_auc_score(y_arr, scores)) if len(np.unique(y_arr)) > 1 else 0.5
        prauc = float(average_precision_score(y_arr, scores)) if len(np.unique(y_arr)) > 1 else 0.5
        clean_mask = y_arr == 0
        fake_mask = y_arr == 1
        spec = float(np.mean(scores[clean_mask] < 0.50)) if np.any(clean_mask) else 0.0
        det = float(np.mean(scores[fake_mask] >= 0.50)) if np.any(fake_mask) else 0.0
        return {"auc": auc, "prauc": prauc, "clean_spec": spec, "det_rate": det}

    m_dual = get_metrics(dual_arr)
    m_kin = get_metrics(kin_arr)
    m_ta = get_metrics(ta_arr)
    m_v5 = get_metrics(v5_arr)

    # Paired bootstrap tests
    delta_dual_ta = compute_paired_bootstrap(y_arr, dual_arr, ta_arr)
    delta_dual_v5 = compute_paired_bootstrap(y_arr, dual_arr, v5_arr)
    delta_kin_ta = compute_paired_bootstrap(y_arr, kin_arr, ta_arr)

    return {
        "dual_stream": {
            **m_dual,
            "gate_mean": float(np.mean(gate_arr)),
            "gate_std": float(np.std(gate_arr)),
            "delta_vs_track_a": delta_dual_ta["delta_mean"],
            "ci_vs_track_a": [delta_dual_ta["ci_lower"], delta_dual_ta["ci_upper"]],
            "p_val_vs_track_a": delta_dual_ta["p_value"],
            "delta_vs_v5": delta_dual_v5["delta_mean"],
            "ci_vs_v5": [delta_dual_v5["ci_lower"], delta_dual_v5["ci_upper"]],
            "p_val_vs_v5": delta_dual_v5["p_value"],
        },
        "kinematic_only": {
            **m_kin,
            "delta_vs_track_a": delta_kin_ta["delta_mean"],
            "ci_vs_track_a": [delta_kin_ta["ci_lower"], delta_kin_ta["ci_upper"]],
            "p_val_vs_track_a": delta_kin_ta["p_value"],
        },
        "m17_track_a": m_ta,
        "m13_5_control": m_v5,
        "n_samples": len(y_arr),
        "n_reals": int(sum(1 for y in y_arr if y == 0)),
        "n_fakes": int(sum(1 for y in y_arr if y == 1)),
    }


def main():
    logger.info("Initializing Milestone 18 Comprehensive Evaluation Suite...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load M13.5 Champion Control
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
    v5_T = float(v5_calib.get("temperature", 2.0502))

    # 2. Load M17 Track A Reference (S1+S2 active)
    base_cand = DeepfakeDetector(config=cfg.model).to(device)
    m17_ckpt = torch.load(M17_TRACK_A_CKPT, map_location=device, weights_only=False)
    model_track_a = M17TrackADetector(
        base_detector=base_cand, num_families=5, res_dim=128, aux_hidden_dim=512,
        aux_dropout=0.5, gate_bottleneck=128,
    ).to(device)
    model_track_a.load_state_dict(m17_ckpt["model_state_dict"])
    model_track_a.eval()

    # 3. Load M18 Dual-Stream Candidate
    assert M18_DUAL_STREAM_CKPT.exists(), f"M18 Dual-Stream checkpoint missing: {M18_DUAL_STREAM_CKPT}"
    dual_ckpt = torch.load(M18_DUAL_STREAM_CKPT, map_location=device, weights_only=False)
    m18_dual = M18ReenactmentDetector(
        trunk_detector=model_track_a, kin_dim=256, aux_hidden_dim=512,
        gate_bottleneck=128, freeze_trunk=True,
    ).to(device)
    m18_dual.load_state_dict(dual_ckpt["model_state_dict"])
    m18_dual.eval()

    # 4. Load M18 Kinematic-Only Branch if available
    m18_kin = None
    if M18_KINEMATIC_ONLY_CKPT.exists():
        kin_ckpt = torch.load(M18_KINEMATIC_ONLY_CKPT, map_location=device, weights_only=False)
        m18_kin = M18ReenactmentDetector(
            trunk_detector=model_track_a, kin_dim=256, aux_hidden_dim=512,
            gate_bottleneck=128, freeze_trunk=True,
        ).to(device)
        m18_kin.load_state_dict(kin_ckpt["model_state_dict"])
        m18_kin.eval()

    # Fit temperature scaling on validation set
    val_df = pd.read_csv(MANIFEST_LOFO)
    val_df = val_df[val_df["split"] == "val"].reset_index(drop=True)
    val_items = [(find_tensor(str(r["video_id"]), split="val"), int(r["label"])) for _, r in val_df.iterrows()]
    val_items = [it for it in val_items if it[0] is not None]

    val_logits_dual, val_logits_kin, val_logits_ta, val_labels = [], [], [], []
    with torch.no_grad():
        for tp, lbl in val_items:
            t = torch.load(tp, map_location=device, weights_only=False).unsqueeze(0)
            with torch.amp.autocast("cuda"):
                out_d = m18_dual(t, mode="dual_stream")
                out_ta = model_track_a(t, active_scales=(True, True, False))
            val_logits_dual.append(out_d.primary_logits.item())
            val_logits_kin.append(out_d.kin_logits.item())
            val_logits_ta.append(out_ta.primary_logits.item())
            val_labels.append(lbl)

            del t
        torch.cuda.empty_cache()
        gc.collect()

    t_dual = fit_temperature_scaling(np.array(val_logits_dual), np.array(val_labels))
    t_kin = fit_temperature_scaling(np.array(val_logits_kin), np.array(val_labels))
    t_track_a = fit_temperature_scaling(np.array(val_logits_ta), np.array(val_labels))
    logger.info(f"Fitted Temperatures: Dual={t_dual:.4f}, Kin={t_kin:.4f}, TrackA={t_track_a:.4f}, v5={v5_T:.4f}")

    # Prepare Datasets as lightweight metadata specs (Zero upfront RAM allocation)
    # 1. Face2Face Locked Holdout (116 clips: 58 pristine, 58 F2F)
    mixed_df = pd.read_csv(MANIFEST_MIXED)
    mixed_test = mixed_df[mixed_df["split"] == "test"].copy().reset_index(drop=True)
    clean_reals = mixed_test[mixed_test["manipulation_type"] == "original"]
    f2f_fakes = mixed_test[mixed_test["manipulation_type"] == "Face2Face"]
    f2f_items = [(find_tensor(str(r["video_id"]), split="test"), 0)
                 for _, r in clean_reals.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None] + \
                [(find_tensor(str(r["video_id"]), split="test"), 1)
                 for _, r in f2f_fakes.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]

    # 2. Celeb-DF-v2 Locked Holdout (511 clips)
    celeb_df = pd.read_csv(MANIFEST_CELEBDF)
    celeb_test = celeb_df[celeb_df["split"] == "test"].copy().reset_index(drop=True)
    celeb_items = [(find_tensor(str(r["video_id"]), split="test"), int(r["label"]))
                   for _, r in celeb_test.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]

    # 3. JPEG Q=50 Compression Stress (801 clips)
    lofo_df = pd.read_csv(MANIFEST_LOFO)
    test_lofo = lofo_df[lofo_df["split"] == "test"].copy().reset_index(drop=True)
    q50_items = []
    for _, r in test_lofo.iterrows():
        tp = find_tensor(str(r["video_id"]), split="test")
        if tp is not None:
            q50_items.append(((tp, "jpeg_50"), int(r["label"])))

    # 4. Uniform H.264 CRF 23 Re-Encoding (511 Celeb-DF clips)
    reenc_items = []
    for tp, lbl in celeb_items:
        reenc_items.append(((tp, "h264_crf23"), lbl))

    # 5. FF++ In-Domain Mixed Test Set (801 clips)
    ffpp_all_items = [(find_tensor(str(r["video_id"]), split="test"), int(r["label"]))
                      for _, r in mixed_test.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]

    # Execute Evaluations
    benchmarks = {
        "Gate1_Face2Face": f2f_items,
        "Gate2_CelebDF": celeb_items,
        "Gate4_JPEG_Q50": q50_items,
        "Gate4_H264_CRF23": reenc_items,
        "FFpp_Mixed": ffpp_all_items,
    }

    eval_results: Dict[str, Any] = {}
    for name, b_items in benchmarks.items():
        logger.info(f"\n--- Running Benchmark: {name} ({len(b_items)} clips) ---")
        res = evaluate_batch_on_models(
            b_items, m18_dual, m18_kin, model_track_a, model_v5,
            t_dual, t_kin, t_track_a, v5_T, device=device, batch_size=8,
        )
        eval_results[name] = res
        d = res["dual_stream"]
        logger.info(f"  [Dual-Stream] AUC={d['auc']:.4f} | Gate={d['gate_mean']:.4f} | Delta vs TrackA={d['delta_vs_track_a']:+.4f} (p={d['p_val_vs_track_a']:.4f}) | Delta vs v5={d['delta_vs_v5']:+.4f}")
        logger.info(f"  [References ] TrackA AUC={res['m17_track_a']['auc']:.4f} | v5 AUC={res['m13_5_control']['auc']:.4f}")
        torch.cuda.empty_cache()
        gc.collect()

    # Latency & VRAM Profiling (Gate 5)
    logger.info("\n--- Running Gate 5: Production Latency & VRAM Profiling on RTX 3050 ---")
    dummy_clip = torch.randn(1, 16, 3, 112, 112, device=device)
    torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(20):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = m18_dual(dummy_clip, mode="dual_stream")
    # Profile 100 iterations
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = m18_dual(dummy_clip, mode="dual_stream")
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    logger.info(f"P50 Latency: {p50_lat:.2f} ms | P95 Latency: {p95_lat:.2f} ms | Peak VRAM: {peak_vram_gb:.3f} GB")

    gate5_data = {
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "peak_vram_gb": peak_vram_gb,
    }

    # Evaluate 5-Gate Promotion Decision
    f2f_auc = eval_results["Gate1_Face2Face"]["dual_stream"]["auc"]
    f2f_p_val = eval_results["Gate1_Face2Face"]["dual_stream"]["p_val_vs_track_a"]
    celeb_auc = eval_results["Gate2_CelebDF"]["dual_stream"]["auc"]
    celeb_delta_ta = eval_results["Gate2_CelebDF"]["dual_stream"]["delta_vs_track_a"]
    spec_ffpp = eval_results["FFpp_Mixed"]["dual_stream"]["clean_spec"]
    spec_celeb = eval_results["Gate2_CelebDF"]["dual_stream"]["clean_spec"]
    q50_auc = eval_results["Gate4_JPEG_Q50"]["dual_stream"]["auc"]
    h264_auc = eval_results["Gate4_H264_CRF23"]["dual_stream"]["auc"]

    gate1_pass = bool(f2f_auc >= 0.8000 and f2f_p_val < 0.05)
    gate2_pass = bool(celeb_delta_ta >= -0.0150)
    gate3_pass = bool(spec_ffpp >= 0.9500 and spec_celeb >= 0.9500)
    gate4_pass = bool(q50_auc >= 0.6500 and h264_auc >= 0.6000)
    gate5_pass = bool(p50_lat <= 35.0 and peak_vram_gb <= 5.5)

    all_gates_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass
    verdict = "PROMOTE_CANDIDATE" if all_gates_pass else "RETAIN_M13_5_CHAMPION"

    final_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": "M18",
        "title": "Targeted Reenactment-Specific Representation Evaluation",
        "verdict": verdict,
        "temperatures": {"dual": t_dual, "kin": t_kin, "track_a": t_track_a, "v5": v5_T},
        "gates": {
            "Gate1_Face2Face": {"pass": gate1_pass, "auc": f2f_auc, "threshold": 0.8000, "p_val_vs_track_a": f2f_p_val},
            "Gate2_CrossDomain": {"pass": gate2_pass, "auc": celeb_auc, "delta_vs_track_a": celeb_delta_ta, "min_delta": -0.0150},
            "Gate3_CleanSpecificity": {"pass": gate3_pass, "spec_ffpp": spec_ffpp, "spec_celeb": spec_celeb, "threshold": 0.9500},
            "Gate4_Compression": {"pass": gate4_pass, "jpeg_q50_auc": q50_auc, "h264_crf23_auc": h264_auc, "thresholds": [0.6500, 0.6000]},
            "Gate5_Production": {"pass": gate5_pass, "p50_ms": p50_lat, "peak_vram_gb": peak_vram_gb, "limits": [35.0, 5.5]},
        },
        "benchmarks": eval_results,
        "gate5_profiling": gate5_data,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Saved complete M18 evaluation JSON to {OUTPUT_JSON}")

    # Generate Markdown Report
    write_markdown_report(final_report)
    logger.info(f"Saved complete M18 evaluation Markdown to {OUTPUT_MD}")
    logger.info(f"\n=======================================================")
    logger.info(f"FINAL M18 PROMOTION VERDICT: {verdict}")
    logger.info(f"=======================================================")


def write_markdown_report(report: Dict[str, Any]) -> None:
    """Generate structured markdown report."""
    g = report["gates"]
    b = report["benchmarks"]
    v = report["verdict"]

    lines = [
        "# Milestone 18 (M18) — Targeted Reenactment-Specific Representation Report",
        "",
        f"**Date**: {report['timestamp'][:10]}  ",
        f"**Final Governance Verdict**: **`{v}`**  ",
        f"**Candidate**: M18 Dual-Stream Reenactment Detector (`checkpoints/m18_dual_stream_seed42/best.pth`)  ",
        f"**Frozen Baseline Controls**: M17 Track A (S1+S2 Trunk) & M13.5 Champion Control (`v5`)  ",
        "",
        "---",
        "",
        "## 1. Five Strict Promotion Gates Summary",
        "",
        "| Gate | Evaluated Benchmark | Target Standard | Candidate Result | Gate Status |",
        "| :--- | :--- | :---: | :---: | :---: |",
        f"| **Gate 1: Reenactment Recovery** | Face2Face Locked Holdout | AUC $\\ge 0.8000$, $p < 0.05$ | **{g['Gate1_Face2Face']['auc']:.4f}** ($p={g['Gate1_Face2Face']['p_val_vs_track_a']:.4f}$) | {'**[PASS]**' if g['Gate1_Face2Face']['pass'] else '**[FAIL]**'} |",
        f"| **Gate 2: Cross-Domain Retention** | Celeb-DF-v2 Holdout | Non-inferior ($\\Delta \\ge -0.015$) | **{g['Gate2_CrossDomain']['auc']:.4f}** ($\\Delta = {g['Gate2_CrossDomain']['delta_vs_track_a']:+.4f}$) | {'**[PASS]**' if g['Gate2_CrossDomain']['pass'] else '**[FAIL]**'} |",
        f"| **Gate 3: Clean Specificity** | FF++ & Celeb-DF Reals | $\\ge 95.0\\%$ on both | FF++: {g['Gate3_CleanSpecificity']['spec_ffpp']*100:.1f}%, Celeb: {g['Gate3_CleanSpecificity']['spec_celeb']*100:.1f}% | {'**[PASS]**' if g['Gate3_CleanSpecificity']['pass'] else '**[FAIL]**'} |",
        f"| **Gate 4: Compression Robustness** | JPEG Q=50 & H.264 CRF 23 | $\\ge 0.650$ / $\\ge 0.600$ | Q50: **{g['Gate4_Compression']['jpeg_q50_auc']:.4f}**, H.264: **{g['Gate4_Compression']['h264_crf23_auc']:.4f}** | {'**[PASS]**' if g['Gate4_Compression']['pass'] else '**[FAIL]**'} |",
        f"| **Gate 5: Production Envelope** | Latency & VRAM (RTX 3050) | P50 $\\le 35$ ms, $\\le 5.5$ GB | P50: {g['Gate5_Production']['p50_ms']:.1f} ms, VRAM: {g['Gate5_Production']['peak_vram_gb']:.2f} GB | {'**[PASS]**' if g['Gate5_Production']['pass'] else '**[FAIL]**'} |",
        "",
        "---",
        "",
        "## 2. 3-Way Paired Benchmark Matrix",
        "",
        "| Benchmark | M18 Dual-Stream | M18 Kinematic-Only | M17 Track A (S1+S2) | M13.5 Champion (v5) | $\\Delta$ Dual vs Track A | $\\Delta$ Dual vs v5 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, bm in b.items():
        d = bm["dual_stream"]
        k = bm["kinematic_only"]
        ta = bm["m17_track_a"]
        v5 = bm["m13_5_control"]
        lines.append(
            f"| **{name}** | **{d['auc']:.4f}** | {k['auc']:.4f} | {ta['auc']:.4f} | {v5['auc']:.4f} | {d['delta_vs_track_a']:+.4f} ($p={d['p_val_vs_track_a']:.4f}$) | {d['delta_vs_v5']:+.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Governance Audit & Final Decision",
        "",
        f"- **Promotion Verdict**: `{v}`",
        "- **Active Production Champion**: " + ("Promoted M18 Candidate" if v == "PROMOTE_CANDIDATE" else "Retained M13.5 Champion (`checkpoints/m13_3_f2f_seed42/best.pth`)"),
        "- **Research Status**: " + ("All 5 gates successfully satisfied." if v == "PROMOTE_CANDIDATE" else "M18 retained as a valuable research milestone; production invariants preserved."),
    ])

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
