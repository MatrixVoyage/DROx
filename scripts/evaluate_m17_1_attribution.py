"""Milestone 17.1 (M17.1) Forensic Attribution, Scale Disentanglement & Codec Matching.

Evaluates frozen M17 Track A across 7 scale configurations and matched controls:
1. Frozen-weight inference scale ablations across Celeb-DF, JPEG Q=50, F2F, and FF++.
2. Gate value tracking (mean +/- std) per ablation.
3. Paired statistical comparisons (ROC-AUC, PR-AUC, Delta vs Full, Delta vs M13.5, 95% bootstrap CIs, p-values).
4. Codec, resolution, and bitrate matched strata.
5. Uniform H.264 CRF 23 re-encoding of the full Celeb-DF test set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
from scripts.evaluate_m17_track_a import (
    find_tensor, apply_simulated_jpeg, fit_temperature_scaling,
    V5_CHAMPION_CKPT, V5_CALIB_CONFIG, M17_TRACK_A_CKPT,
    MANIFEST_CELEBDF, MANIFEST_LOFO, MANIFEST_MIXED
)

logger = setup_logger("m17_1_attribution")

OUTPUT_JSON = PROJECT_ROOT / "reports" / "m17_1_residual_attribution_report.json"
OUTPUT_MD = PROJECT_ROOT / "reports" / "m17_1_residual_attribution_report.md"

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SCALE_CONFIGS = {
    "Full_S123": (True, True, True),
    "S1_Only": (True, False, False),
    "S2_Only": (False, True, False),
    "S3_Only": (False, False, True),
    "S1_S2": (True, True, False),
    "S1_S3": (True, False, True),
    "S2_S3": (False, True, True),
}


def reencode_tensor_h264_crf23(tensor: torch.Tensor, crf: int = 23) -> torch.Tensor:
    """Uniformly re-encode 16 face frames with x264 CRF 23."""
    arr = tensor.cpu().numpy()
    T, C, H, W = arr.shape
    uint8_frames = []
    for t in range(T):
        f = arr[t].transpose(1, 2, 0)
        f = (f * NORM_STD + NORM_MEAN) * 255.0
        uint8_frames.append(np.clip(f, 0, 255).astype(np.uint8))

    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f_in, tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_out:
        in_p = f_in.name
        out_p = f_out.name

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

    try:
        os.remove(in_p)
        os.remove(out_p)
    except OSError:
        pass

    return torch.from_numpy(np.array(dec_frames, dtype=np.float32)).to(tensor.device)


def compute_paired_bootstrap(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_ref: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap delta AUC test."""
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


def evaluate_dataset_batch(
    items: List[Tuple[torch.Tensor, int]],
    model_cand: M17TrackADetector,
    model_v5: FamilyInvariantDeepfakeDetector,
    T_cand: float,
    v5_T: float,
    scale_cfg: Tuple[bool, bool, bool],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate candidate under scale_cfg and v5 control on items using batched inference."""
    y_true = []
    p_cand_list = []
    p_v5_list = []
    gates_list = []

    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            tensors = [it[0].squeeze(0) if it[0].dim() == 5 else it[0] for it in chunk]
            labels = [it[1] for it in chunk]
            batch_tensor = torch.stack(tensors, dim=0)

            with torch.amp.autocast("cuda"):
                out_cand = model_cand(batch_tensor, active_scales=scale_cfg)
                p_cand = torch.sigmoid(out_cand.primary_logits / T_cand).squeeze(-1).tolist()
                gate_mean = out_cand.gate_values.mean(dim=-1).tolist()

                out_v5 = model_v5(batch_tensor, return_dict=True)
                p_v5 = torch.sigmoid(out_v5.primary_logits / v5_T).squeeze(-1).tolist()

            if isinstance(p_cand, float):
                p_cand = [p_cand]
            if isinstance(p_v5, float):
                p_v5 = [p_v5]
            if isinstance(gate_mean, float):
                gate_mean = [gate_mean]

            y_true.extend(labels)
            p_cand_list.extend(p_cand)
            p_v5_list.extend(p_v5)
            gates_list.extend(gate_mean)

    y_arr = np.array(y_true)
    cand_arr = np.array(p_cand_list)
    v5_arr = np.array(p_v5_list)
    gate_arr = np.array(gates_list)

    auc_cand = float(roc_auc_score(y_arr, cand_arr)) if len(np.unique(y_arr)) > 1 else 0.5
    prauc_cand = float(average_precision_score(y_arr, cand_arr)) if len(np.unique(y_arr)) > 1 else 0.5
    auc_v5 = float(roc_auc_score(y_arr, v5_arr)) if len(np.unique(y_arr)) > 1 else 0.5
    prauc_v5 = float(average_precision_score(y_arr, v5_arr)) if len(np.unique(y_arr)) > 1 else 0.5

    clean_mask = y_arr == 0
    fake_mask = y_arr == 1
    clean_spec = float(np.mean(cand_arr[clean_mask] < 0.50)) if np.any(clean_mask) else 0.0
    det_rate = float(np.mean(cand_arr[fake_mask] >= 0.50)) if np.any(fake_mask) else 0.0

    return {
        "auc_cand": auc_cand,
        "prauc_cand": prauc_cand,
        "auc_v5": auc_v5,
        "prauc_v5": prauc_v5,
        "clean_spec": clean_spec,
        "det_rate": det_rate,
        "gate_mean": float(np.mean(gate_arr)),
        "gate_std": float(np.std(gate_arr)),
        "y_true": y_arr,
        "scores_cand": cand_arr,
        "scores_v5": v5_arr,
    }


def main():
    logger.info("Initializing M17.1 Forensic Attribution and Scale Disentanglement Suite...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen M13.5 champion
    cfg = load_config("configs/experiments/m12_full.yaml")
    base_v5 = DeepfakeDetector(config=cfg.model).to(device)
    v5_ckpt = torch.load(V5_CHAMPION_CKPT, map_location=device, weights_only=False)
    model_v5 = FamilyInvariantDeepfakeDetector(
        base_detector=base_v5, num_families=5, aux_hidden_dim=512, aux_dropout=0.5,
        freq_dim=64, texture_dim=64, fusion_dim=512, enable_mid_head=True
    ).to(device)
    model_v5.load_state_dict(v5_ckpt["model_state_dict"])
    model_v5.eval()

    with open(V5_CALIB_CONFIG, "r", encoding="utf-8") as f:
        v5_calib = json.load(f)
    v5_T = float(v5_calib.get("temperature", 2.0502))

    # Load frozen M17 Track A
    base_cand = DeepfakeDetector(config=cfg.model).to(device)
    m17_ckpt = torch.load(M17_TRACK_A_CKPT, map_location=device, weights_only=False)
    model_cand = M17TrackADetector(
        base_detector=base_cand, num_families=5, res_dim=128, aux_hidden_dim=512,
        aux_dropout=0.5, gate_bottleneck=128
    ).to(device)
    model_cand.load_state_dict(m17_ckpt["model_state_dict"])
    model_cand.eval()

    # Temperature fit
    val_df = pd.read_csv(MANIFEST_LOFO)
    val_df = val_df[val_df["split"] == "val"].reset_index(drop=True)
    val_items = [(find_tensor(str(r["video_id"]), split="val"), int(r["label"])) for _, r in val_df.iterrows()]
    val_items = [it for it in val_items if it[0] is not None]

    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for tp, lbl in val_items:
            tensor = torch.load(tp, map_location=device, weights_only=False).unsqueeze(0)
            with torch.amp.autocast("cuda"):
                out = model_cand(tensor)
            val_logits_list.append(out.primary_logits.item())
            val_labels_list.append(lbl)

    T_cand = fit_temperature_scaling(np.array(val_logits_list), np.array(val_labels_list))
    logger.info(f"Fitted M17 Track A Temperature: T={T_cand:.4f}")

    # =========================================================================
    # PREPARE DATASETS
    # =========================================================================
    # 1. Celeb-DF Full Test Set (511 items)
    celeb_df = pd.read_csv(MANIFEST_CELEBDF)
    celeb_test = celeb_df[celeb_df["split"] == "test"].copy().reset_index(drop=True)
    celeb_items = [(torch.load(find_tensor(str(r["video_id"]), split="test"), map_location=device, weights_only=False), int(r["label"]), r)
                   for _, r in celeb_test.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]
    logger.info(f"Loaded {len(celeb_items)} Celeb-DF test tensors.")

    # 2. JPEG Q=50 Compression Stress (280 items)
    lofo_df = pd.read_csv(MANIFEST_LOFO)
    test_lofo = lofo_df[lofo_df["split"] == "test"].copy().reset_index(drop=True)
    q50_items = []
    for _, r in test_lofo.iterrows():
        tp = find_tensor(str(r["video_id"]), split="test")
        if tp is not None:
            t = torch.load(tp, map_location=device, weights_only=False)
            deg = apply_simulated_jpeg(t, quality=50)
            q50_items.append((deg, int(r["label"])))
    logger.info(f"Loaded {len(q50_items)} JPEG Q=50 degraded tensors.")

    # 3. Face2Face Holdout (116 items: 58 pristine, 58 F2F)
    mixed_df = pd.read_csv(MANIFEST_MIXED)
    mixed_test = mixed_df[mixed_df["split"] == "test"].copy().reset_index(drop=True)
    clean_reals = mixed_test[mixed_test["manipulation_type"] == "original"]
    f2f_fakes = mixed_test[mixed_test["manipulation_type"] == "Face2Face"]
    f2f_items = [(torch.load(find_tensor(str(r["video_id"]), split="test"), map_location=device, weights_only=False), 0)
                 for _, r in clean_reals.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None] + \
                [(torch.load(find_tensor(str(r["video_id"]), split="test"), map_location=device, weights_only=False), 1)
                 for _, r in f2f_fakes.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]
    logger.info(f"Loaded {len(f2f_items)} Face2Face test tensors.")

    # 4. In-Domain FF++ Test (290 items: 58 pristine, 58 DF, 58 FS, 58 NT, 58 F2F)
    ffpp_all_items = [(torch.load(find_tensor(str(r["video_id"]), split="test"), map_location=device, weights_only=False), int(r["label"]))
                      for _, r in mixed_test.iterrows() if find_tensor(str(r["video_id"]), split="test") is not None]
    logger.info(f"Loaded {len(ffpp_all_items)} FF++ mixed test tensors.")

    # =========================================================================
    # PHASE 2: 7 SCALE ABLATIONS ACROSS ALL TARGETS
    # =========================================================================
    experiments = {
        "Celeb-DF-v2": [(t, lbl) for t, lbl, _ in celeb_items],
        "JPEG_Q50": q50_items,
        "Face2Face": f2f_items,
        "FF++_Mixed": ffpp_all_items,
    }

    ablation_results: Dict[str, Any] = {}

    for exp_name, dataset_items in experiments.items():
        logger.info(f"\n--- Running Scale Ablations on {exp_name} ({len(dataset_items)} clips) ---")
        exp_evals: Dict[str, Any] = {}

        # 1. First run Full_S123 to serve as reference
        full_eval = evaluate_dataset_batch(dataset_items, model_cand, model_v5, T_cand, v5_T, SCALE_CONFIGS["Full_S123"])
        y_ref = full_eval["y_true"]
        scores_full = full_eval["scores_cand"]
        scores_v5 = full_eval["scores_v5"]

        delta_full_v5 = compute_paired_bootstrap(y_ref, scores_full, scores_v5)
        full_eval["delta_vs_v5"] = delta_full_v5
        exp_evals["Full_S123"] = {
            "auc": full_eval["auc_cand"],
            "prauc": full_eval["prauc_cand"],
            "clean_spec": full_eval["clean_spec"],
            "det_rate": full_eval["det_rate"],
            "gate_mean": full_eval["gate_mean"],
            "gate_std": full_eval["gate_std"],
            "delta_vs_full": 0.0,
            "delta_vs_v5": delta_full_v5["delta_mean"],
            "ci_vs_v5": [delta_full_v5["ci_lower"], delta_full_v5["ci_upper"]],
            "p_val_vs_v5": delta_full_v5["p_value"],
        }
        logger.info(f"  [Full_S123] AUC={full_eval['auc_cand']:.4f} | Gate={full_eval['gate_mean']:.4f} | Delta v5={delta_full_v5['delta_mean']:+.4f} (p={delta_full_v5['p_value']:.4f})")

        # 2. Run remaining 6 ablations
        for scale_name, cfg_tuple in SCALE_CONFIGS.items():
            if scale_name == "Full_S123":
                continue
            cur_eval = evaluate_dataset_batch(dataset_items, model_cand, model_v5, T_cand, v5_T, cfg_tuple)
            cur_scores = cur_eval["scores_cand"]

            delta_vs_full = compute_paired_bootstrap(y_ref, cur_scores, scores_full)
            delta_vs_v5 = compute_paired_bootstrap(y_ref, cur_scores, scores_v5)

            exp_evals[scale_name] = {
                "auc": cur_eval["auc_cand"],
                "prauc": cur_eval["prauc_cand"],
                "clean_spec": cur_eval["clean_spec"],
                "det_rate": cur_eval["det_rate"],
                "gate_mean": cur_eval["gate_mean"],
                "gate_std": cur_eval["gate_std"],
                "delta_vs_full": delta_vs_full["delta_mean"],
                "ci_vs_full": [delta_vs_full["ci_lower"], delta_vs_full["ci_upper"]],
                "p_val_vs_full": delta_vs_full["p_value"],
                "delta_vs_v5": delta_vs_v5["delta_mean"],
                "ci_vs_v5": [delta_vs_v5["ci_lower"], delta_vs_v5["ci_upper"]],
                "p_val_vs_v5": delta_vs_v5["p_value"],
            }
            logger.info(f"  [{scale_name:10s}] AUC={cur_eval['auc_cand']:.4f} | Gate={cur_eval['gate_mean']:.4f} | Delta Full={delta_vs_full['delta_mean']:+.4f} (p={delta_vs_full['p_value']:.4f}) | Delta v5={delta_vs_v5['delta_mean']:+.4f}")

        # Baseline v5 record
        exp_evals["M13_5_Control"] = {
            "auc": full_eval["auc_v5"],
            "prauc": full_eval["prauc_v5"],
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "delta_vs_full": -delta_full_v5["delta_mean"],
            "delta_vs_v5": 0.0,
        }
        ablation_results[exp_name] = exp_evals

    # =========================================================================
    # PHASE 3: MATCHED CONTROLS & UNIFORM RE-ENCODING ON CELEB-DF
    # =========================================================================
    logger.info("\n--- Phase 3: Codec, Resolution & Bitrate Matched Controls on Celeb-DF ---")
    matched_results: Dict[str, Any] = {}

    # Extract metadata for Celeb-DF test clips
    from scripts.analyze_m17_1_confounds import extract_stream_metadata
    celeb_meta_list = []
    for t, lbl, row in celeb_items:
        m = extract_stream_metadata(row["video_path"])
        m["tensor"] = t
        m["label"] = lbl
        celeb_meta_list.append(m)

    # 1. Resolution-Matched Stratum (Width >= 900, Height = 500)
    res_matched = [(m["tensor"], m["label"]) for m in celeb_meta_list if m.get("width", 0) >= 900 and m.get("height", 0) == 500]
    logger.info(f"Resolution-Matched Stratum (W>=900, H=500): {len(res_matched)} clips (Reals: {sum(1 for _, l in res_matched if l == 0)}, Fakes: {sum(1 for _, l in res_matched if l == 1)})")
    res_eval = evaluate_dataset_batch(res_matched, model_cand, model_v5, T_cand, v5_T, SCALE_CONFIGS["Full_S123"])
    delta_res = compute_paired_bootstrap(res_eval["y_true"], res_eval["scores_cand"], res_eval["scores_v5"])
    matched_results["Resolution_Matched"] = {
        "n_samples": len(res_matched),
        "n_reals": int(sum(1 for _, l in res_matched if l == 0)),
        "n_fakes": int(sum(1 for _, l in res_matched if l == 1)),
        "auc_cand": res_eval["auc_cand"],
        "auc_v5": res_eval["auc_v5"],
        "delta_auc": delta_res["delta_mean"],
        "ci": [delta_res["ci_lower"], delta_res["ci_upper"]],
        "p_val": delta_res["p_value"],
        "clean_spec": res_eval["clean_spec"],
    }
    logger.info(f"  Resolution-Matched AUC: Cand={res_eval['auc_cand']:.4f} | v5={res_eval['auc_v5']:.4f} | Delta={delta_res['delta_mean']:+.4f} (p={delta_res['p_value']:.4f})")

    # 2. Bitrate-Matched Stratum (900 <= Bitrate <= 1400 kbps)
    bitrate_matched = [(m["tensor"], m["label"]) for m in celeb_meta_list if 900.0 <= m.get("bitrate_kbps", 0) <= 1400.0]
    logger.info(f"Bitrate-Matched Stratum (900-1400 kbps): {len(bitrate_matched)} clips (Reals: {sum(1 for _, l in bitrate_matched if l == 0)}, Fakes: {sum(1 for _, l in bitrate_matched if l == 1)})")
    bit_eval = evaluate_dataset_batch(bitrate_matched, model_cand, model_v5, T_cand, v5_T, SCALE_CONFIGS["Full_S123"])
    delta_bit = compute_paired_bootstrap(bit_eval["y_true"], bit_eval["scores_cand"], bit_eval["scores_v5"])
    matched_results["Bitrate_Matched"] = {
        "n_samples": len(bitrate_matched),
        "n_reals": int(sum(1 for _, l in bitrate_matched if l == 0)),
        "n_fakes": int(sum(1 for _, l in bitrate_matched if l == 1)),
        "auc_cand": bit_eval["auc_cand"],
        "auc_v5": bit_eval["auc_v5"],
        "delta_auc": delta_bit["delta_mean"],
        "ci": [delta_bit["ci_lower"], delta_bit["ci_upper"]],
        "p_val": delta_bit["p_value"],
        "clean_spec": bit_eval["clean_spec"],
    }
    logger.info(f"  Bitrate-Matched AUC: Cand={bit_eval['auc_cand']:.4f} | v5={bit_eval['auc_v5']:.4f} | Delta={delta_bit['delta_mean']:+.4f} (p={delta_bit['p_value']:.4f})")

    # 3. Complete Uniform H.264 CRF 23 Re-Encoding Control on all 511 Celeb-DF test clips
    logger.info("Applying uniform H.264 CRF 23 re-encoding across all 511 Celeb-DF test clips...")
    t0_reenc = time.time()
    reenc_items = []
    for i, (t, lbl, _) in enumerate(celeb_items):
        t_reenc = reencode_tensor_h264_crf23(t, crf=23)
        reenc_items.append((t_reenc, lbl))
        if (i + 1) % 100 == 0 or (i + 1) == len(celeb_items):
            logger.info(f"  Re-encoded {i + 1}/{len(celeb_items)} clips ({time.time() - t0_reenc:.1f}s)...")

    reenc_eval = evaluate_dataset_batch(reenc_items, model_cand, model_v5, T_cand, v5_T, SCALE_CONFIGS["Full_S123"])
    delta_reenc = compute_paired_bootstrap(reenc_eval["y_true"], reenc_eval["scores_cand"], reenc_eval["scores_v5"])

    # Also evaluate S1, S2, S3 under re-encoding
    s1_reenc = evaluate_dataset_batch(reenc_items, model_cand, model_v5, T_cand, v5_T, SCALE_CONFIGS["S1_Only"])
    s2_reenc = evaluate_dataset_batch(reenc_items, model_cand, model_v5, T_cand, v5_T, SCALE_CONFIGS["S2_Only"])
    s3_reenc = evaluate_dataset_batch(reenc_items, model_cand, model_v5, T_cand, v5_T, SCALE_CONFIGS["S3_Only"])

    matched_results["Uniform_H264_CRF23"] = {
        "n_samples": len(reenc_items),
        "crf": 23,
        "codec": "libx264",
        "full_s123_auc": reenc_eval["auc_cand"],
        "s1_only_auc": s1_reenc["auc_cand"],
        "s2_only_auc": s2_reenc["auc_cand"],
        "s3_only_auc": s3_reenc["auc_cand"],
        "auc_v5": reenc_eval["auc_v5"],
        "delta_full_vs_v5": delta_reenc["delta_mean"],
        "ci_vs_v5": [delta_reenc["ci_lower"], delta_reenc["ci_upper"]],
        "p_val_vs_v5": delta_reenc["p_value"],
        "clean_spec": reenc_eval["clean_spec"],
        "reencode_time_sec": round(time.time() - t0_reenc, 1),
    }
    logger.info(f"  Uniform H.264 Re-encoded AUC: Cand={reenc_eval['auc_cand']:.4f} | v5={reenc_eval['auc_v5']:.4f} | Delta={delta_reenc['delta_mean']:+.4f} (p={delta_reenc['p_value']:.4f})")
    logger.info(f"  Scales under Re-encoding: S1={s1_reenc['auc_cand']:.4f} | S2={s2_reenc['auc_cand']:.4f} | S3={s3_reenc['auc_cand']:.4f}")

    # =========================================================================
    # SAVE JSON REPORT
    # =========================================================================
    final_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": "M17.1",
        "title": "Forensic Attribution, Scale Disentanglement, and Codec Matching for M17 Track A",
        "checkpoint": str(M17_TRACK_A_CKPT),
        "temperature": T_cand,
        "scale_ablations": ablation_results,
        "matched_controls": matched_results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Saved complete M17.1 attribution JSON to {OUTPUT_JSON}")

    # =========================================================================
    # WRITE COMPREHENSIVE MARKDOWN SYNTHESIS
    # =========================================================================
    write_markdown_report(final_report)
    logger.info(f"Saved complete M17.1 attribution Markdown to {OUTPUT_MD}")


def write_markdown_report(report: Dict[str, Any]) -> None:
    """Generate structured markdown synthesis report."""
    ab = report["scale_ablations"]
    mc = report["matched_controls"]

    md = f"""# Milestone 17.1 (M17.1) — Forensic Attribution, Scale Disentanglement & Codec Matching
## Empirical Attribution of M17 Track A Multi-Scale Residuals

**Date**: {report['timestamp'][:10]}  
**Model Under Test**: M17 Track A Candidate (`checkpoints/m17_track_a_seed42/best.pth`, Frozen)  
**Control Baseline**: M13.5 Champion Control (`checkpoints/m13_3_f2f_seed42/best.pth`, Frozen)  
**Calibrated Temperature**: $T = {report['temperature']:.4f}$  

---

## 1. Core Answers to the 5 Research Questions

### Q1: Which residual scales provide the Celeb-DF improvement?
Under the frozen model, comparative scale ablations reveal:
- **Full Pyramidal Residuals ($S1+S2+S3$)**: **{ab['Celeb-DF-v2']['Full_S123']['auc']:.4f}** ROC-AUC ($\Delta = {ab['Celeb-DF-v2']['Full_S123']['delta_vs_v5']:+.4f}$ vs M13.5, $p = {ab['Celeb-DF-v2']['Full_S123']['p_val_vs_v5']:.4f}$)
- **Scale 1 Only ($112\\times 112$ SRM+Bayar)**: **{ab['Celeb-DF-v2']['S1_Only']['auc']:.4f}** ROC-AUC ($\Delta = {ab['Celeb-DF-v2']['S1_Only']['delta_vs_full']:+.4f}$ vs Full, Gate Mean = {ab['Celeb-DF-v2']['S1_Only']['gate_mean']:.4f})
- **Scale 2 Only ($56\\times 56$ Bayar Resampling)**: **{ab['Celeb-DF-v2']['S2_Only']['auc']:.4f}** ROC-AUC ($\Delta = {ab['Celeb-DF-v2']['S2_Only']['delta_vs_full']:+.4f}$ vs Full, Gate Mean = {ab['Celeb-DF-v2']['S2_Only']['gate_mean']:.4f})
- **Scale 3 Only ($28\\times 28$ Macro Block)**: **{ab['Celeb-DF-v2']['S3_Only']['auc']:.4f}** ROC-AUC ($\Delta = {ab['Celeb-DF-v2']['S3_Only']['delta_vs_full']:+.4f}$ vs Full, Gate Mean = {ab['Celeb-DF-v2']['S3_Only']['gate_mean']:.4f})

### Q2: Which scales survive JPEG Q=50 compression?
- **Full Model under Q=50**: **{ab['JPEG_Q50']['Full_S123']['auc']:.4f}** (Control = {ab['JPEG_Q50']['M13_5_Control']['auc']:.4f}, $\Delta = {ab['JPEG_Q50']['Full_S123']['delta_vs_v5']:+.4f}$)
- **Scale 1 Only under Q=50**: **{ab['JPEG_Q50']['S1_Only']['auc']:.4f}**
- **Scale 2 Only under Q=50**: **{ab['JPEG_Q50']['S2_Only']['auc']:.4f}**
- **Scale 3 Only under Q=50**: **{ab['JPEG_Q50']['S3_Only']['auc']:.4f}**

### Q3: Which scales contribute to Face2Face detection?
- **Full Model on F2F**: **{ab['Face2Face']['Full_S123']['auc']:.4f}** (Control = {ab['Face2Face']['M13_5_Control']['auc']:.4f})
- **Scale 1 Only on F2F**: **{ab['Face2Face']['S1_Only']['auc']:.4f}**
- **Scale 2 Only on F2F**: **{ab['Face2Face']['S2_Only']['auc']:.4f}**
- **Scale 3 Only on F2F**: **{ab['Face2Face']['S3_Only']['auc']:.4f}**

### Q4 & Q5: Does Track A survive matched codec, resolution, and bitrate controls?
1. **Resolution-Matched Stratum ($W \\ge 900, H = 500$)**:
   - Cand AUC: **{mc['Resolution_Matched']['auc_cand']:.4f}** vs Control: **{mc['Resolution_Matched']['auc_v5']:.4f}** ($\Delta = {mc['Resolution_Matched']['delta_auc']:+.4f}$, 95% CI: [{mc['Resolution_Matched']['ci'][0]:+.4f}, {mc['Resolution_Matched']['ci'][1]:+.4f}], $p = {mc['Resolution_Matched']['p_val']:.4f}$)
2. **Bitrate-Matched Stratum ($900 \\le \\text{{Bitrate}} \\le 1400$ kbps)**:
   - Cand AUC: **{mc['Bitrate_Matched']['auc_cand']:.4f}** vs Control: **{mc['Bitrate_Matched']['auc_v5']:.4f}** ($\Delta = {mc['Bitrate_Matched']['delta_auc']:+.4f}$, 95% CI: [{mc['Bitrate_Matched']['ci'][0]:+.4f}, {mc['Bitrate_Matched']['ci'][1]:+.4f}], $p = {mc['Bitrate_Matched']['p_val']:.4f}$)
3. **Uniform H.264 CRF 23 Re-Encoding (All 511 Clips)**:
   - Cand AUC: **{mc['Uniform_H264_CRF23']['full_s123_auc']:.4f}** vs Control: **{mc['Uniform_H264_CRF23']['auc_v5']:.4f}** ($\Delta = {mc['Uniform_H264_CRF23']['delta_full_vs_v5']:+.4f}$, 95% CI: [{mc['Uniform_H264_CRF23']['ci_vs_v5'][0]:+.4f}, {mc['Uniform_H264_CRF23']['ci_vs_v5'][1]:+.4f}], $p = {mc['Uniform_H264_CRF23']['p_val_vs_v5']:.4f}$)
   - S1 Re-encoded: {mc['Uniform_H264_CRF23']['s1_only_auc']:.4f} | S2 Re-encoded: {mc['Uniform_H264_CRF23']['s2_only_auc']:.4f} | S3 Re-encoded: {mc['Uniform_H264_CRF23']['s3_only_auc']:.4f}

---

## 2. Complete Frozen-Weight Inference Scale Ablation Matrix

| Scale Configuration | Celeb-DF AUC (PR-AUC) | $\\Delta$ vs Full | $\\Delta$ vs M13.5 | JPEG Q=50 AUC | F2F AUC | FF++ Mixed AUC | Gate Mean $\\pm$ Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for cfg_name in SCALE_CONFIGS.keys():
        c_auc = ab["Celeb-DF-v2"][cfg_name]["auc"]
        c_pr = ab["Celeb-DF-v2"][cfg_name]["prauc"]
        d_full = ab["Celeb-DF-v2"][cfg_name]["delta_vs_full"]
        d_v5 = ab["Celeb-DF-v2"][cfg_name]["delta_vs_v5"]
        q50_auc = ab["JPEG_Q50"][cfg_name]["auc"]
        f2f_auc = ab["Face2Face"][cfg_name]["auc"]
        ff_auc = ab["FF++_Mixed"][cfg_name]["auc"]
        g_mean = ab["Celeb-DF-v2"][cfg_name]["gate_mean"]
        g_std = ab["Celeb-DF-v2"][cfg_name]["gate_std"]

        md += f"| **{cfg_name}** | {c_auc:.4f} ({c_pr:.4f}) | {d_full:+.4f} | {d_v5:+.4f} | {q50_auc:.4f} | {f2f_auc:.4f} | {ff_auc:.4f} | {g_mean:.3f} $\\pm$ {g_std:.3f} |\n"

    ctrl_c = ab["Celeb-DF-v2"]["M13_5_Control"]["auc"]
    ctrl_pr = ab["Celeb-DF-v2"]["M13_5_Control"]["prauc"]
    ctrl_q50 = ab["JPEG_Q50"]["M13_5_Control"]["auc"]
    ctrl_f2f = ab["Face2Face"]["M13_5_Control"]["auc"]
    ctrl_ff = ab["FF++_Mixed"]["M13_5_Control"]["auc"]
    md += f"| **M13.5 Control (`v5`)** | {ctrl_c:.4f} ({ctrl_pr:.4f}) | {ctrl_c - ab['Celeb-DF-v2']['Full_S123']['auc']:+.4f} | 0.0000 | {ctrl_q50:.4f} | {ctrl_f2f:.4f} | {ctrl_ff:.4f} | 0.000 $\\pm$ 0.000 |\n"

    md += """
---

## 3. Scientific Conclusions for M18 Foundation

1. **Leakage vs. Forensic Evidence Finding**:
   - Evaluates whether Track A's advantage survives when resolution, bitrate, and codec are strictly matched or re-encoded.
2. **Scale Hierarchy**:
   - Identifies whether the cross-domain signal is concentrated in high frequencies (Scale 1), resampling boundaries (Scale 2), or macro blocks (Scale 3).
3. **Implications for M18 Architecture**:
   - Confirms which residual scales provide genuine invariant forensic grounding before attaching targeted reenactment supervision.
"""
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
