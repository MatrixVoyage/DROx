"""Authoritative Evaluation Harness for Milestone 13.2.

Evaluates M12 Frozen Baseline, M13.2a-Frozen, and M13.2a-Joint across the exact
same fixed 560-sample benchmark slices:
1. Clean Real (98 queries)
2. In-Domain Seen Fake (174 queries)
3. Celeb-DF Cross-Dataset (74 queries)
4. Held-Out Face2Face (58 queries)
5. Degraded (JPEG) (98 queries)
6. Synthetic OOD (58 queries)

Produces reproducible JSON and Markdown comparative reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augment import RobustAugmentor, RobustAugmentorConfig
from src.models.detector import DeepfakeDetector
from src.models.multi_head_detector import MultiHeadDeepfakeDetector
from src.utils.config import ModelConfig, _dict_to_config
from src.utils.logger import setup_logger

logger = setup_logger("m13_2_eval")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
FREEZE_MANIFEST_PATH = PROJECT_ROOT / "reports" / "m12_baseline_freeze.json"
CALIB_CONFIG_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "calibration_config.json"
M13_2A_FROZEN_CKPT = PROJECT_ROOT / "checkpoints" / "m13_2a_frozen" / "best.pth"
M13_2A_JOINT_CKPT = PROJECT_ROOT / "checkpoints" / "m13_2a_joint" / "best.pth"
M13_2B_STAGE5_CKPT = PROJECT_ROOT / "checkpoints" / "m13_2b_stage5" / "best.pth"

REPORT_JSON = PROJECT_ROOT / "reports" / "m13_2_auxiliary_evaluation.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m13_2_auxiliary_report.md"


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
    assert current_hash == expected_hash, f"CRITICAL: M12 checkpoint modified! {current_hash} != {expected_hash}"
    logger.info(f"M12 integrity verified. Hash: {current_hash[:16]}...")
    return current_hash


def compute_bootstrap_ci(data: np.ndarray, num_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval."""
    if len(data) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(num_bootstrap)]
    alpha = (1.0 - ci) / 2.0
    return (float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1.0 - alpha))))


def load_evaluation_slices(tensors_dir: Path) -> Dict[str, pd.DataFrame]:
    """Build the exact standardized 560-sample evaluation slice dictionary."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    test_df = test_df[test_df["video_id"].apply(lambda v: (tensors_dir / f"{v}.pt").is_file())].reset_index(drop=True)

    clean_real = test_df[test_df["label"] == 0].reset_index(drop=True)
    seen_fake = test_df[(test_df["dataset"] == "ffpp") & (test_df["label"] == 1) & (test_df["manipulation_type"] != "Face2Face")].reset_index(drop=True)
    cross_fake = test_df[(test_df["dataset"] == "celebdf") & (test_df["label"] == 1)].reset_index(drop=True)
    f2f = test_df[test_df["manipulation_type"] == "Face2Face"].reset_index(drop=True)

    return {
        "Clean Real": clean_real,
        "In-Domain Seen Fake": seen_fake,
        "Celeb-DF Cross-Dataset": cross_fake,
        "Held-Out Face2Face": f2f,
    }


def evaluate_model_on_benchmark(
    model_name: str,
    model: nn.Module,
    slices_df: Dict[str, pd.DataFrame],
    tensors_dir: Path,
    device: torch.device,
    is_multi_head: bool = False,
    temperature: float = 1.6369,
) -> Dict[str, Any]:
    """Run full standardized evaluation on a candidate model."""
    logger.info(f"Evaluating model: {model_name}...")
    model.eval()

    jpeg_aug = RobustAugmentor(
        RobustAugmentorConfig(jpeg_p=1.0, blur_p=0.0, noise_p=0.0, resize_p=0.0, reencode_p=0.0, frame_drop_p=0.0),
        seed=42,
    )

    records = []
    latencies = []

    # 1. Standard slices
    for s_name, s_df in slices_df.items():
        for _, row in s_df.iterrows():
            vid = row["video_id"]
            lbl = int(row["label"])
            t = torch.load(tensors_dir / f"{vid}.pt", map_location="cpu", weights_only=False).unsqueeze(0).to(device)

            start = time.perf_counter()
            with torch.no_grad():
                if is_multi_head:
                    out = model(t, return_dict=True)
                    p_prim = torch.sigmoid(out.primary_logits / temperature).item()
                    p_aux = out.p_aux_fake.item()
                    p_fused = (0.5 * p_prim + 0.5 * p_aux)
                    raw_logit = out.primary_logits.item()
                else:
                    logits = model(t)
                    p_prim = torch.sigmoid(logits / temperature).item()
                    p_aux = p_prim
                    p_fused = p_prim
                    raw_logit = logits.item()
            latencies.append((time.perf_counter() - start) * 1000.0)

            records.append({
                "slice": s_name,
                "video_id": vid,
                "true_label": lbl,
                "dataset": row["dataset"],
                "manipulation_type": row["manipulation_type"],
                "raw_logit": raw_logit,
                "p_primary": p_prim,
                "p_aux": p_aux,
                "p_fused": p_fused,
            })

    # 2. Degraded slice (JPEG on Clean Real + In-Domain Fake)
    n_deg = min(len(slices_df["Clean Real"]), len(slices_df["In-Domain Seen Fake"]))
    deg_samples = pd.concat([slices_df["Clean Real"].iloc[:n_deg//2], slices_df["In-Domain Seen Fake"].iloc[:n_deg//2]])
    for _, row in deg_samples.iterrows():
        vid = row["video_id"]
        lbl = int(row["label"])
        t = torch.load(tensors_dir / f"{vid}.pt", map_location="cpu", weights_only=False)
        t_aug, _ = jpeg_aug.augment_tensor(t)
        t_aug = t_aug.unsqueeze(0).to(device)

        with torch.no_grad():
            if is_multi_head:
                out = model(t_aug, return_dict=True)
                p_prim = torch.sigmoid(out.primary_logits / temperature).item()
                p_aux = out.p_aux_fake.item()
                p_fused = 0.5 * p_prim + 0.5 * p_aux
                raw_logit = out.primary_logits.item()
            else:
                logits = model(t_aug)
                p_prim = torch.sigmoid(logits / temperature).item()
                p_aux = p_prim
                p_fused = p_prim
                raw_logit = logits.item()

        records.append({
            "slice": "Degraded (JPEG)",
            "video_id": vid,
            "true_label": lbl,
            "dataset": row["dataset"],
            "manipulation_type": row["manipulation_type"],
            "raw_logit": raw_logit,
            "p_primary": p_prim,
            "p_aux": p_aux,
            "p_fused": p_fused,
        })

    # 3. Synthetic OOD slice
    rng = np.random.default_rng(42)
    for i in range(len(slices_df["Held-Out Face2Face"])):
        noise_arr = rng.normal(loc=0.0, scale=1.0, size=(16, 3, 112, 112)).astype(np.float32)
        noise_t = torch.from_numpy(noise_arr).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_multi_head:
                out = model(noise_t, return_dict=True)
                p_prim = torch.sigmoid(out.primary_logits / temperature).item()
                p_aux = out.p_aux_fake.item()
                p_fused = 0.5 * p_prim + 0.5 * p_aux
                raw_logit = out.primary_logits.item()
            else:
                logits = model(noise_t)
                p_prim = torch.sigmoid(logits / temperature).item()
                p_aux = p_prim
                p_fused = p_prim
                raw_logit = logits.item()

        records.append({
            "slice": "Synthetic OOD",
            "video_id": f"syn_ood_{i:03d}",
            "true_label": 0,
            "dataset": "synthetic",
            "manipulation_type": "gaussian_noise",
            "raw_logit": raw_logit,
            "p_primary": p_prim,
            "p_aux": p_aux,
            "p_fused": p_fused,
        })

    df = pd.DataFrame(records)

    # Compute slice metrics
    slice_summaries = {}
    for s_name in df["slice"].unique():
        sdf = df[df["slice"] == s_name]
        p_prim_vals = sdf["p_primary"].values
        p_aux_vals = sdf["p_aux"].values
        p_fused_vals = sdf["p_fused"].values

        ci_prim = compute_bootstrap_ci(p_prim_vals)
        ci_aux = compute_bootstrap_ci(p_aux_vals)
        ci_fused = compute_bootstrap_ci(p_fused_vals)

        slice_summaries[s_name] = {
            "n": len(sdf),
            "p_primary_mean": round(float(np.mean(p_prim_vals)), 4),
            "p_primary_median": round(float(np.median(p_prim_vals)), 4),
            "p_primary_ci": [round(ci_prim[0], 4), round(ci_prim[1], 4)],
            "p_aux_mean": round(float(np.mean(p_aux_vals)), 4),
            "p_aux_median": round(float(np.median(p_aux_vals)), 4),
            "p_aux_ci": [round(ci_aux[0], 4), round(ci_aux[1], 4)],
            "p_fused_mean": round(float(np.mean(p_fused_vals)), 4),
            "p_fused_median": round(float(np.median(p_fused_vals)), 4),
            "p_fused_ci": [round(ci_fused[0], 4), round(ci_fused[1], 4)],
            "det_rate_tau_0_5": round(float(np.mean(p_fused_vals >= 0.5)), 4),
        }

    # Benchmark Area Under Curve metrics
    # 1. Clean Real vs Face2Face
    cr_df = df[df["slice"] == "Clean Real"]
    f2f_df = df[df["slice"] == "Held-Out Face2Face"]
    cr_f2f_df = pd.concat([cr_df, f2f_df])
    y_cr_f2f = (cr_f2f_df["slice"] == "Held-Out Face2Face").astype(int).values

    f2f_auc_prim = float(roc_auc_score(y_cr_f2f, cr_f2f_df["p_primary"].values))
    f2f_auc_aux = float(roc_auc_score(y_cr_f2f, cr_f2f_df["p_aux"].values))
    f2f_auc_fused = float(roc_auc_score(y_cr_f2f, cr_f2f_df["p_fused"].values))

    # 2. In-domain Seen Fake vs Clean Real
    seen_df = df[df["slice"] == "In-Domain Seen Fake"]
    cr_seen_df = pd.concat([cr_df, seen_df])
    y_seen = (cr_seen_df["slice"] == "In-Domain Seen Fake").astype(int).values
    seen_auc = float(roc_auc_score(y_seen, cr_seen_df["p_fused"].values))
    seen_f1 = float(f1_score(y_seen, (cr_seen_df["p_fused"].values >= 0.5).astype(int)))

    # 3. Celeb-DF Cross-Dataset vs Clean Real
    cross_df = df[df["slice"] == "Celeb-DF Cross-Dataset"]
    cr_cross_df = pd.concat([cr_df, cross_df])
    y_cross = (cr_cross_df["slice"] == "Celeb-DF Cross-Dataset").astype(int).values
    cross_auc = float(roc_auc_score(y_cross, cr_cross_df["p_fused"].values))
    cross_f1 = float(f1_score(y_cross, (cr_cross_df["p_fused"].values >= 0.5).astype(int)))

    # 4. Degraded Slice AUC
    deg_df = df[df["slice"] == "Degraded (JPEG)"]
    deg_auc = float(roc_auc_score(deg_df["true_label"].values, deg_df["p_fused"].values))

    # Computational metrics
    param_count = sum(p.numel() for p in model.parameters())
    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    return {
        "model_name": model_name,
        "parameters": param_count,
        "latency_p50_ms": round(float(np.median(latencies)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "peak_vram_mb": round(vram_mb, 2),
        "slices": slice_summaries,
        "metrics": {
            "face2face_auc_primary": round(f2f_auc_prim, 4),
            "face2face_auc_aux": round(f2f_auc_aux, 4),
            "face2face_auc_fused": round(f2f_auc_fused, 4),
            "face2face_detection_rate": slice_summaries["Held-Out Face2Face"]["det_rate_tau_0_5"],
            "seen_auc": round(seen_auc, 4),
            "seen_f1": round(seen_f1, 4),
            "cross_dataset_auc": round(cross_auc, 4),
            "cross_dataset_f1": round(cross_f1, 4),
            "degraded_auc": round(deg_auc, 4),
        }
    }


def main():
    logger.info("=" * 70)
    logger.info("Executing Milestone 13.2 Comparative Evaluation Harness")
    logger.info("=" * 70)

    # 1. Baseline Integrity check
    m12_hash = verify_m12_integrity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation compute device: {device}")

    with open(CALIB_CONFIG_PATH) as f:
        calib_cfg = json.load(f)
    temperature = float(calib_cfg.get("temperature", 1.6369))

    tensors_dir = PROJECT_ROOT / "processed" / "m12_full" / "test" / "tensors"
    slices_df = load_evaluation_slices(tensors_dir)

    all_results = {}

    # -------------------------------------------------------------
    # 1. Evaluate Frozen M12 Baseline (Reference)
    # -------------------------------------------------------------
    logger.info("\n[1/3] Loading Frozen M12 Baseline...")
    m12_ckpt = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    raw_cfg = m12_ckpt["config"]
    m12_config = _dict_to_config(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    m12_model = DeepfakeDetector(m12_config.model).to(device)
    m12_model.load_state_dict(m12_ckpt["model_state_dict"])
    m12_results = evaluate_model_on_benchmark(
        "Frozen M12 Baseline", m12_model, slices_df, tensors_dir, device, is_multi_head=False, temperature=temperature
    )
    all_results["Frozen M12 Baseline"] = m12_results

    # -------------------------------------------------------------
    # 2. Evaluate M13.2a-Frozen (if checkpoint exists)
    # -------------------------------------------------------------
    if M13_2A_FROZEN_CKPT.exists():
        logger.info("\n[2/3] Loading M13.2a-Frozen (Frozen Trunk + Auxiliary Head)...")
        f_ckpt = torch.load(M13_2A_FROZEN_CKPT, map_location=device, weights_only=False)
        m13_frozen_model = MultiHeadDeepfakeDetector(
            base_detector=DeepfakeDetector(m12_config.model),
            num_families=5,
        ).to(device)
        m13_frozen_model.load_state_dict(f_ckpt["model_state_dict"])
        frozen_results = evaluate_model_on_benchmark(
            "M13.2a-Frozen", m13_frozen_model, slices_df, tensors_dir, device, is_multi_head=True, temperature=temperature
        )
        all_results["M13.2a-Frozen"] = frozen_results
    else:
        logger.warning(f"Checkpoint not found for M13.2a-Frozen at {M13_2A_FROZEN_CKPT}")

    # -------------------------------------------------------------
    # 3. Evaluate M13.2a-Joint (if checkpoint exists)
    # -------------------------------------------------------------
    if M13_2A_JOINT_CKPT.exists():
        logger.info("\n[3/3] Loading M13.2a-Joint (Multi-Task Fine-Tuning)...")
        j_ckpt = torch.load(M13_2A_JOINT_CKPT, map_location=device, weights_only=False)
        m13_joint_model = MultiHeadDeepfakeDetector(
            base_detector=DeepfakeDetector(m12_config.model),
            num_families=5,
        ).to(device)
        m13_joint_model.load_state_dict(j_ckpt["model_state_dict"])
        joint_results = evaluate_model_on_benchmark(
            "M13.2a-Joint", m13_joint_model, slices_df, tensors_dir, device, is_multi_head=True, temperature=temperature
        )
        all_results["M13.2a-Joint"] = joint_results
    else:
        logger.warning(f"Checkpoint not found for M13.2a-Joint at {M13_2A_JOINT_CKPT}")

    # -------------------------------------------------------------
    # 4. Evaluate M13.2b-Stage5 (if checkpoint exists)
    # -------------------------------------------------------------
    if M13_2B_STAGE5_CKPT.exists():
        logger.info("\n[4/4] Loading M13.2b-Stage5 (Stage-5 Head + Penultimate Head)...")
        b_ckpt = torch.load(M13_2B_STAGE5_CKPT, map_location=device, weights_only=False)
        m13_b_model = MultiHeadDeepfakeDetector(
            base_detector=DeepfakeDetector(m12_config.model),
            num_families=5,
            enable_mid_head=True,
            mid_in_channels=112,
            mid_hidden_dim=128,
        ).to(device)
        m13_b_model.load_state_dict(b_ckpt["model_state_dict"])
        b_results = evaluate_model_on_benchmark(
            "M13.2b-Stage5", m13_b_model, slices_df, tensors_dir, device, is_multi_head=True, temperature=temperature
        )
        all_results["M13.2b-Stage5"] = b_results
    else:
        logger.warning(f"Checkpoint not found for M13.2b-Stage5 at {M13_2B_STAGE5_CKPT}")

    # Save JSON report
    with open(REPORT_JSON, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "m12_checkpoint_sha256": m12_hash,
                "eval_slice_counts": {k: len(v) for k, v in slices_df.items()},
                "models": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved evaluation JSON to {REPORT_JSON}")

    # Write Markdown comparative report
    write_markdown_report(all_results, m12_hash)

    # Post-execution integrity check
    verify_m12_integrity()
    logger.info("Evaluation harness complete.")


def write_markdown_report(all_results: Dict[str, Any], m12_hash: str) -> None:
    """Generate Markdown report comparing all models side-by-side."""
    lines = [
        "# Milestone 13.2: Unseen-Manipulation Generalization — Experimental Matrix",
        "",
        f"**M12 Baseline SHA-256**: `{m12_hash}` (Bitwise Frozen Benchmark)  ",
        f"**Evaluation Benchmark**: Fixed 560-Query Authoritative Benchmark  ",
        "**Strict Invariant**: Zero test/Face2Face data leakage. Preprocessing & sampling identical across all models.",
        "",
        "---",
        "",
        "## 1. Executive Summary & Comparative Matrix",
        "",
        "| Evaluation Metric | Frozen M12 Baseline | M13.2a-Frozen (Trunk Frozen) | M13.2a-Joint (Multi-Task Fine-Tuned) | M13.2b-Stage5 (Stage-5 + Penultimate) |",
        "|---|---|---|---|---|",
    ]

    m12 = all_results.get("Frozen M12 Baseline", {}).get("metrics", {})
    m_fro = all_results.get("M13.2a-Frozen", {}).get("metrics", {})
    m_jnt = all_results.get("M13.2a-Joint", {}).get("metrics", {})
    m_b = all_results.get("M13.2b-Stage5", {}).get("metrics", {})

    def get_val(d: Dict[str, Any], k: str, pct: bool = False) -> str:
        if k not in d:
            return "N/A"
        val = d[k]
        return f"{val * 100:.2f}%" if pct else f"{val:.4f}"

    lines.append(f"| **Held-Out Face2Face ROC-AUC** | {get_val(m12, 'face2face_auc_primary')} | {get_val(m_fro, 'face2face_auc_fused')} | {get_val(m_jnt, 'face2face_auc_fused')} | **{get_val(m_b, 'face2face_auc_fused')}** |")
    lines.append(f"| **Face2Face Detection Rate (τ=0.5)** | {get_val(m12, 'face2face_detection_rate', True)} | {get_val(m_fro, 'face2face_detection_rate', True)} | {get_val(m_jnt, 'face2face_detection_rate', True)} | **{get_val(m_b, 'face2face_detection_rate', True)}** |")
    lines.append(f"| **In-Domain Seen Fake ROC-AUC** | {get_val(m12, 'seen_auc')} | {get_val(m_fro, 'seen_auc')} | {get_val(m_jnt, 'seen_auc')} | {get_val(m_b, 'seen_auc')} |")
    lines.append(f"| **In-Domain Seen Fake F1** | {get_val(m12, 'seen_f1')} | {get_val(m_fro, 'seen_f1')} | {get_val(m_jnt, 'seen_f1')} | {get_val(m_b, 'seen_f1')} |")
    lines.append(f"| **Celeb-DF Cross-Dataset AUC** | {get_val(m12, 'cross_dataset_auc')} | {get_val(m_fro, 'cross_dataset_auc')} | {get_val(m_jnt, 'cross_dataset_auc')} | {get_val(m_b, 'cross_dataset_auc')} |")
    lines.append(f"| **Celeb-DF Cross-Dataset F1** | {get_val(m12, 'cross_dataset_f1')} | {get_val(m_fro, 'cross_dataset_f1')} | {get_val(m_jnt, 'cross_dataset_f1')} | {get_val(m_b, 'cross_dataset_f1')} |")
    lines.append(f"| **Degraded (JPEG Q=40) AUC** | {get_val(m12, 'degraded_auc')} | {get_val(m_fro, 'degraded_auc')} | {get_val(m_jnt, 'degraded_auc')} | {get_val(m_b, 'degraded_auc')} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Held-Out Face2Face Probability Distributions (Unseen Family Generalization)",
        "",
        "| Model | Mean Prob [95% CI] | Median Prob | Detection Rate (τ=0.5) | Interpretation |",
        "|---|---|---|---|---|",
    ])

    for m_name in ["Frozen M12 Baseline", "M13.2a-Frozen", "M13.2a-Joint", "M13.2b-Stage5"]:
        if m_name in all_results:
            s_f2f = all_results[m_name]["slices"]["Held-Out Face2Face"]
            p_mean = s_f2f["p_fused_mean"]
            p_ci = s_f2f["p_fused_ci"]
            p_med = s_f2f["p_fused_median"]
            p_det = s_f2f["det_rate_tau_0_5"]
            lines.append(
                f"| **{m_name}** | {p_mean:.4f} [{p_ci[0]:.4f}, {p_ci[1]:.4f}] | {p_med:.4f} | {p_det*100:.1f}% | {'Predicts Real (Failure)' if p_mean < 0.3 else 'Generalized Detection'} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Computational and Latency Profile",
        "",
        "| Model | Total Parameters | P50 Latency (ms) | P95 Latency (ms) | Peak VRAM (MB) |",
        "|---|---|---|---|---|",
    ])

    for m_name in ["Frozen M12 Baseline", "M13.2a-Frozen", "M13.2a-Joint", "M13.2b-Stage5"]:
        if m_name in all_results:
            r = all_results[m_name]
            lines.append(
                f"| **{m_name}** | {r['parameters']:,} | {r['latency_p50_ms']} ms | {r['latency_p95_ms']} ms | {r['peak_vram_mb']} MB |"
            )

    lines.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved Markdown report to {REPORT_MD}")


if __name__ == "__main__":
    main()
