"""Milestone 9 Gate 4: Temporal Sampling Ablation & Architectural Invariant Inspection."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.pipeline import VideoFacePipeline
from src.evaluation.metrics import compute_metrics
from src.utils.config import AppConfig

logger = logging.getLogger("m9_temporal_ablation")


def run_temporal_variant_inference(
    pipeline: VideoFacePipeline,
    model: torch.nn.Module,
    device: str,
    video_path: Path,
    video_id: str,
) -> Tuple[Optional[float], Optional[List[int]], Optional[str]]:
    """Execute video face pipeline and inference under a temporal sampling strategy."""
    try:
        prep = pipeline.process_video(video_path, video_id=video_id)
        if prep.status != "success" or prep.tensor is None:
            return None, None, f"Preprocessing failed with status={prep.status}"

        tensor = prep.tensor
        # Assert strict (16, 3, 112, 112) shape invariant
        if tensor.shape != torch.Size([16, 3, 112, 112]):
            return None, None, f"Invalid tensor shape: {tensor.shape}"

        inp = tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(inp)
            prob = float(torch.sigmoid(logits).squeeze().item())

        indices = prep.frame_indices if prep.frame_indices is not None else []
        return round(prob, 4), indices, None
    except Exception as e:
        return None, None, str(e)


def run_temporal_ablation_suite(
    eval_videos: List[Dict[str, Any]],
    model: torch.nn.Module,
    device: str,
) -> Dict[str, Any]:
    """Evaluate temporal sampling variants with exact frame index logging."""
    logger.info(f"Running Temporal Ablation on {len(eval_videos)} videos...")

    variants = [
        ("T16_uniform", "uniform"),
        ("T16_deterministic_stride", "strided"),
        ("T16_random_seeded", "random"),
    ]

    variant_results: Dict[str, Any] = {}
    baseline_scores: Dict[str, float] = {}

    for var_name, method in variants:
        logger.info(f"Evaluating temporal strategy: {var_name} (method={method})...")
        cfg = AppConfig()
        cfg.data.sampling.num_frames = 16
        cfg.data.sampling.method = method
        pipeline = VideoFacePipeline(config=cfg)

        sample_records: List[Dict[str, Any]] = []
        scores: List[float] = []
        labels: List[int] = []

        for v in eval_videos:
            v_id = v["video_id"]
            v_path = Path(v["video_path"])
            lbl = v["label"]

            prob, indices, err = run_temporal_variant_inference(pipeline, model, device, v_path, v_id)
            if prob is not None:
                if var_name == "T16_uniform":
                    baseline_scores[v_id] = prob
                b_score = baseline_scores.get(v_id, prob)
                score_delta = round(prob - b_score, 4)

                sample_records.append({
                    "video_id": v_id,
                    "label": lbl,
                    "selected_frame_indices": indices,
                    "probability": prob,
                    "baseline_score": b_score,
                    "score_delta": score_delta,
                })
                scores.append(prob)
                labels.append(lbl)
            else:
                logger.warning(f"Temporal eval failed for {v_id} under {var_name}: {err}")

        # Compute metrics
        if len(scores) > 0 and len(set(labels)) > 1:
            y_logits = torch.from_numpy(np.log(np.clip(np.array(scores), 1e-7, 1.0 - 1e-7) / (1.0 - np.clip(np.array(scores), 1e-7, 1.0 - 1e-7)))).view(-1, 1).float()
            y_true = torch.from_numpy(np.array(labels)).view(-1, 1).float()
            m = compute_metrics(y_true=y_true, y_logits=y_logits, threshold=0.5).to_dict()
        else:
            m = {"accuracy": 0.0, "roc_auc": 0.5, "f1": 0.0}

        variant_results[var_name] = {
            "strategy": var_name,
            "sampling_method": method,
            "num_frames": 16,
            "sample_count": len(sample_records),
            "metrics": m,
            "samples": sample_records,
        }

    # Document architectural incompatibility of T != 16
    incompatible_experiments = {
        "T8_uniform": {
            "status": "NOT EVALUATED",
            "reason": "T=8 is architecturally incompatible with frozen LSTM checkpoint contracts (trained on fixed T=16 sequences).",
        },
        "T32_uniform": {
            "status": "NOT EVALUATED",
            "reason": "T=32 is architecturally incompatible with frozen LSTM checkpoint contracts (trained on fixed T=16 sequences).",
        },
    }

    return {
        "evaluated_variants": variant_results,
        "architectural_incompatibilities": incompatible_experiments,
    }


def plot_temporal_comparison(
    temporal_results: Dict[str, Any],
    output_png: Path,
) -> None:
    """Generate comparative visualization of temporal sampling variants."""
    eval_vars = temporal_results["evaluated_variants"]
    strategies = list(eval_vars.keys())
    accuracies = [eval_vars[s]["metrics"]["accuracy"] for s in strategies]
    aucs = [eval_vars[s]["metrics"]["roc_auc"] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#3b82f6")
    rects2 = ax.bar(x + width / 2, aucs, width, label="ROC-AUC", color="#10b981")

    ax.set_ylabel("Metric Value")
    ax.set_title("Temporal Sampling Strategy Comparison (Fixed T=16)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Uniform (Baseline)", "Deterministic Stride", "Random Seeded (s=42)"])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()
