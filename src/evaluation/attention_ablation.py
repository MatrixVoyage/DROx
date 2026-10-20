"""Milestone 9 Gate 5: Spatial Attention Attribution (5A) & Intervention Diagnostic (5B)."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.pipeline import VideoFacePipeline
from src.evaluation.attribution import (
    compute_sequence_attention_stats,
    extract_attention_weights,
)
from src.evaluation.metrics import compute_metrics
from src.utils.config import AppConfig

logger = logging.getLogger("m9_attention_ablation")


def check_uniform_attention_intervention_safety(model: torch.nn.Module, device: str) -> Tuple[bool, str]:
    """Inspect model graph to verify if uniform spatial pooling is safely feasible in-memory."""
    try:
        # Create dummy sequence tensor (1, 16, 3, 112, 112)
        dummy_x = torch.zeros(1, 16, 3, 112, 112, device=device)
        with torch.inference_mode():
            # Check if backbone supports return_spatial_map
            if not hasattr(model, "backbone"):
                return False, "Model does not expose 'backbone' submodule."

            _, spatial_maps = model.backbone(dummy_x, return_spatial_map=True)
            if spatial_maps.shape != torch.Size([1, 16, 2048, 4, 4]):
                return False, f"Unexpected spatial map shape: {spatial_maps.shape}"

            # Check if spatial pooling produces (1, 16, 2048)
            uniform_features = spatial_maps.mean(dim=(-2, -1))
            if uniform_features.shape != torch.Size([1, 16, 2048]):
                return False, f"Unexpected pooled feature shape: {uniform_features.shape}"

            # Check temporal and classifier
            if not hasattr(model, "temporal") or not hasattr(model, "classifier"):
                return False, "Model missing 'temporal' or 'classifier' submodule."

            temp_feat = model.temporal(uniform_features)
            logits = model.classifier(temp_feat)
            if logits.shape != torch.Size([1, 1]):
                return False, f"Unexpected output logits shape: {logits.shape}"

        return True, "Safe inference-only substitution point verified (spatial_maps -> mean((-2, -1)) -> temporal -> classifier)"
    except Exception as e:
        return False, f"In-memory substitution failed: {e}"


def run_gate5a_observational_attribution(
    model: torch.nn.Module,
    pipeline: VideoFacePipeline,
    device: str,
    eval_videos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Gate 5A: Observational analysis of attention entropy, peak weight, and top-4 mass."""
    real_entropies = []
    fake_entropies = []
    real_top4_masses = []
    fake_top4_masses = []

    correct_entropies = []
    error_entropies = []

    case_stats = []

    for v in eval_videos:
        v_id = v["video_id"]
        v_path = Path(v["video_path"])
        lbl = v["label"]

        prep = pipeline.process_video(v_path, video_id=v_id)
        if prep.status != "success" or prep.tensor is None:
            continue

        attn_weights = extract_attention_weights(model, prep.tensor)
        stats = compute_sequence_attention_stats(attn_weights)

        # Standard forward pass to determine correctness at threshold=0.5
        inp = prep.tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            prob = float(torch.sigmoid(model(inp)).squeeze().item())

        pred_lbl = 1 if prob >= 0.5 else 0
        is_correct = (pred_lbl == lbl)

        if lbl == 0:
            real_entropies.append(stats.mean_entropy)
            real_top4_masses.append(stats.mean_top4_mass)
        else:
            fake_entropies.append(stats.mean_entropy)
            fake_top4_masses.append(stats.mean_top4_mass)

        if is_correct:
            correct_entropies.append(stats.mean_entropy)
        else:
            error_entropies.append(stats.mean_entropy)

        case_stats.append({
            "video_id": v_id,
            "label": lbl,
            "probability": round(prob, 4),
            "is_correct": is_correct,
            "mean_entropy": round(stats.mean_entropy, 4),
            "mean_peak_weight": round(stats.mean_max_weight, 4),
            "mean_top4_mass": round(stats.mean_top4_mass, 4),
        })

    return {
        "summary": {
            "real_mean_entropy": round(float(np.mean(real_entropies)), 4) if real_entropies else 0.0,
            "fake_mean_entropy": round(float(np.mean(fake_entropies)), 4) if fake_entropies else 0.0,
            "real_mean_top4_mass": round(float(np.mean(real_top4_masses)), 4) if real_top4_masses else 0.0,
            "fake_mean_top4_mass": round(float(np.mean(fake_top4_masses)), 4) if fake_top4_masses else 0.0,
            "correct_mean_entropy": round(float(np.mean(correct_entropies)), 4) if correct_entropies else 0.0,
            "error_mean_entropy": round(float(np.mean(error_entropies)), 4) if error_entropies else 0.0,
        },
        "sample_count": len(case_stats),
        "cases": case_stats,
    }


def run_gate5b_interventional_ablation(
    model: torch.nn.Module,
    pipeline: VideoFacePipeline,
    device: str,
    eval_videos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Gate 5B: Interventional diagnostic comparing learned soft attention vs uniform spatial pooling."""
    is_safe, reason = check_uniform_attention_intervention_safety(model, device)
    if not is_safe:
        return {
            "status": "BLOCKED",
            "reason": f"Uniform attention intervention BLOCKED — {reason}",
            "metrics": {},
        }

    standard_scores = []
    uniform_scores = []
    labels = []
    comparison_cases = []

    for v in eval_videos:
        v_id = v["video_id"]
        v_path = Path(v["video_path"])
        lbl = v["label"]

        prep = pipeline.process_video(v_path, video_id=v_id)
        if prep.status != "success" or prep.tensor is None:
            continue

        inp = prep.tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            # 1. Standard model forward pass
            std_logits = model(inp)
            std_prob = float(torch.sigmoid(std_logits).squeeze().item())

            # 2. Uniform spatial pooling forward pass
            _, spatial_maps = model.backbone(inp, return_spatial_map=True)
            uniform_frame_features = spatial_maps.mean(dim=(-2, -1))
            temp_feat = model.temporal(uniform_frame_features)
            uni_logits = model.classifier(temp_feat)
            uni_prob = float(torch.sigmoid(uni_logits).squeeze().item())

        standard_scores.append(std_prob)
        uniform_scores.append(uni_prob)
        labels.append(lbl)

        comparison_cases.append({
            "video_id": v_id,
            "label": lbl,
            "standard_attention_prob": round(std_prob, 4),
            "uniform_attention_prob": round(uni_prob, 4),
            "prob_delta": round(uni_prob - std_prob, 4),
        })

    # Compute metrics for both
    def to_metrics(sc: List[float]) -> Dict[str, Any]:
        clipped = np.clip(np.array(sc), 1e-7, 1.0 - 1e-7)
        logits = np.log(clipped / (1.0 - clipped))
        y_log = torch.from_numpy(logits).view(-1, 1).float()
        y_tr = torch.from_numpy(np.array(labels)).view(-1, 1).float()
        return compute_metrics(y_true=y_tr, y_logits=y_log, threshold=0.5).to_dict()

    std_metrics = to_metrics(standard_scores)
    uni_metrics = to_metrics(uniform_scores)

    return {
        "status": "PASS",
        "reason": reason,
        "sample_count": len(comparison_cases),
        "standard_attention_metrics": std_metrics,
        "uniform_attention_metrics": uni_metrics,
        "delta_accuracy": round(uni_metrics["accuracy"] - std_metrics["accuracy"], 4),
        "delta_roc_auc": round(uni_metrics["roc_auc"] - std_metrics["roc_auc"], 4),
        "cases": comparison_cases,
    }


def plot_attention_comparison(
    gate5_data: Dict[str, Any],
    output_png: Path,
) -> None:
    """Generate comparative visualization for Gate 5 attention attribution & intervention."""
    fig, (ax_ent, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Entropy distributions
    g5a = gate5_data["gate_5a_attribution"]["summary"]
    labels = ["Real", "Fake", "Correct", "Error"]
    entropies = [
        g5a["real_mean_entropy"],
        g5a["fake_mean_entropy"],
        g5a["correct_mean_entropy"],
        g5a["error_mean_entropy"],
    ]
    ax_ent.bar(labels, entropies, color=["#10b981", "#ef4444", "#3b82f6", "#f59e0b"], width=0.5)
    ax_ent.set_ylabel("Mean Spatial Entropy (nats)")
    ax_ent.set_title("Attention Weight Entropy Across Subsets")
    ax_ent.grid(True, linestyle=":", alpha=0.6)
    for i, v in enumerate(entropies):
        ax_ent.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # 2. Intervention comparison
    g5b = gate5_data["gate_5b_intervention"]
    if g5b["status"] == "PASS":
        m_std = g5b["standard_attention_metrics"]
        m_uni = g5b["uniform_attention_metrics"]

        x = np.arange(2)
        width = 0.35
        ax_bar.bar(x - width / 2, [m_std["accuracy"], m_std["roc_auc"]], width, label="Learned Soft Attention", color="#3b82f6")
        ax_bar.bar(x + width / 2, [m_uni["accuracy"], m_uni["roc_auc"]], width, label="Uniform Spatial Pooling", color="#8b5cf6")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(["Accuracy", "ROC-AUC"])
        ax_bar.set_ylabel("Metric Value")
        ax_bar.set_title("Intervention Diagnostic: Attention vs Uniform Pooling")
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.legend(loc="lower right")
        ax_bar.grid(True, linestyle=":", alpha=0.6)
    else:
        ax_bar.text(0.5, 0.5, f"Intervention BLOCKED\n{g5b.get('reason', '')}", ha="center", va="center")
        ax_bar.set_title("Intervention Diagnostic")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()
