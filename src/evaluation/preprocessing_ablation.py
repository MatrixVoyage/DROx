"""Milestone 9 Gate 3: One-Factor-At-A-Time (OFAT) Preprocessing Ablation."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.pipeline import VideoFacePipeline
from src.evaluation.metrics import compute_metrics
from src.inference.predictor import VideoPredictor
from src.utils.config import AppConfig, CropConfig, DetectionConfig, SamplingConfig

logger = logging.getLogger("m9_preprocessing_ablation")


def run_single_pipeline_inference(
    pipeline: VideoFacePipeline,
    model: torch.nn.Module,
    device: str,
    video_path: Path,
    video_id: str,
) -> Tuple[Optional[float], Optional[List[int]], Optional[str]]:
    """Execute pipeline processing and model inference for a single video."""
    try:
        prep_res = pipeline.process_video(video_path, video_id=video_id)
        if prep_res.status != "success" or prep_res.tensor is None:
            return None, None, f"Preprocessing failed: status={prep_res.status}"

        # Feed (16, 3, 112, 112) tensor to frozen model
        inp = prep_res.tensor.unsqueeze(0).to(device)  # (1, 16, 3, 112, 112)
        with torch.inference_mode():
            logits = model(inp)
            prob = float(torch.sigmoid(logits).squeeze().item())

        frame_indices = prep_res.frame_indices if prep_res.frame_indices is not None else []
        return round(prob, 4), frame_indices, None
    except Exception as e:
        return None, None, str(e)


def evaluate_ofat_experiment(
    exp_name: str,
    config: AppConfig,
    is_out_of_contract: bool,
    videos: List[Dict[str, Any]],
    baseline_predictions: Dict[str, float],
    model: torch.nn.Module,
    device: str,
) -> Dict[str, Any]:
    """Evaluate a single OFAT variation on the evaluation subset and compare against baseline."""
    pipeline = VideoFacePipeline(config=config)
    sample_records: List[Dict[str, Any]] = []
    scores: List[float] = []
    labels: List[int] = []

    for v in videos:
        v_id = v["video_id"]
        v_path = Path(v["video_path"])
        lbl = v["label"]
        b_score = baseline_predictions.get(v_id)

        prob, indices, err = run_single_pipeline_inference(pipeline, model, device, v_path, v_id)
        if prob is not None:
            score_delta = round(prob - b_score, 4) if b_score is not None else 0.0
            sample_records.append({
                "video_id": v_id,
                "label": lbl,
                "frame_indices": indices,
                "baseline_score": b_score,
                "alternative_score": prob,
                "score_delta": score_delta,
            })
            scores.append(prob)
            labels.append(lbl)
        else:
            logger.warning(f"Failed to process {v_id} under {exp_name}: {err}")

    # Compute metrics
    if len(scores) > 0 and len(set(labels)) > 1:
        y_logits = torch.from_numpy(np.log(np.clip(np.array(scores), 1e-7, 1.0 - 1e-7) / (1.0 - np.clip(np.array(scores), 1e-7, 1.0 - 1e-7)))).view(-1, 1).float()
        y_true = torch.from_numpy(np.array(labels)).view(-1, 1).float()
        m = compute_metrics(y_true=y_true, y_logits=y_logits, threshold=0.5).to_dict()
    else:
        m = {"accuracy": 0.0, "roc_auc": 0.5, "f1": 0.0}

    return {
        "experiment": exp_name,
        "is_out_of_contract": is_out_of_contract,
        "contract_label": "out_of_contract diagnostic" if is_out_of_contract else "production contract",
        "sample_count": len(sample_records),
        "metrics": m,
        "samples": sample_records,
    }


def run_preprocessing_ablation_suite(
    eval_videos: List[Dict[str, Any]],
    model: torch.nn.Module,
    device: str,
) -> Dict[str, Any]:
    """Execute complete One-Factor-At-A-Time preprocessing ablation on 30 test videos."""
    logger.info(f"Running OFAT Preprocessing Ablation on {len(eval_videos)} videos...")

    # 1. Baseline Configuration
    base_cfg = AppConfig()
    base_cfg.data.crop.margin_ratio = 0.20
    base_cfg.data.detection.selection_strategy = "largest"
    base_cfg.data.normalization = "imagenet"
    base_cfg.data.sampling.method = "uniform"

    base_pipeline = VideoFacePipeline(config=base_cfg)
    baseline_preds: Dict[str, float] = {}
    baseline_samples: List[Dict[str, Any]] = []
    base_scores: List[float] = []
    base_labels: List[int] = []

    for v in eval_videos:
        v_id = v["video_id"]
        v_path = Path(v["video_path"])
        lbl = v["label"]
        prob, indices, _ = run_single_pipeline_inference(base_pipeline, model, device, v_path, v_id)
        if prob is not None:
            baseline_preds[v_id] = prob
            baseline_samples.append({
                "video_id": v_id,
                "label": lbl,
                "frame_indices": indices,
                "baseline_score": prob,
                "alternative_score": prob,
                "score_delta": 0.0,
            })
            base_scores.append(prob)
            base_labels.append(lbl)

    if len(base_scores) > 0 and len(set(base_labels)) > 1:
        y_base_logits = torch.from_numpy(np.log(np.clip(np.array(base_scores), 1e-7, 1.0 - 1e-7) / (1.0 - np.clip(np.array(base_scores), 1e-7, 1.0 - 1e-7)))).view(-1, 1).float()
        y_base_true = torch.from_numpy(np.array(base_labels)).view(-1, 1).float()
        base_metrics = compute_metrics(y_true=y_base_true, y_logits=y_base_logits, threshold=0.5).to_dict()
    else:
        base_metrics = {"accuracy": 0.0, "roc_auc": 0.5, "f1": 0.0}

    results: Dict[str, Any] = {
        "baseline": {
            "experiment": "baseline_margin0.2_largest_imagenet_uniform",
            "is_out_of_contract": False,
            "contract_label": "production contract",
            "sample_count": len(baseline_samples),
            "metrics": base_metrics,
            "samples": baseline_samples,
        }
    }

    # 2. OFAT Variations
    variations = [
        # Margin variations
        ("margin_0.0", 0.0, "largest", "imagenet", "uniform", False),
        ("margin_0.1", 0.1, "largest", "imagenet", "uniform", False),
        ("margin_0.3", 0.3, "largest", "imagenet", "uniform", False),
        ("margin_0.4", 0.4, "largest", "imagenet", "uniform", False),
        # Face selection variation
        ("selection_highest_confidence", 0.2, "highest_confidence", "imagenet", "uniform", False),
        # Normalization variation (out-of-contract diagnostic)
        ("normalization_none", 0.2, "largest", "none", "uniform", True),
        # Sampling variation
        ("sampling_strided", 0.2, "largest", "imagenet", "strided", False),
    ]

    for name, margin, sel, norm, samp, out_of_contract in variations:
        logger.info(f"Evaluating OFAT variation: {name} (out_of_contract={out_of_contract})...")
        var_cfg = AppConfig()
        var_cfg.data.crop.margin_ratio = margin
        var_cfg.data.detection.selection_strategy = sel
        var_cfg.data.normalization = norm
        var_cfg.data.sampling.method = samp

        res = evaluate_ofat_experiment(
            exp_name=name,
            config=var_cfg,
            is_out_of_contract=out_of_contract,
            videos=eval_videos,
            baseline_predictions=baseline_preds,
            model=model,
            device=device,
        )

        # Attach deltas vs baseline
        res["delta_accuracy_vs_baseline"] = round(res["metrics"]["accuracy"] - base_metrics["accuracy"], 4)
        res["delta_roc_auc_vs_baseline"] = round(res["metrics"]["roc_auc"] - base_metrics["roc_auc"], 4)
        results[name] = res

    return results
