"""Milestone 9 Gate 6: Structured Cross-Dataset Error Diagnosis & Error Gallery."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("m9_error_diagnosis")


def categorize_sample_at_threshold(
    probability: float,
    label: int,
    threshold: float,
) -> str:
    """Categorize prediction as TP, FP, TN, or FN at a specified decision threshold."""
    pred_fake = probability >= threshold
    if label == 1 and pred_fake:
        return "TP"
    elif label == 0 and pred_fake:
        return "FP"
    elif label == 0 and not pred_fake:
        return "TN"
    else:
        return "FN"


def build_error_matrix_for_dataset(
    records: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """Compute error matrix (TP, FP, TN, FN, FPR, FNR) at a specific decision threshold."""
    tp, fp, tn, fn = 0, 0, 0, 0
    for r in records:
        cat = categorize_sample_at_threshold(r["probability"], r["label"], threshold)
        if cat == "TP":
            tp += 1
        elif cat == "FP":
            fp += 1
        elif cat == "TN":
            tn += 1
        elif cat == "FN":
            fn += 1

    total_real = fp + tn
    total_fake = tp + fn
    fpr = fp / total_real if total_real > 0 else 0.0
    fnr = fn / total_fake if total_fake > 0 else 0.0

    return {
        "threshold": threshold,
        "total_samples": len(records),
        "total_real": total_real,
        "total_fake": total_fake,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }


def analyze_multithreshold_errors(
    ffpp_records: List[Dict[str, Any]],
    celebdf_records: List[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Compute error matrices across baseline and validation-derived optimal thresholds."""
    results = {}
    for t_name, t_val in thresholds.items():
        results[t_name] = {
            "threshold_value": t_val,
            "ffpp_c23_test": build_error_matrix_for_dataset(ffpp_records, t_val),
            "celebdf_v2_test": build_error_matrix_for_dataset(celebdf_records, t_val),
        }
    return results


def analyze_family_error_breakdown(
    ffpp_records: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Analyze error rates across specific manipulation families at threshold=0.50."""
    families = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    breakdown = {}

    for fam in families:
        fam_recs = [r for r in ffpp_records if r.get("manipulation_type") == fam]
        matrix = build_error_matrix_for_dataset(fam_recs, threshold)
        breakdown[fam] = matrix

    return breakdown


def generate_error_gallery(
    ffpp_records: List[Dict[str, Any]],
    celebdf_records: List[Dict[str, Any]],
    gallery_root: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate representative face image galleries for error cases."""
    gallery_root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "highest_fp": gallery_root / "highest_fp",
        "highest_fn": gallery_root / "highest_fn",
        "borderline": gallery_root / "borderline",
        "family_examples": gallery_root / "family_examples",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    all_test = ffpp_records + celebdf_records

    # 1. Highest False Positives (real videos with highest predicted fake probability)
    reals = [r for r in all_test if r["label"] == 0]
    sorted_fps = sorted(reals, key=lambda x: x["probability"], reverse=True)[:5]

    # 2. Highest False Negatives (fake videos with lowest predicted fake probability)
    fakes = [r for r in all_test if r["label"] == 1]
    sorted_fns = sorted(fakes, key=lambda x: x["probability"])[:5]

    # 3. Borderline cases (closest to 0.50)
    borderline = sorted(all_test, key=lambda x: abs(x["probability"] - 0.50))[:5]

    # 4. Family examples (one representative per family)
    families = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    fam_examples = []
    for fam in families:
        fam_recs = [r for r in ffpp_records if r.get("manipulation_type") == fam]
        if fam_recs:
            fam_examples.append(fam_recs[0])

    gallery_meta: Dict[str, List[Dict[str, Any]]] = {
        "highest_fp": [],
        "highest_fn": [],
        "borderline": [],
        "family_examples": [],
    }

    def save_sample_preview(sample: Dict[str, Any], dest_folder: Path, prefix: str) -> Optional[str]:
        v_path = Path(sample["video_path"])
        if not v_path.exists():
            return None
        cap = cv2.VideoCapture(str(v_path))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None

        # Resize to thumbnail 224x224
        thumb = cv2.resize(frame, (224, 224))
        # Add metadata text
        cv2.putText(thumb, f"P={sample['probability']:.3f} L={sample['label']}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if sample['label'] == 0 else (0, 0, 255), 1)
        out_file = dest_folder / f"{prefix}_{sample['video_id']}.jpg"
        cv2.imwrite(str(out_file), thumb)
        return str(out_file)

    for i, s in enumerate(sorted_fps):
        p = save_sample_preview(s, dirs["highest_fp"], f"fp_{i}")
        gallery_meta["highest_fp"].append({"video_id": s["video_id"], "prob": s["probability"], "preview": p})

    for i, s in enumerate(sorted_fns):
        p = save_sample_preview(s, dirs["highest_fn"], f"fn_{i}")
        gallery_meta["highest_fn"].append({"video_id": s["video_id"], "prob": s["probability"], "preview": p})

    for i, s in enumerate(borderline):
        p = save_sample_preview(s, dirs["borderline"], f"border_{i}")
        gallery_meta["borderline"].append({"video_id": s["video_id"], "prob": s["probability"], "preview": p})

    for s in fam_examples:
        fam = s.get("manipulation_type", "unknown")
        p = save_sample_preview(s, dirs["family_examples"], f"fam_{fam}")
        gallery_meta["family_examples"].append({"video_id": s["video_id"], "family": fam, "prob": s["probability"], "preview": p})

    return gallery_meta
