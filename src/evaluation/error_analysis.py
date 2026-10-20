"""Systematic error analysis and face detection correlation for deepfake video detection.

Categorizes prediction errors into False Positives (pristine/real videos classified as deepfake)
and False Negatives (manipulated videos classified as pristine), extracts diagnostic metadata
(MTCNN confidence, bounding-box scales, aspect ratios, fallback detector triggers),
and computes empirical associations without making unfounded causal claims.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ErrorSampleMetadata:
    """Detailed diagnostic metadata for a single false positive or false negative sample."""

    video_id: str
    video_path: str
    dataset: str
    manipulation_type: str
    ground_truth: int
    predicted_probability: float
    threshold: float
    error_type: str  # "false_positive" or "false_negative"
    margin: float  # Absolute distance from decision threshold

    # Face detection diagnostics
    mean_mtcnn_confidence: Optional[float] = None
    min_mtcnn_confidence: Optional[float] = None
    mean_bbox_area: Optional[float] = None
    mean_bbox_aspect_ratio: Optional[float] = None
    num_fallback_frames: int = 0
    fallback_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorrelationFinding:
    """Association finding between detection feature and error occurrence.

    Strictly uses non-causal terminology ('associated with', 'correlated with').
    """

    feature_name: str
    correct_samples_mean: float
    error_samples_mean: float
    point_biserial_correlation: float
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def categorize_errors(
    predictions: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Tuple[List[ErrorSampleMetadata], List[ErrorSampleMetadata], List[Dict[str, Any]]]:
    """Identify false positives, false negatives, and correct predictions.

    Args:
        predictions: List of sample prediction dicts. Each must have:
            - 'video_id' or 'path'
            - 'label' (int: 0=real, 1=fake)
            - 'probability' (float in [0, 1])
            Optional fields: 'dataset', 'manipulation', 'mtcnn_confidences',
            'bboxes', 'fallback_frames'.
        threshold: Decision threshold for deepfake classification.

    Returns:
        Tuple of (false_positives, false_negatives, correct_samples).
    """
    fps: List[ErrorSampleMetadata] = []
    fns: List[ErrorSampleMetadata] = []
    correct: List[Dict[str, Any]] = []

    for pred in predictions:
        video_id = str(pred.get("video_id", pred.get("path", "unknown")))
        video_path = str(pred.get("path", video_id))
        dataset = str(pred.get("dataset", "unknown"))
        manip = str(pred.get("manipulation", pred.get("manipulation_type", "unknown")))
        label = int(pred["label"])
        prob = float(pred["probability"])

        # Face detection metrics
        mtcnn_confs = pred.get("mtcnn_confidences") or pred.get("detection_confidences") or []
        bboxes = pred.get("bboxes") or []
        fallback_frames = int(
            pred.get("num_fallback_frames", pred.get("fallback_count", pred.get("fallback_frames", 0)))
        )
        total_frames = int(pred.get("num_frames", 16))
        fallback_ratio = float(fallback_frames / total_frames) if total_frames > 0 else 0.0

        mean_conf = float(np.mean(mtcnn_confs)) if len(mtcnn_confs) > 0 else None
        min_conf = float(np.min(mtcnn_confs)) if len(mtcnn_confs) > 0 else None

        # Bounding box metrics: bboxes format expected [x1, y1, x2, y2]
        areas: List[float] = []
        aspect_ratios: List[float] = []
        for box in bboxes:
            if len(box) >= 4:
                w = max(0.0, float(box[2] - box[0]))
                h = max(0.0, float(box[3] - box[1]))
                areas.append(w * h)
                if h > 0:
                    aspect_ratios.append(w / h)

        mean_area = float(np.mean(areas)) if areas else None
        mean_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else None

        predicted_class = 1 if prob >= threshold else 0
        margin = float(abs(prob - threshold))

        if label == 0 and predicted_class == 1:
            fps.append(
                ErrorSampleMetadata(
                    video_id=video_id,
                    video_path=video_path,
                    dataset=dataset,
                    manipulation_type=manip,
                    ground_truth=label,
                    predicted_probability=round(prob, 4),
                    threshold=threshold,
                    error_type="false_positive",
                    margin=round(margin, 4),
                    mean_mtcnn_confidence=round(mean_conf, 4) if mean_conf is not None else None,
                    min_mtcnn_confidence=round(min_conf, 4) if min_conf is not None else None,
                    mean_bbox_area=round(mean_area, 2) if mean_area is not None else None,
                    mean_bbox_aspect_ratio=round(mean_aspect, 4) if mean_aspect is not None else None,
                    num_fallback_frames=fallback_frames,
                    fallback_ratio=round(fallback_ratio, 4),
                )
            )
        elif label == 1 and predicted_class == 0:
            fns.append(
                ErrorSampleMetadata(
                    video_id=video_id,
                    video_path=video_path,
                    dataset=dataset,
                    manipulation_type=manip,
                    ground_truth=label,
                    predicted_probability=round(prob, 4),
                    threshold=threshold,
                    error_type="false_negative",
                    margin=round(margin, 4),
                    mean_mtcnn_confidence=round(mean_conf, 4) if mean_conf is not None else None,
                    min_mtcnn_confidence=round(min_conf, 4) if min_conf is not None else None,
                    mean_bbox_area=round(mean_area, 2) if mean_area is not None else None,
                    mean_bbox_aspect_ratio=round(mean_aspect, 4) if mean_aspect is not None else None,
                    num_fallback_frames=fallback_frames,
                    fallback_ratio=round(fallback_ratio, 4),
                )
            )
        else:
            correct.append(pred)

    # Sort false positives and false negatives descending by margin (most egregious first)
    fps.sort(key=lambda x: x.margin, reverse=True)
    fns.sort(key=lambda x: x.margin, reverse=True)

    return fps, fns, correct


def compute_biserial_correlation(
    feature_values: List[float],
    is_error: List[int],
) -> float:
    """Compute point-biserial correlation between a continuous feature and binary error status."""
    if len(feature_values) != len(is_error) or len(feature_values) < 2:
        return 0.0

    x = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(is_error, dtype=np.float64)

    # Check for zero variance
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0

    r = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if math.isnan(r) else round(r, 4)


def analyze_detection_correlations(
    predictions: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> List[CorrelationFinding]:
    """Analyze empirical associations between face detection attributes and classification errors.

    Strictly preserves non-causal scientific language.
    """
    fps, fns, correct = categorize_errors(predictions, threshold=threshold)
    error_ids = {s.video_id for s in fps}.union({s.video_id for s in fns})

    conf_values: List[float] = []
    area_values: List[float] = []
    aspect_values: List[float] = []
    fallback_values: List[float] = []
    is_error: List[int] = []

    for p in predictions:
        vid = str(p.get("video_id", p.get("path", "")))
        err = 1 if vid in error_ids else 0

        # Confidence
        confs = p.get("mtcnn_confidences") or []
        if confs:
            conf_values.append(float(np.mean(confs)))
        else:
            conf_values.append(float(p.get("mean_mtcnn_confidence", 0.95)))

        # Area
        areas = [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in p.get("bboxes", []) if len(b) >= 4]
        if areas:
            area_values.append(float(np.mean(areas)))
        else:
            area_values.append(float(p.get("mean_bbox_area", 10000.0)))

        # Aspect
        aspects = [
            (b[2] - b[0]) / (b[3] - b[1])
            for b in p.get("bboxes", [])
            if len(b) >= 4 and (b[3] - b[1]) > 0
        ]
        if aspects:
            aspect_values.append(float(np.mean(aspects)))
        else:
            aspect_values.append(float(p.get("mean_bbox_aspect_ratio", 0.75)))

        # Fallback ratio
        tot_frames = int(p.get("num_frames", 16))
        fb_cnt = int(p.get("num_fallback_frames", p.get("fallback_count", 0)))
        fb_rat = fb_cnt / tot_frames if tot_frames > 0 else 0.0
        fallback_values.append(fb_rat)

        is_error.append(err)

    findings: List[CorrelationFinding] = []

    features = [
        ("mtcnn_confidence", conf_values, "higher detection confidence is associated with"),
        ("face_bounding_box_area", area_values, "face bounding box area is correlated with"),
        ("face_aspect_ratio", aspect_values, "face bounding box aspect ratio is correlated with"),
        ("fallback_detector_ratio", fallback_values, "fallback face detection rate is associated with"),
    ]

    for feat_name, vals, phrase in features:
        if len(vals) == len(is_error) and len(vals) > 0:
            arr_vals = np.array(vals)
            arr_err = np.array(is_error)

            correct_m = float(np.mean(arr_vals[arr_err == 0])) if np.any(arr_err == 0) else 0.0
            error_m = float(np.mean(arr_vals[arr_err == 1])) if np.any(arr_err == 1) else 0.0
            r = compute_biserial_correlation(vals, is_error)

            corr_dir = "positively" if r > 0 else "negatively"
            interp = (
                f"Empirical data shows {feat_name} is {corr_dir} correlated (r={r:.3f}) with prediction error; "
                f"mean for correct predictions is {correct_m:.3f} vs {error_m:.3f} for error cases. "
                f"This observed relationship does not establish direct causality."
            )

            findings.append(
                CorrelationFinding(
                    feature_name=feat_name,
                    correct_samples_mean=round(correct_m, 4),
                    error_samples_mean=round(error_m, 4),
                    point_biserial_correlation=r,
                    interpretation=interp,
                )
            )

    return findings


def generate_error_analysis_summary(
    predictions: List[Dict[str, Any]],
    threshold: float = 0.5,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Generate complete error taxonomy and statistical correlation analysis."""
    fps, fns, correct = categorize_errors(predictions, threshold=threshold)
    correlations = analyze_detection_correlations(predictions, threshold=threshold)

    total_samples = len(predictions)
    total_fps = len(fps)
    total_fns = len(fns)
    total_errors = total_fps + total_fns

    total_real = sum(1 for p in predictions if int(p["label"]) == 0)
    total_fake = sum(1 for p in predictions if int(p["label"]) == 1)

    fpr = (total_fps / total_real) if total_real > 0 else 0.0
    fnr = (total_fns / total_fake) if total_fake > 0 else 0.0

    # Family breakdowns for errors
    fam_breakdown: Dict[str, Dict[str, Any]] = {}
    for p in predictions:
        fam = str(p.get("manipulation", p.get("manipulation_type", "unknown")))
        if fam not in fam_breakdown:
            fam_breakdown[fam] = {"total": 0, "errors": 0, "error_rate": 0.0}
        fam_breakdown[fam]["total"] += 1

    for e in fps + fns:
        fam = e.manipulation_type
        if fam in fam_breakdown:
            fam_breakdown[fam]["errors"] += 1

    for fam, d in fam_breakdown.items():
        d["error_rate"] = round(d["errors"] / d["total"], 4) if d["total"] > 0 else 0.0

    return {
        "threshold": threshold,
        "total_samples": total_samples,
        "total_real": total_real,
        "total_fake": total_fake,
        "total_errors": total_errors,
        "overall_error_rate": round(total_errors / total_samples, 4) if total_samples > 0 else 0.0,
        "false_positive_count": total_fps,
        "false_positive_rate": round(fpr, 4),
        "false_negative_count": total_fns,
        "false_negative_rate": round(fnr, 4),
        "manipulation_family_errors": fam_breakdown,
        "correlations": [c.to_dict() for c in correlations],
        "top_false_positives": [fp.to_dict() for fp in fps[:top_k]],
        "top_false_negatives": [fn.to_dict() for fn in fns[:top_k]],
    }
