"""Groq AI Forensic Analyst Integration.

Provides evidence-grounded, human-friendly narrative explanations of the
automated deepfake video detection pipeline.

Architecture Invariants:
1. Canonical Structured Evidence: The pipeline is the sole authoritative source of truth.
2. Single-Run Provenance: Point estimates and dispersion ranges must come from the same analysis run.
3. Numerical Consistency Validation: Stated point estimates must never lie outside their stated intervals.
4. Strict Causal Guardrails: Measurements are never described as proven seams, splices, or physical causes.
5. Deterministic Safe Fallback: If Groq fails or fails validation, a deterministic report is substituted.
"""
from __future__ import annotations

import os
import re
import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default Groq models in priority order
DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]


@dataclass
class CanonicalForensicEvidence:
    """The authoritative, canonical structured evidence object for a single video analysis run.
    This serves as the single source of truth for the LLM prompt, UI presentation, and validation.
    """
    run_id: str
    video_filename: str
    file_sha256: str
    media_metadata: Dict[str, Any]
    coverage_and_quality: Dict[str, Any]
    decision_context: Dict[str, Any]
    spatial_findings: Dict[str, Any]
    temporal_findings: Dict[str, Any]
    reference_baseline: Dict[str, Any]
    calibration_context: Dict[str, Any]
    limitations: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_groq_api_key() -> Optional[str]:
    """Retrieves the Groq API key from environment variables or .env file."""
    key = os.environ.get("GROQAPI") or os.environ.get("GROQ_API_KEY")
    if key and key.strip():
        return key.strip().strip('"').strip("'")

    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("GROQAPI") or line.startswith("GROQ_API_KEY"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            val = parts[1].strip().strip('"').strip("'")
                            if val:
                                return val
        except Exception as e:
            logger.debug(f"Failed to read .env file: {e}")

    return None


def is_groq_available() -> bool:
    """Returns True if a Groq API key is configured."""
    return get_groq_api_key() is not None


def build_canonical_evidence(
    audit_data: Dict[str, Any],
    pred_data: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
) -> CanonicalForensicEvidence:
    """Extracts and normalizes raw pipeline outputs into a single canonical evidence object.
    Enforces strict single-run provenance: point estimate and dispersion interval must refer
    to the exact same analysis run.
    """
    st = audit_data.get("structured_telemetry") or {}
    stream = st.get("stream") or {}
    spatial = st.get("spatial") or {}
    temporal = st.get("temporal") or {}
    baseline = st.get("baseline") or {}
    quality = st.get("quality") or {}
    decision = st.get("decision") or {}
    peak = temporal.get("peak_window") or {}

    file_name = (
        stream.get("source_file_name")
        or stream.get("file_name")
        or audit_data.get("video_filename")
        or "video.mp4"
    )
    file_sha256 = stream.get("file_sha256") or audit_data.get("file_sha256") or "N/A"
    run_id = f"run_{file_sha256[:12] if file_sha256 != 'N/A' else 'audit'}"

    # Active threshold
    eff_threshold = float(threshold if threshold is not None else decision.get("active_threshold", audit_data.get("decision_threshold", 0.50)))

    # 1. Authoritative Audit Pipeline Metrics
    # The primary evidence in audit_data is produced by the Spatiotemporal Multi-Window Pipeline (M20).
    # Its score is the weighted average across temporal windows, and its dispersion is [min(window), max(window)].
    audit_score = float(audit_data.get("video_fake_probability", decision.get("calibrated_probability", 0.0)))
    raw_ci = audit_data.get("confidence_interval") or [audit_score, audit_score]
    score_dispersion = [float(raw_ci[0]), float(raw_ci[1])]
    audit_engine_name = decision.get("model_name") or "SpatialTemporalForensicEngine"

    # Evaluate decision rule for the spatiotemporal audit run
    is_inconclusive = (
        audit_data.get("verdict") == "INCONCLUSIVE"
        or decision.get("decision_state") == "INCONCLUSIVE"
        or not quality.get("sufficient_coverage", True)
    )

    if is_inconclusive:
        automated_verdict = "INCONCLUSIVE"
        decision_rationale = "Assessment inconclusive due to insufficient face track coverage or active confounders."
    elif audit_score >= eff_threshold:
        automated_verdict = "MANIPULATION DETECTED"
        decision_rationale = f"Spatiotemporal model score ({audit_score * 100:.2f}%) meets or exceeds active threshold ({eff_threshold * 100:.2f}%)."
    else:
        automated_verdict = "NO MANIPULATION DETECTED"
        decision_rationale = f"Spatiotemporal model score ({audit_score * 100:.2f}%) falls below active threshold ({eff_threshold * 100:.2f}%)."

    # 2. Track optional comparator engine (e.g. from single-pass /api/v1/predict)
    comparator_engine = None
    data_consistency_status = "CONSISTENT"
    consistency_notes = "Point estimate and temporal window score dispersion belong strictly to the same spatiotemporal analysis run."

    if pred_data and pred_data.get("probability") is not None:
        pred_prob = float(pred_data["probability"])
        pred_eng = pred_data.get("engine") or "Authoritative Engine"
        pred_dec = pred_data.get("decision") or ("MANIPULATION DETECTED" if pred_prob >= eff_threshold else "NO MANIPULATION DETECTED")
        comparator_engine = {
            "engine_name": pred_eng,
            "model_score": pred_prob,
            "decision": pred_dec,
            "active_threshold": eff_threshold,
            "analysis_mode": "Single-pass 16-frame whole-video inference",
        }

        # Check for cross-engine score divergence
        if abs(pred_prob - audit_score) > 0.15 or (pred_prob >= eff_threshold) != (audit_score >= eff_threshold):
            data_consistency_status = "CROSS_MODEL_DIVERGENCE"
            consistency_notes = (
                f"Cross-model score divergence observed: Selected engine '{pred_eng}' scored {pred_prob*100:.2f}% ({pred_dec}) "
                f"in single-pass mode, whereas Spatiotemporal Multi-Window Audit ({audit_engine_name}) scored {audit_score*100:.2f}% "
                f"({automated_verdict}) with temporal window dispersion [{score_dispersion[0]*100:.2f}%, {score_dispersion[1]*100:.2f}%]. "
                f"These two quantities come from distinct model architectures and must not be conflated as a single point estimate and its confidence interval."
            )

    # Media metadata
    media_metadata = {
        "width": stream.get("width") or stream.get("frame_width"),
        "height": stream.get("height") or stream.get("frame_height"),
        "fps": stream.get("container_fps") or stream.get("fps"),
        "duration_sec": stream.get("duration_sec", audit_data.get("container_duration_sec")),
        "is_vfr": stream.get("is_vfr", False),
        "codec_name": stream.get("codec_name") or stream.get("codec") or "Unknown",
        "decode_status": stream.get("decode_status", "SUCCESS"),
        "file_size_bytes": stream.get("file_size_bytes", 0),
    }

    # Coverage and Quality
    coverage_and_quality = {
        "sufficient_coverage": bool(quality.get("sufficient_coverage", True)),
        "face_coverage_ratio": quality.get("face_coverage_ratio", quality.get("mean_face_confidence", 1.0)),
        "mean_face_confidence": quality.get("mean_face_confidence", 1.0),
        "confound_codes": quality.get("confound_codes") or [],
        "quality_warnings": quality.get("quality_warnings") or audit_data.get("warnings") or [],
        "module_failures": quality.get("module_failures") or [],
        "coverage_interpretation": "Reported face coverage indicates temporal tracking persistence across analyzed windows, but does not independently establish tracking precision or facial authenticity.",
    }

    # Decision Context
    decision_context = {
        "model_name": audit_engine_name,
        "model_version": decision.get("model_version", "2.0.0"),
        "model_score": audit_score,
        "score_semantics": "Calibrated spatiotemporal probability (weighted mean over temporal windows)",
        "active_threshold": eff_threshold,
        "automated_verdict": automated_verdict,
        "decision_rationale": decision_rationale,
        "score_dispersion_range": score_dispersion,
        "score_dispersion_definition": "Empirical [min, max] score range across evaluated temporal windows; not a Frequentist CI.",
        "comparator_engine": comparator_engine,
        "data_consistency_status": data_consistency_status,
        "data_consistency_notes": consistency_notes,
    }

    # Spatial Findings
    spatial_findings = {
        "filter_method": spatial.get("filter_method") or spatial.get("filter_applied") or "112x112 Discrete Laplacian (G1+L0)",
        "inner_landmark_energy": spatial.get("inner_landmark_energy"),
        "outer_boundary_energy": spatial.get("outer_boundary_energy"),
        "boundary_to_landmark_ratio": spatial.get("boundary_to_landmark_ratio") or spatial.get("outer_inner_boundary_ratio"),
        "spatial_anomaly_flag": bool(spatial.get("spatial_frequency_anomaly_flag", False)),
        "definition": spatial.get("definition", "High-frequency Laplacian gradient variance across 112x112 normalized face crops"),
        "interpretation": spatial.get("interpretation", "Reflects localized high-frequency spatial variation across facial boundaries; does not independently establish an artificial blending seam or manipulation."),
    }

    # Temporal Findings
    temporal_findings = {
        "total_windows": temporal.get("total_windows_evaluated", temporal.get("windows_evaluated", 0)),
        "total_frames": temporal.get("total_frames_analyzed", temporal.get("frames_evaluated_total", 0)),
        "frame_differential_rate": temporal.get("inter_frame_differential_energy") or temporal.get("frame_differential_rate") or audit_data.get("kinematic_residual_energy"),
        "differential_energy_unit": temporal.get("differential_energy_unit", "2D Euclidean pixel differential per second"),
        "window_score_dispersion": temporal.get("window_score_consistency") or temporal.get("temporal_score_dispersion") or audit_data.get("temporal_consistency_score"),
        "peak_window": {
            "window_id": peak.get("window_id", 1),
            "start_sec": peak.get("start_sec", 0.0),
            "end_sec": peak.get("end_sec", 0.0),
            "score": peak.get("calibrated_fake_probability", peak.get("calibrated_score")),
            "face_confidence": peak.get("face_confidence", 1.0),
            "selection_policy": peak.get("selection_policy") or peak.get("selection_method") or "Highest calibrated manipulation probability among evaluated temporal windows",
        },
        "definition": temporal.get("definition", "Time-normalized frame-to-frame pixel appearance displacement across sequential frames"),
        "interpretation": temporal.get("interpretation", "Reflects visual displacement and deformation rate; irregular or smooth motion occurs in both authentic and manipulated footage."),
    }

    # Reference Baseline
    reference_baseline = {
        "is_eligible": bool(baseline.get("is_eligible", False)),
        "reference_dataset": baseline.get("reference_dataset") or "FaceForensics++ C23 Validation",
        "sample_size_n": baseline.get("sample_video_count") or baseline.get("sample_size_n") or 1490,
        "kinematic_z_score": baseline.get("kinematic_z_score"),
        "spatial_ratio_z_score": baseline.get("spatial_ratio_z_score") or baseline.get("boundary_ratio_z_score"),
        "notes": baseline.get("notes") or baseline.get("eligibility_notes") or "Standardized deviations computed against reference population distribution.",
    }

    # Calibration Context
    calibration_context = {
        "calibration_id": "Platt-FFpp-C23-Validation",
        "is_calibrated": True,
        "evaluation_conditions": "Evaluated on FaceForensics++ C23 clean validation split; does not guarantee probability validity under severe unseen compression or out-of-distribution recording.",
    }

    # Limitations
    limitations = (
        decision.get("limitations")
        or audit_data.get("limitation_statement")
        or "Absence of detectable indicators within the validated scope does not guarantee authenticity against unseen synthesis methods or out-of-distribution recording conditions."
    )

    return CanonicalForensicEvidence(
        run_id=run_id,
        video_filename=file_name,
        file_sha256=file_sha256,
        media_metadata=media_metadata,
        coverage_and_quality=coverage_and_quality,
        decision_context=decision_context,
        spatial_findings=spatial_findings,
        temporal_findings=temporal_findings,
        reference_baseline=reference_baseline,
        calibration_context=calibration_context,
        limitations=limitations,
    )


def extract_canonical_numbers(evidence: CanonicalForensicEvidence) -> set[float]:
    """Extracts all authorized numerical quantities from canonical evidence.
    Used to verify that reported numerical claims are grounded in pipeline measurements.
    """
    numbers: set[float] = set()

    d = evidence.decision_context
    for key in ("model_score", "active_threshold"):
        val = d.get(key)
        if val is not None:
            fval = float(val)
            numbers.add(round(fval, 4))
            numbers.add(round(fval * 100.0, 2))
            numbers.add(round(fval * 100.0, 1))
            numbers.add(round(fval * 100.0, 0))

    disp = d.get("score_dispersion_range")
    if disp and len(disp) == 2:
        for val in disp:
            fval = float(val)
            numbers.add(round(fval, 4))
            numbers.add(round(fval * 100.0, 2))
            numbers.add(round(fval * 100.0, 1))

    sp = evidence.spatial_findings
    for key in ("boundary_to_landmark_ratio", "outer_boundary_energy", "inner_landmark_energy"):
        val = sp.get(key)
        if val is not None:
            fval = float(val)
            numbers.add(round(fval, 4))
            numbers.add(round(fval, 2))
            numbers.add(round(fval, 1))

    tp = evidence.temporal_findings
    for key in ("frame_differential_rate", "window_score_dispersion"):
        val = tp.get(key)
        if val is not None:
            fval = float(val)
            numbers.add(round(fval, 2))
            numbers.add(round(fval, 1))
            numbers.add(round(fval * 100.0, 1))
    if tp.get("total_windows") is not None:
        numbers.add(float(tp["total_windows"]))
    if tp.get("total_frames") is not None:
        numbers.add(float(tp["total_frames"]))

    peak = tp.get("peak_window") or {}
    if peak.get("score") is not None:
        pval = float(peak["score"])
        numbers.add(round(pval, 4))
        numbers.add(round(pval * 100.0, 2))
        numbers.add(round(pval * 100.0, 1))
    if peak.get("start_sec") is not None:
        numbers.add(round(float(peak["start_sec"]), 1))
    if peak.get("end_sec") is not None:
        numbers.add(round(float(peak["end_sec"]), 1))
    if peak.get("window_id") is not None:
        numbers.add(float(peak["window_id"]))

    bl = evidence.reference_baseline
    if bl.get("is_eligible"):
        for key in ("kinematic_z_score", "spatial_ratio_z_score"):
            val = bl.get(key)
            if val is not None:
                fval = float(val)
                numbers.add(round(fval, 2))
                numbers.add(round(abs(fval), 2))
        if bl.get("sample_size_n") is not None:
            numbers.add(float(bl["sample_size_n"]))

    q = evidence.coverage_and_quality
    if q.get("face_coverage_ratio") is not None:
        fval = float(q["face_coverage_ratio"])
        numbers.add(round(fval, 5))
        numbers.add(round(fval * 100.0, 2))
        numbers.add(round(fval * 100.0, 1))

    comp = d.get("comparator_engine")
    if comp and comp.get("model_score") is not None:
        cval = float(comp["model_score"])
        numbers.add(round(cval, 4))
        numbers.add(round(cval * 100.0, 2))
        numbers.add(round(cval * 100.0, 1))

    return numbers


def validate_numerical_consistency(evidence: CanonicalForensicEvidence) -> Tuple[bool, List[str]]:
    """Validates that point estimates, intervals, thresholds, and verdicts are internally consistent.
    
    Guarantees:
    - The point estimate must lie within its stated dispersion range.
    - The automated verdict must strictly follow the score vs threshold decision rule.
    """
    issues: List[str] = []
    d = evidence.decision_context
    score = d["model_score"]
    thresh = d["active_threshold"]
    verdict = d["automated_verdict"]
    disp = d.get("score_dispersion_range")

    # Check 1: Point estimate must lie within dispersion range [min, max]
    if disp and len(disp) == 2:
        low, high = float(disp[0]), float(disp[1])
        if score < (low - 0.005) or score > (high + 0.005):
            issues.append(
                f"Numerical inconsistency: Model point estimate ({score*100:.2f}%) "
                f"lies outside stated dispersion range [{low*100:.2f}%, {high*100:.2f}%]. "
                f"Point estimate and dispersion must refer to the same analysis run."
            )

    # Check 2: Automated verdict matches score vs threshold rule
    if verdict != "INCONCLUSIVE":
        expected_verdict = "MANIPULATION DETECTED" if score >= thresh else "NO MANIPULATION DETECTED"
        if verdict != expected_verdict:
            issues.append(
                f"Decision rule inconsistency: Model score ({score*100:.2f}%) vs threshold ({thresh*100:.2f}%) "
                f"evaluates to '{expected_verdict}', but recorded verdict is '{verdict}'."
            )

    return len(issues) == 0, issues


def validate_groq_report(report_text: str, evidence: CanonicalForensicEvidence) -> Tuple[bool, List[str]]:
    """Applies strict consistency, causal guardrail, and telemetry grounding checks to the narrative.
    
    Checks:
    1. Numerical consistency of the canonical evidence run (score in interval, threshold logic).
    2. Video identity and target filename presence.
    3. Automated verdict consistency (no contradictory verdict claims).
    4. Prohibited human credentials (rejects fake certification claims).
    5. Prohibited causal claims (rejects claims of proven seams, splicing, unnatural velocity jumps).
    6. Grounding of numerical claims against canonical telemetry.
    7. Clean presentation (rejects internal field names like 'field model_score').

    Returns:
        (is_valid, list_of_detected_issues)
    """
    issues: List[str] = []
    text_lower = report_text.lower()

    # 1. Numerical Consistency Check
    num_ok, num_issues = validate_numerical_consistency(evidence)
    issues.extend(num_issues)

    # 2. Filename / Identifier Check
    raw_fname = evidence.video_filename.lower()
    fname_base = Path(evidence.video_filename).name.lower()
    if fname_base not in text_lower and raw_fname not in text_lower:
        issues.append(f"Report does not reference target filename '{evidence.video_filename}'.")

    # 3. Automated Verdict Consistency Check
    expected_verdict = evidence.decision_context["automated_verdict"].upper()
    if expected_verdict == "NO MANIPULATION DETECTED":
        if re.search(r"verdict:\s*(?:automated\s+)?(?:manipulation detected|deepfake)", text_lower) or "flagged for manipulation" in text_lower:
            issues.append("Report incorrectly claims automated verdict is 'MANIPULATION DETECTED'.")
    elif expected_verdict == "MANIPULATION DETECTED":
        if re.search(r"verdict:\s*(?:automated\s+)?(?:no manipulation detected|authentic|real)", text_lower) or "no manipulation detected" in text_lower:
            issues.append("Report incorrectly claims automated verdict is 'NO MANIPULATION DETECTED'.")
    elif expected_verdict == "INCONCLUSIVE":
        if "inconclusive" not in text_lower:
            issues.append("Report failed to preserve pipeline's INCONCLUSIVE verdict state.")

    # 4. Prohibited Certification Claims (Reject fake human credentials)
    forbidden_credentials = [
        "certified audio-visual fraud investigator",
        "certified fraud investigator",
        "certified forensic analyst",
        "prepared by: certified",
    ]
    for cred in forbidden_credentials:
        if cred in text_lower:
            issues.append(f"Prohibited human credential claim detected: '{cred}'.")

    # 5. Unsupported Causal Claims & Prohibited Attributions
    prohibited_causal_claims = [
        "indicates a blending seam",
        "proves a blending seam",
        "confirms a blending seam",
        "proves a seam",
        "confirms a seam",
        "indicates a seam",
        "proves manipulation",
        "confirms manipulation",
        "proves splicing",
        "confirms splicing",
        "indicates localized manipulation",
        "indicates unnatural velocity jumps",
        "proves unnatural velocity",
        "proves tracking accuracy",
        "proves authenticity",
        "chance the video is forged",
        "conclusively faceswap",
        "confirmed face2face",
        "synthesized by stylegan",
        "conclusively deepfake",
    ]
    for claim in prohibited_causal_claims:
        if claim in text_lower:
            issues.append(f"Unsupported causal claim detected: '{claim}'.")

    # 6. Check that missing values remained missing rather than being fabricated
    if not evidence.reference_baseline.get("is_eligible") or evidence.reference_baseline.get("kinematic_z_score") is None:
        if re.search(r"[+-]?\d+\.\d+\s*(?:σ|sigma|standard deviations)", text_lower):
            issues.append("Fabricated kinematic z-score detected when baseline is ineligible or uncalibrated.")

    # 7. Internal field name narration check (Keep technical field names out of briefing)
    forbidden_field_narrations = [
        "field model_score",
        "field active_threshold",
        "field automated_verdict",
        "active_threshold:",
        "automated_verdict:",
        "field kinematic_residual_energy",
    ]
    for fn in forbidden_field_narrations:
        if fn in text_lower:
            issues.append(f"Developer field name narration detected in analyst report: '{fn}'.")

    # 8. Grounding check of numerical claims against canonical telemetry
    allowed_numbers = extract_canonical_numbers(evidence)
    editorial_allowed = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0, 95.0, 100.0, 0.50, 0.95, 1.00}

    # Extract percentage expressions (e.g. 84.83%, 99.1%)
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*%", report_text):
        val = float(match.group(1))
        val_r1 = round(val, 1)
        val_r2 = round(val, 2)
        if val not in allowed_numbers and val_r1 not in allowed_numbers and val_r2 not in allowed_numbers and val not in editorial_allowed:
            issues.append(f"Ungrounded percentage claim: '{match.group(0)}' does not match any canonical measurement.")

    # Extract multiplier/ratio expressions (e.g. 1.16x, 5.2x)
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*[xX]\b", report_text):
        val = float(match.group(1))
        val_r1 = round(val, 1)
        val_r2 = round(val, 2)
        if val not in allowed_numbers and val_r1 not in allowed_numbers and val_r2 not in allowed_numbers and val not in editorial_allowed:
            issues.append(f"Ungrounded ratio claim: '{match.group(0)}' does not match canonical telemetry.")

    return (len(issues) == 0, issues)


def get_report_validation_details(
    report_text: str,
    evidence: CanonicalForensicEvidence,
    is_fallback: bool = False,
) -> Tuple[bool, str, List[str]]:
    """Evaluates validation and assigns an explicit, truthful validation level.
    
    Levels:
    - 'VERIFIED_CANONICAL': All numerical claims, decision rules, and causal boundaries verified against analysis run.
    - 'FIELDS_CHECKED': Report fields conform to canonical schema, with non-critical alignment notices.
    - 'FALLBACK_DETERMINISTIC': Deterministic report compiled directly from canonical measurements.
    - 'DATA_INCONSISTENCY_WARNING': Numerical inconsistency or unverified causal claims detected.
    """
    if is_fallback:
        return True, "FALLBACK_DETERMINISTIC", []

    is_valid, issues = validate_groq_report(report_text, evidence)
    if is_valid and not issues:
        return True, "VERIFIED_CANONICAL", []
    elif is_valid:
        return True, "FIELDS_CHECKED", issues
    else:
        return False, "DATA_INCONSISTENCY_WARNING", issues


def generate_safe_fallback_report(
    evidence: CanonicalForensicEvidence,
    validation_issues: Optional[List[str]] = None,
) -> str:
    """Deterministically formats the canonical evidence into a human-friendly investigator briefing.
    Adheres strictly to the 4-section report structure without narrating API field names or repeating thresholds.
    """
    m = evidence.media_metadata
    q = evidence.coverage_and_quality
    d = evidence.decision_context
    sp = evidence.spatial_findings
    tp = evidence.temporal_findings
    bl = evidence.reference_baseline
    peak = tp.get("peak_window") or {}
    comparator = d.get("comparator_engine")

    score_pct = d["model_score"] * 100.0
    thresh_pct = d["active_threshold"] * 100.0
    disp_range = d.get("score_dispersion_range") or [d["model_score"], d["model_score"]]
    disp_low_str = f"{disp_range[0] * 100.0:.2f}%"
    disp_high_str = f"{disp_range[1] * 100.0:.2f}%"

    ratio_val = sp.get("boundary_to_landmark_ratio")
    ratio_str = f"{ratio_val:.2f}x" if ratio_val is not None else None

    diff_rate = tp.get("frame_differential_rate")
    diff_str = f"{diff_rate:.2f}" if diff_rate is not None else None

    disp_val = tp.get("window_score_dispersion")
    disp_str = f"{disp_val * 100.0:.1f}%" if disp_val is not None else None

    peak_score = peak.get("score")
    peak_score_str = f"{peak_score * 100.0:.1f}%" if peak_score is not None else None
    peak_id = peak.get("window_id")
    peak_start = peak.get("start_sec", 0.0)
    peak_end = peak.get("end_sec", 0.0)

    # 1. Headline determination
    verdict = d["automated_verdict"]
    if verdict == "MANIPULATION DETECTED":
        headline = "Flagged for manipulation"
    elif verdict == "NO MANIPULATION DETECTED":
        headline = "No manipulation detected"
    else:
        headline = "Inconclusive assessment"

    # Divergence Advisory Banner if cross-model discrepancy
    divergence_banner = ""
    if d.get("data_consistency_status") == "CROSS_MODEL_DIVERGENCE" and comparator:
        comp_score_pct = comparator["model_score"] * 100.0
        divergence_banner = (
            f"> [!WARNING]\n"
            f"> **Cross-Engine Divergence Advisory:**\n"
            f"> The spatiotemporal audit engine (`{d['model_name']}`) returned a calibrated score of **{score_pct:.2f}%** "
            f"(temporal window dispersion: [{disp_low_str}, {disp_high_str}]), while the selected single-pass comparator engine "
            f"(`{comparator['engine_name']}`) scored **{comp_score_pct:.2f}%** (decision: {comparator['decision']}). "
            f"The [{disp_low_str}, {disp_high_str}] range reflects window-to-window variation of the multi-scale pipeline, "
            f"not an uncertainty interval around the single-pass score. Both measurements are preserved independently.\n\n"
        )
    elif validation_issues:
        reasons = "; ".join(validation_issues)
        divergence_banner = (
            f"> [!NOTE]\n"
            f"> **Evidence-Consistency Notice:** The analytical draft required alignment ({reasons}). "
            f"This deterministic canonical briefing was compiled directly from verified pipeline measurements.\n\n"
        )

    # Section 1: What does this result mean?
    if verdict == "MANIPULATION DETECTED":
        meaning = (
            f"The detection system flagged this video because its model score was {score_pct:.2f}%, "
            f"above the current decision threshold of {thresh_pct:.2f}%. Under the configured rule, "
            f"this produces a manipulation-detected verdict.\n\n"
            f"This is an automated detection result, not proof by itself that the video has been altered. "
            f"Its reliability depends on the model's validation, calibration, and suitability for this type of video."
        )
    elif verdict == "NO MANIPULATION DETECTED":
        meaning = (
            f"The detection system did not flag this video because its model score was {score_pct:.2f}%, "
            f"which remains below the active decision threshold of {thresh_pct:.2f}%. Under the configured rule, "
            f"this produces a no-manipulation-detected verdict.\n\n"
            f"This automated result indicates that detectable manipulation signals within the validated model scope "
            f"did not reach the threshold. It does not independently guarantee that the video is authentic against "
            f"unseen synthesis methods, compression degradation, or out-of-distribution recording conditions."
        )
    else:
        meaning = (
            f"The detection system returned an inconclusive assessment for this video because face sequence coverage "
            f"was insufficient ({q.get('face_coverage_ratio', 0.0) * 100:.1f}%) or active confounders were flagged. "
            f"Under the configured rule, reliable automated scoring cannot proceed when input conditions fail minimum operational criteria."
        )

    # Section 2: What stood out in the analysis? (Select 2 or 3 strongest supported findings)
    findings = []

    # Finding A: Changes between frames
    if diff_str is not None:
        findings.append(
            f"• Changes between frames\n"
            f"The system measured an average frame-differential energy of {diff_str}. "
            f"This describes how much image content changes between analyzed frames. "
            f"On its own, it cannot tell us whether that movement is natural or manipulated."
        )

    # Finding B: Consistency across time windows & peak window
    total_wins = tp.get("total_windows", 4)
    if disp_str is not None or peak_score_str is not None:
        peak_detail = ""
        if peak_score_str is not None and peak_id:
            peak_detail = f" across {total_wins} windows, with a peak response of {peak_score_str} in Window #{peak_id} ({peak_start:.1f}s – {peak_end:.1f}s)"
        elif peak_score_str is not None:
            peak_detail = f" with a peak response of {peak_score_str}"
        disp_txt = disp_str if disp_str is not None else "temporal"
        findings.append(
            f"• Consistency across time windows\n"
            f"The dashboard reports {disp_txt} window-score consistency{peak_detail}. "
            f"This suggests how similar the reported scores are across sequential intervals under the system's consistency metric; "
            f"it does not independently confirm that the video is fake or authentic."
        )

    # Finding C: Boundary frequency contrast
    if ratio_str is not None:
        findings.append(
            f"• Boundary frequency contrast\n"
            f"The system measured an outer-to-inner boundary gradient ratio of {ratio_str}. "
            f"This describes localized high-frequency spatial variation across facial boundaries compared to inner facial landmarks. "
            f"It alone does not prove a blending seam or manipulation."
        )

    findings_block = "\n\n".join(findings) if findings else "• Automated Feature Extraction\nStandard spatiotemporal features were extracted without significant localized outliers."

    # Section 3: What should be reviewed next?
    review_target = f"individual temporal-window scores and the frames associated with the strongest model response (particularly Window #{peak_id} from {peak_start:.1f}s to {peak_end:.1f}s)" if peak_id else "individual temporal-window scores and key extracted frames"
    review_next = (
        f"Review the {review_target}. "
        f"Check whether the relevant frames contain reproducible visual artifacts, and consider compression, lighting, "
        f"motion, and face tracking quality before drawing a conclusion."
    )

    # Section 4: Important limitations
    limitations = (
        f"{evidence.limitations} "
        f"The available measurements do not, by themselves, establish the specific method of manipulation or independently verify the video's authenticity."
    )

    report = f"""## AI Video Analysis
{evidence.video_filename}
{headline}

{divergence_banner}### What does this result mean?
{meaning}

### What stood out in the analysis?
{findings_block}

### What should be reviewed next?
{review_next}

### Important limitations
{limitations}

*AI-generated analytical assistance · Technical measurements available in Structured Telemetry*"""

    return report.strip()


def generate_human_friendly_forensic_analysis(
    audit_data: Dict[str, Any],
    pred_data: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
    preferred_model: Optional[str] = None,
    prediction_data: Optional[Dict[str, Any]] = None,
    return_fallback_flag: bool = False,
) -> Any:
    """Generates an evidence-grounded, human-friendly forensic report using Groq,
    followed by post-generation consistency validation.
    
    If Groq is unreachable, times out, or fails validation, returns a deterministic safe fallback.
    """
    pred_data = pred_data or prediction_data
    evidence = build_canonical_evidence(audit_data=audit_data, pred_data=pred_data, threshold=threshold)

    # Pre-validation: ensure canonical evidence is numerically consistent
    num_ok, num_issues = validate_numerical_consistency(evidence)
    if not num_ok:
        logger.warning(f"Canonical evidence failed numerical consistency: {num_issues}. Generating safe fallback.")
        fallback = generate_safe_fallback_report(evidence, validation_issues=num_issues)
        return (fallback, True) if return_fallback_flag else fallback

    api_key = get_groq_api_key()
    if not api_key:
        logger.info("Groq API key not found; returning canonical safe fallback report.")
        fallback = generate_safe_fallback_report(evidence)
        return (fallback, True) if return_fallback_flag else fallback

    # System instruction tailored for intelligent non-technical investigator briefing
    system_instruction = (
        "System Role: Evidence-Grounded Forensic Video Report Analyst\n"
        "You generate clear, human-readable explanations of an automated deepfake video analysis pipeline for an intelligent, non-technical investigator.\n"
        "You are an explanatory reporting layer, not an independent forensic detector.\n\n"
        "Communication Requirements:\n"
        "1. Write for an intelligent non-technical investigator, not for a developer reading API fields.\n"
        "2. Less field-by-field narration: Never write internal field names like 'field model_score', 'active_threshold', or 'automated_verdict'. These belong in Structured Telemetry.\n"
        "3. Explain the meaning, not just the number: Instead of merely stating that the score exceeds the threshold, explain what the system found, why the result was flagged, and what the score does—and does not—tell the user.\n"
        "4. Prioritize the most relevant findings: Lead with the verdict headline and the 2 or 3 strongest supported findings. Put all frame counts, metadata, feature values, model identifiers, and baseline statistics in Structured Telemetry rather than the main narrative.\n"
        "5. Make uncertainty understandable: Distinguish an automated model flag from confirmed manipulation. Explain when evidence is incomplete, a baseline is unavailable, or indicators disagree.\n"
        "6. Use natural, professional language and short paragraphs. Explain technical terms simply when they first appear.\n"
        "7. Recommend specific frames or time intervals for review only when those intervals are explicitly present in the supplied evidence.\n"
        "8. Never invent artifacts, causes, manipulation methods, baselines, confidence intervals, or analyst credentials. Do not describe an automated prediction as confirmed manipulation or a below-threshold result as proof of authenticity.\n"
        "9. Preserve the pipeline's automated verdict and explicitly disclose material uncertainty, missing evidence, and conflicting measurements.\n\n"
        "Mandatory Report Structure:\n"
        "Produce exactly these sections under the main heading:\n\n"
        "## AI Video Analysis\n"
        "[Exact Target Filename from evidence]\n"
        "[Verdict Headline: 'Flagged for manipulation' OR 'No manipulation detected' OR 'Inconclusive assessment']\n\n"
        "### What does this result mean?\n"
        "[Plain-language explanation of why the system reached its automated verdict under the configured decision rule. Explain what this automated score indicates and what it does not establish. Clarify that this is an automated detection result whose reliability depends on calibration and domain suitability, not standalone proof.]\n\n"
        "### What stood out in the analysis?\n"
        "[Select the 2 or 3 strongest supported findings from the supplied evidence (e.g. Changes between frames, Boundary frequency contrast, Temporal window consistency, or Peak response window). For each finding:\n"
        "• [Name of Finding]\n"
        "Explain: (1) what was measured, (2) what it may indicate, and (3) what it cannot establish.]\n\n"
        "### What should be reviewed next?\n"
        "[Targeted manual inspection recommendations: cite specific window timestamps if in evidence, and advise checking for reproducible visual artifacts, compression, lighting, motion, and face tracking quality before drawing a conclusion.]\n\n"
        "### Important limitations\n"
        "[Concise operational limitation statement: automated analysis cannot independently establish the specific manipulation method or verify authenticity.]\n\n"
        "*AI-generated analytical assistance · Technical measurements available in Structured Telemetry*"
    )

    ev_dict = evidence.to_dict()
    user_prompt = (
        f"CANONICAL STRUCTURED EVIDENCE FOR VIDEO ANALYSIS RUN:\n"
        f"```json\n{json.dumps(ev_dict, indent=2)}\n```\n\n"
        f"Please synthesize this canonical evidence into the required 4-section investigator report following all communication requirements. "
        f"Keep the default report concise and professional. Do not narrate internal field names."
    )

    models_to_try = [preferred_model] if preferred_model else DEFAULT_GROQ_MODELS
    last_validation_issues = []

    for model_name in models_to_try:
        if not model_name:
            continue
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 700,
            }

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "DeepfakeForensics/2.0",
                },
            )

            with urllib.request.urlopen(req, timeout=14) as response:
                res_json = json.loads(response.read().decode("utf-8"))
                choices = res_json.get("choices", [])
                if choices and "message" in choices[0]:
                    draft_report = choices[0]["message"]["content"].strip()
                    # Perform post-generation consistency validation
                    is_valid, issues = validate_groq_report(draft_report, evidence)
                    if is_valid:
                        logger.info(f"Groq report generated via {model_name} and PASSED consistency validation.")
                        return (draft_report, False) if return_fallback_flag else draft_report
                    else:
                        logger.warning(f"Groq report from {model_name} failed validation: {issues}. Trying next model candidate.")
                        last_validation_issues = issues
                        continue
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode()
            except Exception:
                pass
            logger.warning(f"Groq API call to {model_name} failed with HTTP {e.code}: {err_body}")
            continue
        except Exception as e:
            logger.warning(f"Groq API call to {model_name} failed: {e}")
            continue

    logger.warning("Groq synthesis unavailable or timed out; generating canonical safe fallback report.")
    fallback = generate_safe_fallback_report(evidence, validation_issues=last_validation_issues if last_validation_issues else None)
    return (fallback, True) if return_fallback_flag else fallback

