"""Complete schema for Forensic Evidence Engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch


class FindingType(str, Enum):
    MODEL_PREDICTION = "MODEL_PREDICTION"          # Detector logit/probability
    MODEL_EXPLANATION = "MODEL_EXPLANATION"        # Grad-CAM heatmap (model attention)
    ANOMALY_ONLY = "ANOMALY_ONLY"                  # Statistical deviation without validated deepfake link
    VALIDATED_FORENSIC_EVIDENCE = "VALIDATED_FORENSIC_EVIDENCE"  # Proven discriminative deepfake indicator


class EvidenceSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConfoundType(str, Enum):
    VIDEO_COMPRESSION = "VIDEO_COMPRESSION"        # H.264/H.265 blocking, quantization
    MOTION_BLUR = "MOTION_BLUR"                    # Rapid optical flow, camera shake
    TRACKING_INSTABILITY = "TRACKING_INSTABILITY"  # Low confidence, track fragmentations
    LOW_RESOLUTION = "LOW_RESOLUTION"              # Face crop < 64x64
    EXTREME_LIGHTING = "EXTREME_LIGHTING"          # Severe under/over exposure


class ModalityGroup(str, Enum):
    SPATIAL_FREQUENCY = "SPATIAL_FREQUENCY"        # FFT, DCT, noise residuals
    FACIAL_KINEMATICS = "FACIAL_KINEMATICS"        # 3D Landmark velocities, eye/mouth decoupling
    TEMPORAL_COHERENCE = "TEMPORAL_COHERENCE"      # Frame differentials, optical flow
    CROSS_MODAL = "CROSS_MODAL"                    # Lip-audio synchrony


class ManipulationAssessment(str, Enum):
    SUPPORTING_MANIPULATION = "SUPPORTING_MANIPULATION"
    NO_SIGNIFICANT_INDICATORS = "NO_SIGNIFICANT_INDICATORS"
    INCONCLUSIVE = "INCONCLUSIVE"


class ProvenanceAssessment(str, Enum):
    VALID_CREDENTIAL = "VALID_CREDENTIAL"
    INVALID_CREDENTIAL = "INVALID_CREDENTIAL"
    NO_CREDENTIAL_FOUND = "NO_CREDENTIAL_FOUND"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"


class ModuleExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    SKIPPED_UNSUPPORTED = "SKIPPED_UNSUPPORTED"
    SKIPPED_CONFOUNDED = "SKIPPED_CONFOUNDED"
    FAILED = "FAILED"
    UNCALIBRATED = "UNCALIBRATED"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"


class ConfoundCode(str, Enum):
    VIDEO_COMPRESSION = "video_compression"
    MOTION_BLUR = "motion_blur"
    RAPID_MOTION = "rapid_motion"
    TRACKING_INSTABILITY = "tracking_instability"
    LOW_RESOLUTION = "low_resolution"
    EXTREME_LIGHTING = "extreme_lighting"


@dataclass(frozen=True)
class EvidenceDependency:
    """Tracks module dependency to prevent correlated signals from being counted as independent."""
    finding_id: str
    modality: str
    dependency_group: str     # e.g., SPATIAL_TEXTURE, COMPRESSION_SENSITIVE, TRACKING_DERIVED, LANDMARK_KINEMATICS, MODEL_EXPLANATION
    calibration_id: Optional[str] = None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PluginResult:
    """Standardized result returned by all forensic model adapters and signal extractors."""
    plugin_name: str
    plugin_version: str
    status: str
    findings: tuple[Any, ...] = field(default_factory=tuple)
    coverage: float = 0.0
    calibration_id: Optional[str] = None
    error_code: Optional[str] = None


class ForensicPlugin:
    """Standard plugin interface for emerging detector models and forensic extractors."""
    name: str = "base_plugin"
    version: str = "1.0.0"

    def can_analyze(self, ctx: Any) -> ModuleExecutionStatus:
        raise NotImplementedError

    def analyze(self, ctx: Any) -> PluginResult:
        raise NotImplementedError


@dataclass
class MeasurementDefinition:
    name: str
    description: str
    unit: str
    modality_group: ModalityGroup
    directional_tail: str                          # "UPPER_TAIL" | "LOWER_TAIL" | "TWO_TAILED"
    validated_for_manipulation: bool = False       # True only if validated to distinguish deepfakes
    relevant_confounds: List[ConfoundType] = field(default_factory=list)


@dataclass
class BaselineComparison:
    baseline_version: str
    metric_name: str
    sample_value: float
    baseline_n: int
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    baseline_median: Optional[float] = None
    baseline_iqr: Optional[float] = None
    baseline_lower_threshold: Optional[float] = None
    baseline_upper_threshold: Optional[float] = None
    z_score: Optional[float] = None
    p_value: Optional[float] = None
    is_anomaly: bool = False
    is_calibrated: bool = False


@dataclass
class EvidenceFinding:
    finding_id: str
    module: str
    modality_group: ModalityGroup
    finding_type: FindingType
    severity: EvidenceSeverity
    title: str
    description: str
    numeric_value: Optional[float] = None
    baseline_comparison: Optional[BaselineComparison] = None
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    frame_indices: List[int] = field(default_factory=list)
    visual_artifact_uri: Optional[str] = None
    alternative_explanations: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)


@dataclass
class ForensicEvidenceReport:
    report_id: str
    video_filename: str
    file_sha256: str
    container_duration_sec: float
    analyzed_duration_sec: float
    total_frames_analyzed: int
    manipulation_assessment: ManipulationAssessment
    provenance_assessment: ProvenanceAssessment
    executive_summary: str
    findings: List[EvidenceFinding]
    module_statuses: Dict[str, ModuleExecutionStatus]
    quality_assessment: Dict[str, Any]
    limitations_and_caveats: List[str]
    provenance_details: Dict[str, Any]
    reproducibility_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Custom serializer converting numpy/torch/enum/pathlib objects to JSON primitives."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, bool):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
            return obj

        return _convert(self)


# ==============================================================================
# Video-Specific Forensic Observations & Structured Telemetry
# ==============================================================================

@dataclass
class StreamTelemetry:
    """Detailed container and stream telemetry derived from actual video input."""
    source_file_name: str
    file_sha256: str
    file_size_bytes: int
    duration_sec: float
    width: int
    height: int
    codec_name: str
    container_fps: float
    is_vfr: bool
    bitrate_kbps: float
    decode_status: str = "SUCCESS"


@dataclass
class SpatialTelemetry:
    """Spatial measurement telemetry and mathematical definitions."""
    filter_method: str
    target_face_size: int
    overall_saliency_mean: float
    inner_landmark_energy: float
    outer_boundary_energy: float
    boundary_to_landmark_ratio: float
    definition: str = "High-frequency Laplacian gradient variance across 112x112 normalized face crops"
    interpretation: str = "Reflects localized high-frequency spatial variation; does not independently confirm artificial boundary compositing."


@dataclass
class PeakWindowDetail:
    """Details of the highest-scoring temporal observation window."""
    window_id: int
    start_sec: float
    end_sec: float
    frame_count: int
    face_confidence: float
    calibrated_fake_probability: float
    selection_method: str = "Highest calibrated manipulation probability among evaluated temporal windows"


@dataclass
class TemporalTelemetry:
    """Temporal measurements, window boundaries, and appearance displacement rate."""
    total_windows_evaluated: int
    window_duration_sec: float
    frames_per_window: int
    total_frames_analyzed: int
    inter_frame_differential_energy: float
    differential_energy_unit: str = "2D Euclidean pixel differential per second"
    window_score_consistency: float = 1.0
    temporal_series: List[float] = field(default_factory=list)
    peak_window: Optional[PeakWindowDetail] = None
    definition: str = "Time-normalized frame-to-frame pixel appearance displacement across sequential frames"
    interpretation: str = "Reflects visual displacement and deformation rate; irregular or smooth motion occurs in both authentic and manipulated footage."


@dataclass
class BaselineTelemetry:
    """Comparison results against an empirical reference distribution."""
    is_eligible: bool
    reference_dataset: Optional[str] = None
    sample_video_count: Optional[int] = None
    kinematic_z_score: Optional[float] = None
    kinematic_is_anomaly: bool = False
    spatial_ratio_z_score: Optional[float] = None
    spatial_ratio_is_anomaly: bool = False
    notes: str = ""


@dataclass
class QualityAndConfounderTelemetry:
    """Explicit tracking coverage, quality flags, and confounders."""
    face_detected_any: bool
    mean_face_confidence: float
    confound_codes: List[str] = field(default_factory=list)
    quality_warnings: List[str] = field(default_factory=list)
    sufficient_coverage: bool = True


@dataclass
class ForensicDecisionTelemetry:
    """Final decision state, threshold policy, and legal disclaimer."""
    decision_state: str  # SUPPORTING_MANIPULATION, NO_SIGNIFICANT_INDICATORS, INCONCLUSIVE
    confidence_level: str  # HIGH, MEDIUM, LOW, BORDERLINE
    calibrated_probability: float
    confidence_interval: Tuple[float, float]
    active_threshold: float
    model_name: str
    model_version: str
    calibration_method: str
    attribution: str = "Unspecified / Binary Detection Only"
    limitations: str = (
        "Absence of detectable indicators within the validated scope does not guarantee authenticity against "
        "unseen synthesis methods or out-of-distribution recording conditions."
    )


@dataclass
class VideoSpecificTelemetry:
    """Unified container for all structured forensic telemetry."""
    schema_version: str = "2.0.0"
    stream: StreamTelemetry = field(default_factory=lambda: StreamTelemetry("", "", 0, 0.0, 0, 0, "", 0.0, False, 0.0))
    spatial: SpatialTelemetry = field(default_factory=lambda: SpatialTelemetry("", 112, 0.0, 0.0, 0.0, 0.0))
    temporal: TemporalTelemetry = field(default_factory=lambda: TemporalTelemetry(0, 0.0, 0, 0, 0.0))
    baseline: BaselineTelemetry = field(default_factory=lambda: BaselineTelemetry(False))
    quality: QualityAndConfounderTelemetry = field(default_factory=lambda: QualityAndConfounderTelemetry(False, 0.0))
    decision: ForensicDecisionTelemetry = field(default_factory=lambda: ForensicDecisionTelemetry("", "", 0.0, (0.0, 0.0), 0.5, "", "", ""))

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to a JSON-serializable dictionary."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, bool):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
            return obj

        return _convert(self)


def format_telemetry_narrative(telemetry: VideoSpecificTelemetry) -> str:
    """Formats structured telemetry into clean, deterministic, standardized plain-text sections."""
    s = telemetry.stream
    sp = telemetry.spatial
    tp = telemetry.temporal
    bl = telemetry.baseline
    qc = telemetry.quality
    dec = telemetry.decision

    # 1. Stream Telemetry
    hash_short = s.file_sha256[:20] + "..." if len(s.file_sha256) > 20 else (s.file_sha256 or "N/A")
    vfr_label = "VFR (Variable Frame Rate)" if s.is_vfr else "CFR (Constant Frame Rate)"
    sec1 = (
        f"[1. STREAM TELEMETRY]\n"
        f"• Source File: {s.source_file_name} ({s.file_size_bytes:,} bytes)\n"
        f"• SHA-256: {hash_short}\n"
        f"• Stream: {s.width}x{s.height} | {s.container_fps:.1f} fps ({vfr_label}) | {s.codec_name} @ {s.bitrate_kbps:.0f} kbps\n"
        f"• Duration: {s.duration_sec:.2f}s total ({s.decode_status})"
    )

    # 2. Spatial Observations
    sec2 = (
        f"[2. SPATIAL OBSERVATIONS]\n"
        f"• Decomposition: {sp.filter_method} ({sp.target_face_size}x{sp.target_face_size} crop)\n"
        f"• High-Frequency Energy: Inner Landmarks = {sp.inner_landmark_energy:.4f}, Outer Boundary = {sp.outer_boundary_energy:.4f}\n"
        f"• Boundary-to-Landmark Ratio: {sp.boundary_to_landmark_ratio:.2f}x\n"
        f"• Physical Definition: {sp.definition}\n"
        f"• Measurement Note: {sp.interpretation}"
    )

    # 3. Temporal Observations & Peak Window
    peak_str = "None"
    if tp.peak_window:
        pw = tp.peak_window
        peak_str = (
            f"Window #{pw.window_id} ({pw.start_sec:.1f}s-{pw.end_sec:.1f}s) score: {pw.calibrated_fake_probability * 100:.1f}%\n"
            f"  Selection policy: {pw.selection_method}"
        )
    sec3 = (
        f"[3. TEMPORAL OBSERVATIONS & PEAK WINDOW]\n"
        f"• Temporal Sampling: {tp.total_windows_evaluated} windows ({tp.window_duration_sec:.1f}s, {tp.frames_per_window} frames/window, {tp.total_frames_analyzed} frames analyzed)\n"
        f"• Differential Rate: {tp.inter_frame_differential_energy:.2f} {tp.differential_energy_unit}\n"
        f"• Peak Window: {peak_str}\n"
        f"• Window Consistency: {tp.window_score_consistency * 100:.1f}% (inter-window score dispersion metric; not proof of authenticity)"
    )

    # 4. Baseline Comparisons
    if bl.is_eligible and bl.reference_dataset:
        kin_z_str = f"{bl.kinematic_z_score:+.2f}σ" if bl.kinematic_z_score is not None else "N/A"
        kin_ano = " (Elevated Anomaly)" if bl.kinematic_is_anomaly else " (Nominal)"
        sp_z_str = f"{bl.spatial_ratio_z_score:+.2f}σ" if bl.spatial_ratio_z_score is not None else "N/A"
        sp_ano = " (Elevated Anomaly)" if bl.spatial_ratio_is_anomaly else " (Nominal)"
        sec4 = (
            f"[4. BASELINE COMPARISONS]\n"
            f"• Reference Distribution: {bl.reference_dataset} (N={bl.sample_video_count})\n"
            f"• Kinematic Z-Score: {kin_z_str}{kin_ano}\n"
            f"• Boundary Ratio Z-Score: {sp_z_str}{sp_ano}\n"
            f"• Status: Valid calibrated baseline comparison"
        )
    else:
        sec4 = (
            f"[4. BASELINE COMPARISONS]\n"
            f"• Status: UNAVAILABLE ({bl.notes or 'No calibrated empirical baseline distribution configured for this metric'})\n"
            f"• Policy: Standardized z-scores omitted to prevent unvalidated comparison claims."
        )

    # 5. Quality & Confounders
    conf_str = ", ".join(qc.confound_codes) if qc.confound_codes else "None detected"
    warn_str = "; ".join(qc.quality_warnings) if qc.quality_warnings else "None"
    face_str = f"Tracked (mean confidence: {qc.mean_face_confidence * 100:.1f}%)" if qc.face_detected_any else "NO HUMAN FACE TRACKED"
    sec5 = (
        f"[5. QUALITY & CONFOUNDERS]\n"
        f"• Face Tracking: {face_str}\n"
        f"• Active Confounders: {conf_str}\n"
        f"• Quality Warnings: {warn_str}\n"
        f"• Operating Coverage: {'Sufficient' if qc.sufficient_coverage else 'INSUFFICIENT (Rejection != Fake)'}"
    )

    # 6. Forensic Assessment & Limitations
    sec6 = (
        f"[6. FORENSIC ASSESSMENT & LIMITATIONS]\n"
        f"• Assessment State: {dec.decision_state} (Confidence: {dec.confidence_level})\n"
        f"• Model Prediction: {dec.calibrated_probability * 100:.1f}% vs Active Threshold {dec.active_threshold:.2f} (Model: {dec.model_name})\n"
        f"• Score Dispersion: [{dec.confidence_interval[0] * 100:.1f}% - {dec.confidence_interval[1] * 100:.1f}%]\n"
        f"• Manipulation Attribution: {dec.attribution}\n"
        f"• Legal & Forensic Disclaimer: {dec.limitations}"
    )

    return f"{sec1}\n\n{sec2}\n\n{sec3}\n\n{sec4}\n\n{sec5}\n\n{sec6}"

