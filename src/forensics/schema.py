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
