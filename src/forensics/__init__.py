"""Forensics package providing independent signal extractors, video loaders, trackers, and spatiotemporal pipeline."""

from src.forensics.loader import ForensicVideoLoader
from src.forensics.tracking import FaceTracker, compute_iou, compute_centroid_distance
from src.forensics.spatial_temporal_pipeline import SpatialTemporalForensicEngine
from src.forensics.context import (
    AnalysisContext,
    FaceTrack,
    ForensicAnalysisConfig,
    ForensicProvenanceAudit,
    ForensicVideoReport,
    FrameReference,
    TemporalWindow,
    TrackedFaceSequence,
    VideoMetadata,
    WindowForensicAnalysis,
)
from src.forensics.schema import (
    BaselineComparison,
    ConfoundType,
    EvidenceFinding,
    EvidenceSeverity,
    FindingType,
    ForensicEvidenceReport,
    ManipulationAssessment,
    MeasurementDefinition,
    ModalityGroup,
    ModuleExecutionStatus,
    ProvenanceAssessment,
)
from src.forensics.signals import (
    AudioSignal,
    CompressionSignal,
    FrequencySignal,
    IdentitySignal,
    LandmarkSignal,
    LipSyncSignal,
    MotionSignal,
    ProvenanceSignal,
    SpatialSignal,
    TemporalSignal,
)

__all__ = [
    # Pipeline & Loaders
    "ForensicVideoLoader",
    "FaceTracker",
    "SpatialTemporalForensicEngine",
    "compute_iou",
    "compute_centroid_distance",
    # Context & Container Data Structures
    "AnalysisContext",
    "FaceTrack",
    "ForensicAnalysisConfig",
    "ForensicProvenanceAudit",
    "ForensicVideoReport",
    "FrameReference",
    "TemporalWindow",
    "TrackedFaceSequence",
    "VideoMetadata",
    "WindowForensicAnalysis",
    # Forensic Evidence Schema
    "BaselineComparison",
    "ConfoundType",
    "EvidenceFinding",
    "EvidenceSeverity",
    "FindingType",
    "ForensicEvidenceReport",
    "ManipulationAssessment",
    "MeasurementDefinition",
    "ModalityGroup",
    "ModuleExecutionStatus",
    "ProvenanceAssessment",
    # Signal Extractors
    "SpatialSignal",
    "TemporalSignal",
    "MotionSignal",
    "FrequencySignal",
    "LandmarkSignal",
    "AudioSignal",
    "LipSyncSignal",
    "IdentitySignal",
    "CompressionSignal",
    "ProvenanceSignal",
]
