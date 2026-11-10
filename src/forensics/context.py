"""Ingestion context and dual-pathway frame sampling with exact Presentation Timestamps."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch


def validate_sample_count(sample_count: int) -> None:
    """Validates that requested sample count is positive and non-zero."""
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")


def resolve_torch_device(prefer_cuda: bool = True) -> torch.device:
    """Selects explicit PyTorch device rather than OpenCV CUDA backend."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@dataclass(frozen=True)
class SampledFrame:
    """Safe data contract for frame sampling preserving index and exactness lineage."""
    requested_index: Optional[int]
    decoded_index: Optional[int]
    pts_seconds: Optional[float]
    pts_is_exact: bool
    image: np.ndarray
    decode_status: str


@dataclass
class LandmarkSequence:
    """Temporal facial landmark sequence with explicit frame validity mask."""
    points: np.ndarray        # (T, N, 3); invalid entries masked or NaN
    valid_frames: np.ndarray  # (T,) boolean mask
    timestamps: np.ndarray    # (T,) float seconds


def validate_landmark_sequence(seq: LandmarkSequence) -> None:
    """Validates alignment of landmark coordinates, boolean validity mask, and timestamps."""
    if len(seq.points) != len(seq.timestamps):
        raise ValueError("Points and timestamps must align")
    if len(seq.valid_frames) != len(seq.timestamps):
        raise ValueError("Validity mask and timestamps must align")


@dataclass(frozen=True)
class QualityAssessment:
    """Computed quality gating metrics for spatial and temporal analysis eligibility."""
    flags: Tuple[str, ...]
    face_coverage: float
    timestamp_coverage: float
    decode_failure_count: int
    usable_for_spatial: bool
    usable_for_temporal: bool
    reasons: Tuple[str, ...]


@dataclass
class FrameReference:
    sample_idx: int                       # Array position in pathway
    original_frame_idx: Optional[int]    # Video frame index (if strictly determinable)
    pts_sec: float                        # Presentation timestamp in seconds
    is_pts_exact: bool = True             # True if derived from container PTS packet
    image_bgr: np.ndarray = None          # Decoded frame (BGR)


@dataclass
class TemporalWindow:
    window_idx: int = 0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    fps: float = 25.0
    frames: List[FrameReference] = field(default_factory=list)
    window_id: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    target_frames: int = 16

    def __post_init__(self) -> None:
        if self.window_id is not None and self.window_idx == 0:
            self.window_idx = self.window_id
        elif self.window_id is None:
            self.window_id = self.window_idx

        if self.start_sec is not None and self.start_time_sec == 0.0:
            self.start_time_sec = self.start_sec
        elif self.start_sec is None:
            self.start_sec = self.start_time_sec

        if self.end_sec is not None and self.end_time_sec == 0.0:
            self.end_time_sec = self.end_sec
        elif self.end_sec is None:
            self.end_sec = self.end_time_sec


@dataclass
class FaceTrack:
    track_id: int
    frame_indices: List[int]              # Original frame indices or sample indices
    timestamps: List[float]               # Presentation timestamps (PTS)
    bounding_boxes: List[List[float]]     # [x1, y1, x2, y2] in pixels
    face_crops: List[np.ndarray]          # (112, 112) normalized crops
    detection_confidences: List[float]
    landmarks_3d: Optional[np.ndarray] = None  # (T, 468, 3) relative normalized coordinates
    validity_mask: Optional[np.ndarray] = None  # (T,) boolean mask: True if landmark is valid

    @property
    def confidences(self) -> List[float]:
        return self.detection_confidences


@dataclass
class TrackedFaceSequence:
    track_id: int
    face_crops: List[np.ndarray] = field(default_factory=list)
    detection_confidences: List[float] = field(default_factory=list)
    bounding_boxes: List[List[float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    @property
    def confidences(self) -> List[float]:
        return self.detection_confidences


@dataclass
class ForensicAnalysisConfig:
    target_face_size: int = 112
    min_track_length: int = 4
    pyramidal_alpha: float = 0.50
    confidence_threshold: float = 0.50
    use_platt_scaling: bool = True
    frames_per_window: int = 16
    window_duration_sec: float = 2.0
    max_windows: int = 4


@dataclass
class WindowForensicAnalysis:
    window_id: int
    start_sec: float
    end_sec: float
    frame_count: int
    detected_face: bool
    face_confidence: float
    raw_fake_probability: float
    calibrated_fake_probability: float
    spatial_saliency: np.ndarray
    kinematic_residual_energy: float
    frame_anomaly_scores: List[float]
    processing_time_ms: float
    diagnostic_notes: str = ""


@dataclass
class ForensicProvenanceAudit:
    model_checkpoint: str
    git_commit: str
    evaluation_timestamp: str
    runtime_device: str
    pyramidal_alpha: float
    calibrated_threshold: float
    calibration_method: str
    p50_latency_ms: float


@dataclass
class ForensicVideoReport:
    video_path: str
    verdict: str
    verdict_confidence: str
    video_fake_probability: float
    confidence_interval: Tuple[float, float]
    detected_manipulation_family: str
    temporal_consistency_score: float
    windows_analyzed: List[WindowForensicAnalysis]
    overall_spatial_saliency: np.ndarray
    temporal_anomaly_series: List[float]
    kinematic_residual_energy: float
    provenance: ForensicProvenanceAudit
    forensic_explanation: str = ""
    structured_telemetry: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
            return obj

        return _convert(self)


@dataclass
class VideoMetadata:
    container_duration_sec: float
    container_fps: float
    total_frames_reported: int
    frame_width: int
    frame_height: int
    video_codec: str
    bitrate_kbps: Optional[float]
    is_vfr: bool                          # True if variable frame rate detected
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None


@dataclass
class AnalysisContext:
    video_path: Path
    file_sha256: str
    file_size_bytes: int
    metadata: VideoMetadata

    # 1. Spatial Pathway: Sparse Representative Frames for Single-Frame Forensics & Grad-CAM
    spatial_frames: List[FrameReference]

    # 2. Temporal Pathway: Dense Sequential Windows for 3D Kinematics & M18
    temporal_windows: List[TemporalWindow]

    # Disentangled Tracked Faces across Temporal Windows
    face_tracks: List[FaceTrack] = field(default_factory=list)

    # Audio Track (Optional)
    audio_waveform: Optional[np.ndarray] = None

    # Model Outputs & Persistent Artifacts
    detector_outputs: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: Path = field(default_factory=lambda: Path("reports/artifacts"))
    warnings: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)

    @classmethod
    def build_from_video(
        cls,
        video_path: str | Path,
        spatial_sample_count: int = 16,
        temporal_window_length: int = 16,
        max_temporal_windows: int = 2,
        artifact_dir: Optional[Path] = None,
    ) -> AnalysisContext:
        p = Path(video_path)
        if not p.is_file():
            raise FileNotFoundError(f"Video file not found: {p}")

        validate_sample_count(spatial_sample_count)
        validate_sample_count(temporal_window_length)

        # 1. Chunked 64 KB SHA-256 Calculation (guards RAM)
        hasher = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        file_sha256 = hasher.hexdigest()

        # 2. Parse Container Metadata via FFprobe
        meta = cls.extract_metadata(p)

        # 3. Ingest Dual Pathways
        spatial_frames = cls._extract_spatial_pathway(p, meta, spatial_sample_count)
        temporal_windows = cls._extract_temporal_pathway(p, meta, temporal_window_length, max_temporal_windows)

        target_dir = artifact_dir or Path("reports/artifacts")
        target_dir.mkdir(parents=True, exist_ok=True)

        ctx = cls(
            video_path=p,
            file_sha256=file_sha256,
            file_size_bytes=p.stat().st_size,
            metadata=meta,
            spatial_frames=spatial_frames,
            temporal_windows=temporal_windows,
            artifact_dir=target_dir,
        )

        # Basic Quality Flags
        if meta.bitrate_kbps and meta.bitrate_kbps < 800.0:
            ctx.quality_flags.append("VIDEO_COMPRESSION")
        if meta.frame_width < 480 or meta.frame_height < 480:
            ctx.quality_flags.append("LOW_RESOLUTION")
        if meta.is_vfr:
            ctx.warnings.append("Variable frame rate (VFR) detected; temporal intervals measured via PTS.")

        return ctx

    @staticmethod
    def extract_metadata(video_path: Path) -> VideoMetadata:
        """Extracts container-level metadata using FFprobe with fallback to OpenCV."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return AnalysisContext._extract_metadata_opencv(video_path)

        try:
            cmd = [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration,bit_rate:stream=codec_name,codec_type,width,height,r_frame_rate,avg_frame_rate,sample_rate",
                "-of", "json", str(video_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)

            streams = data.get("streams", [])
            v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
            a_stream = audio_streams[0] if audio_streams else None

            r_fps_str = v_stream.get("r_frame_rate", "25/1")
            avg_fps_str = v_stream.get("avg_frame_rate", r_fps_str)
            try:
                r_num, r_den = map(float, r_fps_str.split("/"))
                r_fps = r_num / r_den if r_den > 0 else 25.0
            except Exception:
                r_fps = 25.0

            is_vfr = r_fps_str != avg_fps_str
            duration = float(data.get("format", {}).get("duration", 0.0))
            bitrate = data.get("format", {}).get("bit_rate")
            bitrate_kbps = float(bitrate) / 1000.0 if bitrate else None

            return VideoMetadata(
                container_duration_sec=duration,
                container_fps=r_fps,
                total_frames_reported=int(duration * r_fps) if duration > 0 else 0,
                frame_width=int(v_stream.get("width", 0)),
                frame_height=int(v_stream.get("height", 0)),
                video_codec=str(v_stream.get("codec_name", "unknown")),
                bitrate_kbps=bitrate_kbps,
                is_vfr=is_vfr,
                has_audio=len(audio_streams) > 0,
                audio_codec=str(a_stream.get("codec_name")) if a_stream else None,
                audio_sample_rate=int(a_stream.get("sample_rate")) if a_stream and a_stream.get("sample_rate") else None,
            )
        except Exception:
            return AnalysisContext._extract_metadata_opencv(video_path)

    @staticmethod
    def _extract_metadata_opencv(video_path: Path) -> VideoMetadata:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return VideoMetadata(
            container_duration_sec=total / fps if fps > 0 else 0.0,
            container_fps=fps,
            total_frames_reported=total,
            frame_width=w,
            frame_height=h,
            video_codec="unknown",
            bitrate_kbps=None,
            is_vfr=False,
            has_audio=False,
        )

    @classmethod
    def _extract_spatial_pathway(cls, path: Path, meta: VideoMetadata, count: int) -> List[FrameReference]:
        """Sparse uniform sampling across the entire video for spatial texture and Grad-CAM."""
        cap = cv2.VideoCapture(str(path))
        total = meta.total_frames_reported or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or count)
        if total <= 0:
            total = count

        step = max(1, total // count)
        target_indices = [min(i * step, total - 1) for i in range(count)]
        target_indices = sorted(list(set(target_indices)))

        frames: List[FrameReference] = []
        for s_idx, target_idx in enumerate(target_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            pts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            pts_sec = pts_msec / 1000.0 if pts_msec > 0 else (target_idx / meta.container_fps if meta.container_fps > 0 else 0.0)
            frames.append(FrameReference(
                sample_idx=s_idx,
                original_frame_idx=target_idx,
                pts_sec=pts_sec,
                is_pts_exact=pts_msec > 0 and not meta.is_vfr,
                image_bgr=frame,
            ))
        cap.release()
        return frames

    @classmethod
    def _extract_temporal_pathway(
        cls,
        path: Path,
        meta: VideoMetadata,
        window_len: int,
        max_windows: int,
    ) -> List[TemporalWindow]:
        """Sequential dense decoding of consecutive frames without seeking, preserving exact motion continuity."""
        cap = cv2.VideoCapture(str(path))
        windows: List[TemporalWindow] = []
        current_frames: List[FrameReference] = []
        frame_counter = 0

        while cap.isOpened() and len(windows) < max_windows:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            pts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            pts_sec = pts_msec / 1000.0 if pts_msec > 0 else (frame_counter / meta.container_fps if meta.container_fps > 0 else 0.0)

            ref = FrameReference(
                sample_idx=len(current_frames),
                original_frame_idx=frame_counter,
                pts_sec=pts_sec,
                is_pts_exact=pts_msec > 0,
                image_bgr=frame,
            )
            current_frames.append(ref)
            frame_counter += 1

            if len(current_frames) == window_len:
                windows.append(TemporalWindow(
                    window_idx=len(windows),
                    start_time_sec=current_frames[0].pts_sec,
                    end_time_sec=current_frames[-1].pts_sec,
                    fps=meta.container_fps,
                    frames=list(current_frames),
                ))
                current_frames.clear()

        cap.release()
        return windows
