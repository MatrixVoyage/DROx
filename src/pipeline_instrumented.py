"""Step 2: Full Pipeline Instrumentation.

Wraps the inference pipeline and emits structured frame-level and video-level execution traces
for failure attribution across all detection stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger("m16.pipeline")


class PipelineMode(str, Enum):
    M13_5_CONTROL = "M13_5_CONTROL"
    M16_EXPERIMENTAL = "M16_EXPERIMENTAL"


class FailureStage(str, Enum):
    FACE_DETECTION = "face_detection"
    FRAME_QUALITY = "frame_quality"
    FEATURE_EXTRACT = "feature_extract"
    ATTENTION = "attention"
    TEMPORAL_MODEL = "temporal_model"
    CLASSIFICATION = "classification"
    THRESHOLD = "threshold"


@dataclass
class FrameTrace:
    """Diagnostic trace captured for a single sampled frame."""

    frame_idx: int
    timestamp_ms: float
    face_detected: bool
    face_bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    face_confidence: Optional[float] = None
    face_area_ratio: Optional[float] = None  # face_area / frame_area
    head_pose_deg: Optional[List[float]] = None  # [yaw, pitch, roll]
    blur_score: float = 0.0  # Laplacian variance (sharpness)
    compression_score: float = 0.0  # Blocking artifact ratio
    resnext_logit: Optional[float] = None  # Per-frame pre-temporal score
    attention_entropy: Optional[float] = None  # Spatial attention diffusion
    attention_peak_region: Optional[str] = None  # "eyes", "mouth", "boundary", "uniform"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineTrace:
    """Comprehensive diagnostic trace for an end-to-end video inference run."""

    video_id: str
    video_path: str
    pipeline_mode: str  # M13_5_CONTROL or M16_EXPERIMENTAL
    total_frames: int
    sampled_frames: int
    faces_detected: int
    detection_rate: float  # faces_detected / sampled_frames
    frame_traces: List[FrameTrace] = field(default_factory=list)

    # Failure flags
    rejected_low_quality: bool = False
    rejection_reason: str = ""

    # Feature extraction
    feature_shape: Optional[List[int]] = None
    feature_norm_mean: Optional[float] = None
    feature_norm_std: Optional[float] = None

    # Temporal output
    lstm_hidden_norms: List[float] = field(default_factory=list)
    temporal_variance: Optional[float] = None  # Low = temporally dead/frozen

    # Multi-threshold verdicts in calibrated space (T=2.0502)
    raw_logit: Optional[float] = None
    calibrated_prob: Optional[float] = None
    verdict_at_050: Optional[str] = None  # Standard policy
    verdict_at_019: Optional[str] = None  # Youden-optimal policy
    verdict_at_013: Optional[str] = None  # High-sensitivity policy (spec >= 90%)

    # Branch outputs if multi-domain
    branch_scores: Optional[Dict[str, float]] = None

    # Timing and attribution
    inference_ms: Optional[float] = None
    failure_stage: Optional[str] = None  # Where the pipeline encountered a failure

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InstrumentedPipeline:
    """Wraps the deepfake detection pipeline and emits a detailed PipelineTrace for every analyzed video."""

    def __init__(
        self,
        predictor_or_checkpoint: Any = "checkpoints/m13_3_f2f_seed42/best.pth",
        mode: PipelineMode | str = PipelineMode.M13_5_CONTROL,
        device: Optional[str] = None,
        temperature: float = 2.0502,
        target_resolution: Optional[int] = None,
    ):
        """Initialize the instrumented pipeline with strict control/experimental separation.

        Args:
            predictor_or_checkpoint: VideoPredictor instance or path to checkpoint.
            mode: PipelineMode.M13_5_CONTROL (strictly 112x112, identical to M13.5 champion)
                  or PipelineMode.M16_EXPERIMENTAL.
            device: Target execution device.
            temperature: Calibration temperature for scaling logits.
            target_resolution: Crop resolution (strictly 112 for M13_5_CONTROL; configurable for M16_EXPERIMENTAL).
        """
        self.mode = PipelineMode(mode) if isinstance(mode, str) else mode
        self.T = float(temperature)

        if self.mode == PipelineMode.M13_5_CONTROL:
            # In control mode, enforce bit-for-bit M13.5 invariant: resolution is strictly 112
            self.target_resolution = 112
            self.margin_expansion = 0.20
        else:
            self.target_resolution = target_resolution or 224
            self.margin_expansion = 0.20

        if hasattr(predictor_or_checkpoint, "model") and hasattr(predictor_or_checkpoint, "pipeline"):
            self.predictor = predictor_or_checkpoint
        else:
            from src.inference.predictor import VideoPredictor

            self.predictor = VideoPredictor(
                checkpoint_path=predictor_or_checkpoint,
                device=device,
            )

        self.model = self.predictor.model
        self.device = self.predictor.device

    def analyze(self, video_path: str | Path) -> PipelineTrace:
        """Execute instrumented inference emitting full stage attribution."""
        t0 = time.perf_counter()
        vid = Path(video_path)
        if not vid.is_file():
            raise FileNotFoundError(f"Video file not found: {vid}")

        trace = PipelineTrace(
            video_id=vid.stem,
            video_path=str(vid.resolve()),
            pipeline_mode=self.mode.value,
            total_frames=0,
            sampled_frames=0,
            faces_detected=0,
            detection_rate=0.0,
        )

        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        trace.total_frames = max(0, total)

        if total <= 0:
            trace.rejected_low_quality = True
            trace.rejection_reason = "UNREADABLE_OR_EMPTY_VIDEO"
            trace.failure_stage = FailureStage.FRAME_QUALITY
            trace.inference_ms = (time.perf_counter() - t0) * 1000.0
            cap.release()
            return trace

        # Sample 16 frames uniformly (matching baseline pipeline behavior)
        num_samples = min(16, max(1, total))
        indices = np.linspace(0, total - 1, num_samples, dtype=int)
        trace.sampled_frames = len(indices)

        face_detector = self.predictor.pipeline.face_detector

        frames: List[np.ndarray] = []
        face_crops: List[np.ndarray] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            ft = FrameTrace(
                frame_idx=int(idx),
                timestamp_ms=float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                face_detected=False,
                face_bbox=None,
                face_confidence=None,
                face_area_ratio=None,
                head_pose_deg=None,
                blur_score=self._blur_score(frame),
                compression_score=self._blocking_score(frame),
            )

            # Detect face
            try:
                det_result = face_detector.detect(frame)
                if det_result and len(det_result) > 0:
                    primary = det_result[0]
                    bbox = [int(v) for v in primary.bbox]
                    conf = float(primary.confidence)

                    trace.faces_detected += 1
                    ft.face_detected = True
                    ft.face_bbox = bbox
                    ft.face_confidence = conf

                    h, w = frame.shape[:2]
                    bw = max(0, bbox[2] - bbox[0])
                    bh = max(0, bbox[3] - bbox[1])
                    ft.face_area_ratio = float((bw * bh) / max(1, w * h))
                    ft.head_pose_deg = self._estimate_head_pose(frame, bbox)

                    # Crop face
                    crop = self._crop_face(frame, bbox)
                    face_crops.append(crop)
            except Exception as e:
                logger.debug(f"Face detection failed on frame {idx}: {e}")

            trace.frame_traces.append(ft)
            frames.append(frame)

        cap.release()
        trace.detection_rate = float(trace.faces_detected / max(trace.sampled_frames, 1))

        # Check operational envelope (minimum 8 detected faces)
        if trace.faces_detected < 8:
            trace.rejected_low_quality = True
            trace.rejection_reason = f"LOW_QUALITY: only {trace.faces_detected}/{trace.sampled_frames} faces detected"
            trace.failure_stage = FailureStage.FACE_DETECTION
            trace.inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return trace

        # Prepare normalized tensor for model forward pass
        # Rescale face crops to standard 112x112 for M13.5 backbone
        crop_tensors = []
        for c in face_crops[:16]:
            resized = cv2.resize(c, (112, 112))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # ImageNet normalization
            norm = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            crop_tensors.append(torch.from_numpy(norm).permute(2, 0, 1))

        # Pad or slice to exactly 16 frames
        while len(crop_tensors) < 16:
            crop_tensors.append(crop_tensors[-1].clone() if crop_tensors else torch.zeros(3, 112, 112))
        tensor_seq = torch.stack(crop_tensors[:16], dim=0).unsqueeze(0).to(self.device, dtype=torch.float32)

        # Forward pass with internal state capture
        with torch.no_grad():
            if hasattr(self.model, "fusion_head"):
                out = self.model(tensor_seq, return_dict=True)
                raw_logit = float(out.primary_logits[0, 0].item())
                features = out.temporal_features[0]  # shape: (D,) or (T, D)
                trace.branch_scores = {
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                    "p_composite": round(float(out.p_composite[0, 0].item()), 4),
                    "p_freq": round(float(out.p_freq[0, 0].item()), 4),
                    "p_texture": round(float(out.p_texture[0, 0].item()), 4),
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                }
            else:
                out_logits, features = self.model(tensor_seq, return_embeddings=True)
                raw_logit = float(out_logits[0, 0].item())

        trace.raw_logit = round(raw_logit, 4)
        scaled_logit = raw_logit / self.T
        cal_p = float(1.0 / (1.0 + np.exp(-scaled_logit)))
        trace.calibrated_prob = round(cal_p, 4)

        # Multi-threshold verdicts
        trace.verdict_at_050 = "DEEPFAKE" if cal_p >= 0.5000 else "REAL"
        trace.verdict_at_019 = "DEEPFAKE" if cal_p >= 0.1912 else "REAL"
        trace.verdict_at_013 = "DEEPFAKE" if cal_p >= 0.1260 else "REAL"

        # Feature representations
        if isinstance(features, torch.Tensor):
            feat_cpu = features.cpu()
            trace.feature_shape = list(feat_cpu.shape)
            trace.feature_norm_mean = round(float(feat_cpu.norm(dim=-1).mean().item()), 4)
            trace.feature_norm_std = round(float(feat_cpu.norm(dim=-1).std().item()), 4)
            if feat_cpu.dim() >= 2:
                trace.temporal_variance = round(float(feat_cpu.var(dim=0).mean().item()), 6)
            else:
                trace.temporal_variance = 0.0

        trace.inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return trace

    @staticmethod
    def _blur_score(frame: np.ndarray) -> float:
        """Laplacian variance sharpness score — low indicates motion blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _blocking_score(frame: np.ndarray) -> float:
        """JPEG/video 8x8 blocking artifact metric — high indicates heavy compression."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h_diff = np.abs(np.diff(gray[:, 7::8], axis=1))
        h_interior = np.abs(np.diff(gray[:, ::1], axis=1))
        mean_b = float(h_diff.mean()) if h_diff.size > 0 else 0.0
        mean_i = float(h_interior.mean()) if h_interior.size > 0 else 0.0
        return float(mean_b / (mean_i + 1e-6))

    @staticmethod
    def _estimate_head_pose(frame: np.ndarray, bbox: List[int]) -> List[float]:
        """Estimate coarse head pose [yaw, pitch, roll] in degrees from face bounding box aspect and center."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = bh / bw
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        yaw = float(((cx / w) - 0.5) * 60.0)
        pitch = float(((cy / h) - 0.5) * 45.0 + (aspect - 1.25) * 20.0)
        roll = 0.0
        return [round(yaw, 1), round(pitch, 1), round(roll, 1)]

    def _crop_face(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Safely crop and expand face bounding box."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        # 15% margin expansion
        bw = x2 - x1
        bh = y2 - y1
        mx = int(bw * 0.15)
        my = int(bh * 0.15)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = np.zeros((self.target_resolution, self.target_resolution, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.target_resolution, self.target_resolution))
