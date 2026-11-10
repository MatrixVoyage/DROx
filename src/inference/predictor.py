"""Inference engine for end-to-end deepfake video classification.

Provides standalone, read-only inference for both raw video files and preprocessed tensors.
Decoupled completely from training logic, dataset manifests, and ONNX dependencies.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

import json
from src.data.pipeline import VideoFacePipeline
from src.inference.uncertainty import (
    CalibratedDecisionPolicy,
    MCDropoutEstimator,
    OODFeatureEstimator,
    QualityAssessor,
    QualitySignals,
    UncertaintySignals,
)
from src.models.detector import DeepfakeDetector
from src.utils.config import AppConfig, _dict_to_config, load_config
from src.utils.logger import setup_logger


@dataclass
class PredictionResult:
    """Strongly typed container for video inference predictions."""

    video_path: Optional[str]
    verdict: str  # "Deepfake", "Real", "Reject", or "Error"
    probability: float  # Sigmoid probability of deepfake in [0.0, 1.0]
    raw_logit: float  # Unbounded model scalar output
    confidence: float  # Decision confidence max(p, 1 - p) in [0.5, 1.0] or 0.0 if Reject
    threshold: float  # Active decision threshold
    frames_analyzed: int  # Number of sampled frames
    faces_detected: int  # Number of frames with detected face crops
    device: str  # Evaluation device ("cuda" or "cpu")
    model_version: str  # Checkpoint filename or model identifier
    latency_ms: Dict[str, float]  # Stage-by-stage latency in milliseconds
    error: Optional[str] = None  # None if successful, or error code
    metadata: Optional[Dict[str, Any]] = None  # Diagnostic metadata
    # Calibrated 3-state policy fields (M12.4):
    decision: Optional[str] = None  # "REAL", "FAKE", "REJECT"
    rejection_code: Optional[str] = None  # e.g. "LOW_QUALITY_AND_HIGH_UNCERTAINTY"
    rejection_reason: Optional[str] = None  # Mutually understandable explanation
    uncertainty: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None
    calibration_info: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize prediction result to standard Python dictionary."""
        return asdict(self)


class VideoPredictor:
    """Standalone video deepfake predictor engine.

    Operates in torch.inference_mode() with zero gradient computation or training dependencies.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: Optional[Union[AppConfig, str, Path]] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        allow_demo_fallback: bool = True,
    ) -> None:
        """Initialize predictor by loading checkpoint and configuring pipeline.

        Args:
            checkpoint_path: Path to trained PyTorch checkpoint (.pth).
            config: Optional AppConfig or path to YAML config. If None, restored from checkpoint.
            device: Target device ("cuda" or "cpu"). If None, resolves from config/hardware.
            threshold: Binary decision threshold (default: 0.5).
            allow_demo_fallback: If True, falls back to initialized model weights if checkpoint is missing.
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else Path("checkpoints/default/best.pth")
        self.threshold = float(threshold)
        self.logger = setup_logger("video_predictor")
        self.is_demo_mode = False

        if not self.checkpoint_path.is_file():
            if not allow_demo_fallback:
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
            self.logger.warning(
                f"Checkpoint file not found at '{self.checkpoint_path}'. "
                "Operating in DEMO / SIMULATION mode with initialized architecture. "
                "Run 'python scripts/download_weights.py' to download trained production checkpoints."
            )
            self.is_demo_mode = True
            if config is not None:
                self.config = load_config(config) if isinstance(config, (str, Path)) else config
            else:
                self.config = load_config("configs/default.yaml")

            if device:
                self.config.device = device
            effective_device_str = self.config.get_effective_device()
            self.device = torch.device(effective_device_str)

            self.model = DeepfakeDetector(self.config.model)
            self.model.to(self.device)
            self.model.eval()

            self.pipeline = VideoFacePipeline(config=self.config)
            self.model_version = f"demo-mode-unweighted ({self.checkpoint_path.name})"
            self._init_calibration()
            self.logger.info(
                f"Initialized VideoPredictor in DEMO mode on {self.device} (threshold={self.threshold})"
            )
            return

        # Load checkpoint payload read-only
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Resolve AppConfig
        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
        is_m18 = any(k.startswith("kin_branch.") or k.startswith("kinematic_branch.") for k in state_dict.keys()) or "m18" in str(self.checkpoint_path).lower()
        is_m17 = ((any(k.startswith("res_branch.") or k.startswith("res_stream.") or k.startswith("srm_layer.") for k in state_dict.keys()) or "m17" in str(self.checkpoint_path).lower()) and not is_m18)
        is_family_invariant = any(
            k.startswith("fusion_head.") or k.startswith("freq_branch.") or k.startswith("texture_branch.")
            for k in state_dict.keys()
        )
        is_multi_head = any(k.startswith("aux_head.") or k.startswith("mid_head.") for k in state_dict.keys())

        if config is not None:
            if isinstance(config, (str, Path)):
                self.config = load_config(config)
            else:
                self.config = config
        elif "config" in checkpoint_data and checkpoint_data["config"] is not None and (
            isinstance(checkpoint_data["config"], AppConfig) or (isinstance(checkpoint_data["config"], dict) and "model" in checkpoint_data["config"])
        ):
            raw_cfg = checkpoint_data["config"]
            self.config = _dict_to_config(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
        elif is_family_invariant or is_m17 or is_m18:
            self.config = load_config("configs/experiments/m12_full.yaml")
        else:
            self.config = load_config("configs/default.yaml")

        # Resolve execution device
        if device:
            self.config.device = device
        effective_device_str = self.config.get_effective_device()
        self.device = torch.device(effective_device_str)

        # Build model based on checkpoint state_dict
        if is_m18:
            from src.models.m17_detector import M17TrackADetector
            from src.models.m18_detector import M18ReenactmentDetector
            base_det = DeepfakeDetector(self.config.model)
            trunk = M17TrackADetector(base_detector=base_det, num_families=5, res_dim=128, aux_hidden_dim=512)
            self.model = M18ReenactmentDetector(trunk_detector=trunk, kin_dim=256, aux_hidden_dim=512, freeze_trunk=True)
        elif is_m17:
            from src.models.m17_detector import M17TrackADetector
            base_det = DeepfakeDetector(self.config.model)
            self.model = M17TrackADetector(base_detector=base_det, num_families=5, res_dim=128, aux_hidden_dim=512)
        elif is_family_invariant:
            from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector
            has_mid = any(k.startswith("mid_head.") for k in state_dict.keys())
            base_det = DeepfakeDetector(self.config.model)
            self.model = FamilyInvariantDeepfakeDetector(
                base_detector=base_det,
                num_families=5,
                enable_mid_head=has_mid,
                mid_in_channels=112,
                mid_hidden_dim=128,
            )
        elif is_multi_head:
            from src.models.multi_head_detector import MultiHeadDeepfakeDetector
            has_mid = any(k.startswith("mid_head.") for k in state_dict.keys())
            base_det = DeepfakeDetector(self.config.model)
            self.model = MultiHeadDeepfakeDetector(
                base_detector=base_det,
                num_families=5,
                enable_mid_head=has_mid,
                mid_in_channels=112,
                mid_hidden_dim=128,
            )
        else:
            self.model = DeepfakeDetector(self.config.model)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Build video preprocessing pipeline
        self.pipeline = VideoFacePipeline(config=self.config)

        self.model_version = self.checkpoint_path.name
        self._init_calibration()
        self.logger.info(
            f"Initialized VideoPredictor with checkpoint '{self.model_version}' on {self.device} (threshold={self.threshold})"
        )

    def _init_calibration(self) -> None:
        """Initialize calibration, uncertainty estimation, and quality assessment components."""
        ckpt_dir = self.checkpoint_path.parent
        calib_cfg_path = ckpt_dir / "calibration_config.json"
        ood_ref_path = ckpt_dir / "ood_reference.json"

        cls_thresh = self.threshold
        max_var = 0.04
        ood_thresh = 3.0
        calib_ver = "uncalibrated"
        calib_dataset = "none"

        temperature = 1.0

        reject_on_ood = True
        if calib_cfg_path.is_file():
            try:
                with open(calib_cfg_path, "r") as f:
                    c_data = json.load(f)
                cls_thresh = c_data.get("classification_threshold", cls_thresh)
                max_var = c_data.get("max_uncertainty_variance", max_var)
                ood_thresh = c_data.get("ood_distance_threshold", ood_thresh)
                temperature = float(c_data.get("temperature", 1.0))
                calib_ver = c_data.get("calibration_version", "m12_full_validation")
                calib_dataset = c_data.get("calibration_dataset", "M12 Full Validation Split")
                reject_on_ood = c_data.get("reject_on_ood", True)
            except Exception as e:
                self.logger.warning(f"Could not load calibration config: {e}")

        if "m13" in str(self.checkpoint_path).lower():
            # Milestone 13.5 promotion gate: OOD is diagnostic only unless separately validated
            reject_on_ood = False

        self.decision_policy = CalibratedDecisionPolicy(
            classification_threshold=cls_thresh,
            max_uncertainty_variance=max_var,
            ood_distance_threshold=ood_thresh,
            reject_on_ood=reject_on_ood,
        )
        self.mc_estimator = MCDropoutEstimator(model=self.model, num_samples=10, temperature=temperature)

        # Feature dim: EfficientNet-B0 penultimate feature dim is 1280, ResNeXt is 2048
        backbone_name = getattr(getattr(self.config, "model", None), "backbone", None)
        feat_dim = 1280 if backbone_name and "efficientnet" in getattr(backbone_name, "name", "").lower() else 1280
        self.ood_estimator = OODFeatureEstimator(feature_dim=feat_dim)
        if ood_ref_path.is_file():
            try:
                self.ood_estimator.load(ood_ref_path)
            except Exception as e:
                self.logger.warning(f"Could not load OOD reference: {e}")

        self.quality_assessor = QualityAssessor(max_fallback_ratio=0.50, min_detected_frames=8)

        self.calibration_metadata = {
            "version": calib_ver,
            "dataset": calib_dataset,
            "temperature": temperature,
            "thresholds": {
                "classification_threshold": cls_thresh,
                "max_uncertainty_variance": max_var,
                "ood_distance_threshold": ood_thresh,
                "mc_samples": 10,
            },
        }

    def _get_model_info_dict(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        p_str = str(self.checkpoint_path).lower()
        if "m18" in p_str:
            name = "m18-dual-stream"
            version = "m18"
        elif "m17" in p_str:
            name = "m17-sensor-residuals"
            version = "m17"
        elif "m13_3" in p_str or "m13_4" in p_str:
            name = "v5-family-invariant"
            version = "m13.4"
        elif "m13" in p_str:
            name = "v4-generalization"
            version = "m13.2b"
        elif "m12" in p_str:
            name = "v3-robust"
            version = "m12"
        elif "m10" in p_str:
            name = "v2-evolved"
            version = "m10"
        else:
            name = "v1-baseline"
            version = "v1"

        backbone_name = getattr(getattr(self.config, "model", None), "backbone", None)
        b_str = getattr(backbone_name, "name", "unknown") if backbone_name else "unknown"
        temporal_cfg = getattr(getattr(self.config, "model", None), "temporal", None)
        t_str = getattr(temporal_cfg, "type", "unknown") if temporal_cfg else "unknown"

        arch_str = f"{b_str} + {t_str}"
        if name == "v5-family-invariant":
            arch_str += " + LocalFreq + LocalTexture + MultiDomainFusion"
        elif name == "v4-generalization":
            arch_str += " + Stage5MidHook + AuxFamilyHead"
        elif name == "m18-dual-stream":
            arch_str = "M17TrackA Trunk + 3D Differential Kinematics + Dynamic Gating"
        elif name == "m17-sensor-residuals":
            arch_str = "EfficientNet-B0 + SRM + Bayar Multi-Scale Residuals"

        res_dict = {
            "name": name,
            "version": version,
            "architecture": arch_str,
            "device": str(self.device),
        }
        if getattr(self, "is_demo_mode", False):
            res_dict["demo_mode"] = True
            res_dict["demo_notice"] = (
                "Running with initialized backbone weights. Official weights can be downloaded via "
                "'python scripts/download_weights.py'."
            )
        return res_dict

    def predict_tensor(
        self,
        tensor: torch.Tensor,
        threshold: Optional[float] = None,
        video_id: Optional[str] = None,
        calibrated: bool = False,
        detected_frames: Optional[int] = None,
    ) -> PredictionResult:
        """Run inference on an already preprocessed face sequence tensor of shape (T, 3, H, W) or (1, T, 3, H, W)."""
        active_threshold = threshold if threshold is not None else self.threshold

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)

        if tensor.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B, T, C, H, W), got shape {tuple(tensor.shape)}")

        b, t, c, h, w = tensor.shape
        det_f = detected_frames if detected_frames is not None else t
        model_cfg = getattr(self.config, "model", None)
        backbone_cfg = getattr(model_cfg, "backbone", None) if model_cfg else None
        exp_h, exp_w = getattr(backbone_cfg, "input_size", [112, 112])
        if c != 3 or h != exp_h or w != exp_w:
            raise ValueError(f"Expected tensor with spatial dims (3, {exp_h}, {exp_w}), got ({c}, {h}, {w})")

        if not torch.isfinite(tensor).all():
            raise ValueError("Input tensor contains NaN or Inf values.")

        t0 = time.perf_counter()

        if calibrated:
            # Deterministic forward pass for unscaled logit and diagnostic branch scores
            tensor_dev = tensor.to(self.device, dtype=torch.float32)
            clean_raw_logit = 0.0
            clean_raw_prob = 0.0
            branch_scores = None
            with torch.inference_mode():
                if hasattr(self.model, "fusion_head"):
                    out = self.model(tensor_dev, return_dict=True)
                    clean_raw_logit = float(out.primary_logits[0, 0].item())
                    clean_raw_prob = float(torch.sigmoid(out.primary_logits)[0, 0].item())
                    branch_scores = {
                        "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                        "p_composite": round(float(out.p_composite[0, 0].item()), 4),
                        "p_freq": round(float(out.p_freq[0, 0].item()), 4),
                        "p_texture": round(float(out.p_texture[0, 0].item()), 4),
                        "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                    }
                elif hasattr(self.model, "aux_head"):
                    out = self.model(tensor_dev, return_dict=True)
                    clean_raw_logit = float(out.primary_logits[0, 0].item())
                    clean_raw_prob = float(torch.sigmoid(out.primary_logits)[0, 0].item())
                    branch_scores = {
                        "p_primary": round(float(out.p_primary[0, 0].item()), 4),
                        "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                        "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                    }
                else:
                    logits = self.model(tensor_dev)
                    clean_raw_logit = float(logits[0, 0].item())
                    clean_raw_prob = float(torch.sigmoid(logits)[0, 0].item())

            quality_signals = self.quality_assessor.assess(
                total_frames=t,
                detected_frames=det_f,
                previous_bbox_frames=0,
                center_crop_frames=0,
                failed_frames=0,
            )
            mean_prob, var_p, entropy, mean_features = self.mc_estimator.estimate(tensor, device=self.device)
            mc_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            t_ood = time.perf_counter()
            ood_dist = self.ood_estimator.compute_distance(mean_features)
            is_ood = self.ood_estimator.is_ood(ood_dist)
            ood_time_ms = round((time.perf_counter() - t_ood) * 1000.0, 2)

            unc_signals = UncertaintySignals(
                mean_probability=mean_prob,
                epistemic_variance=var_p,
                entropy=entropy,
                mc_samples=10,
                ood_distance=ood_dist,
                is_ood=is_ood,
                is_uncertain=var_p > self.decision_policy.max_uncertainty_variance,
            )

            verdict, confidence, code, reason = self.decision_policy.decide(
                mean_prob, quality_signals, unc_signals
            )
            decision = "REJECT" if verdict == "Reject" else ("FAKE" if verdict == "Deepfake" else "REAL")
            total_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            diagnostics = {
                "ood_status": "Within validated range" if not is_ood else f"Elevated feature distance ({ood_dist:.2f} > {self.decision_policy.ood_distance_threshold}) [Diagnostic only]",
                "uncertainty_status": "Low epistemic uncertainty" if not unc_signals.is_uncertain else f"Elevated variance ({var_p:.4f} > {self.decision_policy.max_uncertainty_variance})",
                "quality_status": "Acceptable face detection" if quality_signals.is_acceptable else f"Quality issue: {quality_signals.rejection_reason}",
            }
            if branch_scores:
                diagnostics["branch_scores"] = branch_scores

            pred_metadata: Dict[str, Any] = {
                "tensor_input": True,
                "raw_probability": round(clean_raw_prob, 4),
                "calibrated_probability": round(mean_prob, 4),
            }
            if branch_scores:
                pred_metadata["branch_scores"] = branch_scores

            return PredictionResult(
                video_path=video_id,
                verdict=verdict,
                probability=round(mean_prob, 4),
                raw_logit=round(clean_raw_logit, 4),
                confidence=round(confidence, 4),
                threshold=self.decision_policy.classification_threshold,
                frames_analyzed=t,
                faces_detected=t,
                device=str(self.device),
                model_version=self.model_version,
                latency_ms={"mc_dropout_ms": mc_time_ms, "ood_ms": ood_time_ms, "total_ms": total_time_ms},
                error=None,
                metadata=pred_metadata,
                decision=decision,
                rejection_code=code,
                rejection_reason=reason,
                uncertainty=asdict(unc_signals),
                quality=asdict(quality_signals),
                model_info=self._get_model_info_dict(),
                calibration_info=self.calibration_metadata,
                diagnostics=diagnostics,
            )

        tensor = tensor.to(self.device, dtype=torch.float32)
        branch_scores = None
        with torch.inference_mode():
            if hasattr(self.model, "kinematic_branch"):
                out = self.model(tensor, mode="dual_stream")
                raw_logit = float(out.primary_logits[0, 0].item())
                prob = float(out.p_fused[0, 0].item())
                branch_scores = {
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                    "p_kin": round(float(out.p_kin[0, 0].item()), 4),
                    "gate_values": round(float(out.gate_values[0, 0].item()), 4) if hasattr(out, "gate_values") else 0.5,
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                }
            elif hasattr(self.model, "res_branch") or hasattr(self.model, "srm_layer"):
                out = self.model(tensor)
                raw_logit = float(out.primary_logits[0, 0].item())
                prob = float(out.p_fused[0, 0].item())
                branch_scores = {
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                    "p_res": round(float(out.p_res[0, 0].item()), 4),
                    "p_composite": round(float(out.p_composite[0, 0].item()), 4),
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                }
            elif hasattr(self.model, "fusion_head"):
                out = self.model(tensor, return_dict=True)
                raw_logit = float(out.primary_logits[0, 0].item())
                prob = float(out.p_fused[0, 0].item())
                branch_scores = {
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                    "p_composite": round(float(out.p_composite[0, 0].item()), 4),
                    "p_freq": round(float(out.p_freq[0, 0].item()), 4),
                    "p_texture": round(float(out.p_texture[0, 0].item()), 4),
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                }
            elif hasattr(self.model, "aux_head"):
                out = self.model(tensor, return_dict=True)
                raw_logit = float(out.primary_logits[0, 0].item())
                prob = float(out.p_fused[0, 0].item())
                branch_scores = {
                    "p_primary": round(float(out.p_primary[0, 0].item()), 4),
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                }
            else:
                logits = self.model(tensor)
                raw_logit = float(logits[0, 0].item())
                prob = float(torch.sigmoid(logits)[0, 0].item())

        model_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        verdict = "Deepfake" if prob >= active_threshold else "Real"
        confidence = float(prob if verdict == "Deepfake" else 1.0 - prob)

        pred_metadata: Dict[str, Any] = {"tensor_input": True}
        if branch_scores:
            pred_metadata["branch_scores"] = branch_scores

        return PredictionResult(
            video_path=video_id,
            verdict=verdict,
            probability=round(prob, 4),
            raw_logit=round(raw_logit, 4),
            confidence=round(confidence, 4),
            threshold=active_threshold,
            frames_analyzed=t,
            faces_detected=t,
            device=str(self.device),
            model_version=self.model_version,
            latency_ms={"model_forward_ms": model_time_ms, "total_ms": model_time_ms},
            error=None,
            metadata=pred_metadata,
            decision="FAKE" if verdict == "Deepfake" else "REAL",
            model_info=self._get_model_info_dict(),
        )

    def predict_video(
        self,
        video_path: Union[str, Path],
        threshold: Optional[float] = None,
        video_id: Optional[str] = None,
        calibrated: bool = False,
    ) -> PredictionResult:
        """Run full end-to-end inference on a raw video file.

        Executes:
        Video sampling -> MTCNN face detection -> Cropping & normalization -> Model forward pass -> Verdict.

        Args:
            video_path: Absolute or relative path to video file (.mp4, .avi, .mov).
            threshold: Optional threshold override.
            video_id: Optional stable video identifier for deterministic frame sampling.
            calibrated: If True, executes calibrated 3-state decision policy (MC-Dropout + OOD + Quality).

        Returns:
            Populated PredictionResult instance.
        """
        video_p = Path(video_path)
        active_threshold = threshold if threshold is not None else self.threshold

        if not video_p.is_file():
            return PredictionResult(
                video_path=str(video_p),
                verdict="Error",
                probability=0.0,
                raw_logit=0.0,
                confidence=0.0,
                threshold=active_threshold,
                frames_analyzed=0,
                faces_detected=0,
                device=str(self.device),
                model_version=self.model_version,
                latency_ms={"total_ms": 0.0},
                error=f"FILE_NOT_FOUND: {video_p} does not exist.",
            )

        t_total_start = time.perf_counter()

        # Run preprocessing pipeline
        t_prep_start = time.perf_counter()
        prep_result = self.pipeline.process_video(video_p, video_id=video_id)
        prep_time_ms = round((time.perf_counter() - t_prep_start) * 1000.0, 2)

        total_frames = len(prep_result.frame_indices) if prep_result.frame_indices else 16
        detected_frames = prep_result.detected_frames
        counters = getattr(prep_result, "detection_counters", {}) or {}
        quality_signals = self.quality_assessor.assess(
            total_frames=total_frames,
            detected_frames=detected_frames,
            previous_bbox_frames=counters.get("previous_bbox", 0),
            center_crop_frames=counters.get("center_crop", 0),
            failed_frames=counters.get("failed", 0),
        )

        if prep_result.status != "success" or prep_result.tensor is None:
            total_time_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)
            err_code = prep_result.error_code.value if prep_result.error_code else prep_result.status
            if calibrated:
                return PredictionResult(
                    video_path=str(video_p.resolve()),
                    verdict="Reject",
                    probability=0.0,
                    raw_logit=0.0,
                    confidence=0.0,
                    threshold=active_threshold,
                    frames_analyzed=total_frames,
                    faces_detected=detected_frames,
                    device=str(self.device),
                    model_version=self.model_version,
                    latency_ms={"preprocessing_ms": prep_time_ms, "total_ms": total_time_ms},
                    error=None,
                    metadata=prep_result.to_metadata_dict(),
                    decision="REJECT",
                    rejection_code="LOW_QUALITY",
                    rejection_reason=f"Low Video Quality: {quality_signals.rejection_reason or prep_result.error_message}",
                    quality=asdict(quality_signals),
                    uncertainty=None,
                    model_info=self._get_model_info_dict(),
                    calibration_info=self.calibration_metadata,
                    diagnostics={
                        "quality_status": quality_signals.rejection_reason or "Extraction failed",
                        "ood_status": "Not evaluated",
                        "uncertainty_status": "Not evaluated",
                    },
                )
            return PredictionResult(
                video_path=str(video_p),
                verdict="Error",
                probability=0.0,
                raw_logit=0.0,
                confidence=0.0,
                threshold=active_threshold,
                frames_analyzed=total_frames,
                faces_detected=detected_frames,
                device=str(self.device),
                model_version=self.model_version,
                latency_ms={"preprocessing_ms": prep_time_ms, "total_ms": total_time_ms},
                error=f"{err_code}: {prep_result.error_message}",
                metadata=prep_result.to_metadata_dict(),
            )

        # Tensor shape (T, 3, 112, 112) -> forward pass
        tensor = prep_result.tensor.unsqueeze(0).to(self.device, dtype=torch.float32)

        if calibrated:
            # Deterministic forward pass for unscaled logit and diagnostic branch scores
            tensor_dev = tensor.to(self.device, dtype=torch.float32)
            clean_raw_logit = 0.0
            clean_raw_prob = 0.0
            branch_scores = None
            with torch.inference_mode():
                if hasattr(self.model, "kinematic_branch"):
                    out = self.model(tensor_dev, mode="dual_stream")
                    clean_raw_logit = float(out.primary_logits[0, 0].item())
                    clean_raw_prob = float(out.p_fused[0, 0].item())
                    branch_scores = {
                        "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                        "p_kin": round(float(out.p_kin[0, 0].item()), 4),
                        "gate_values": round(float(out.gate_values[0, 0].item()), 4) if hasattr(out, "gate_values") else 0.5,
                        "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                    }
                elif hasattr(self.model, "residual_stream") or hasattr(self.model, "res_branch") or hasattr(self.model, "srm_layer"):
                    out = self.model(tensor_dev)
                    clean_raw_logit = float(out.primary_logits[0, 0].item())
                    clean_raw_prob = float(out.p_fused[0, 0].item())
                    branch_scores = {
                        "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                        "p_res": round(float(getattr(out, "p_res", out.p_fused)[0, 0].item()), 4),
                        "p_composite": round(float(getattr(out, "p_composite", out.p_fused)[0, 0].item()), 4),
                        "p_aux_fake": round(float(getattr(out, "p_aux_fake", out.p_fused)[0, 0].item()), 4),
                    }
                elif hasattr(self.model, "dynamics_branch"):
                    out = self.model(tensor_dev)
                    clean_raw_logit = float(out.primary_logits[0, 0].item())
                    clean_raw_prob = float(out.p_fused[0, 0].item())
                    branch_scores = {
                        "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                        "p_dyn": round(float(getattr(out, "p_dyn", out.p_fused)[0, 0].item()), 4),
                        "p_composite": round(float(getattr(out, "p_composite", out.p_fused)[0, 0].item()), 4),
                        "p_aux_fake": round(float(getattr(out, "p_aux_fake", out.p_fused)[0, 0].item()), 4),
                    }
                elif hasattr(self.model, "fusion_head"):
                    try:
                        out = self.model(tensor_dev, return_dict=True)
                    except TypeError:
                        out = self.model(tensor_dev)
                    clean_raw_logit = float(out.primary_logits[0, 0].item())
                    clean_raw_prob = float(torch.sigmoid(out.primary_logits)[0, 0].item())
                    branch_scores = {
                        "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                        "p_composite": round(float(getattr(out, "p_composite", out.p_fused)[0, 0].item()), 4),
                        "p_freq": round(float(getattr(out, "p_freq", out.p_fused)[0, 0].item()), 4),
                        "p_texture": round(float(getattr(out, "p_texture", out.p_fused)[0, 0].item()), 4),
                        "p_aux_fake": round(float(getattr(out, "p_aux_fake", out.p_fused)[0, 0].item()), 4),
                    }
                elif hasattr(self.model, "aux_head"):
                    try:
                        out = self.model(tensor_dev, return_dict=True)
                    except TypeError:
                        out = self.model(tensor_dev)
                    clean_raw_logit = float(out.primary_logits[0, 0].item())
                    clean_raw_prob = float(torch.sigmoid(out.primary_logits)[0, 0].item())
                    branch_scores = {
                        "p_primary": round(float(getattr(out, "p_primary", getattr(out, "p_fused", torch.sigmoid(out.primary_logits)))[0, 0].item()), 4),
                        "p_aux_fake": round(float(getattr(out, "p_aux_fake", torch.zeros(1, 1))[0, 0].item()), 4),
                        "p_fused": round(float(getattr(out, "p_fused", torch.sigmoid(out.primary_logits))[0, 0].item()), 4),
                    }
                else:
                    logits = self.model(tensor_dev)
                    clean_raw_logit = float(logits[0, 0].item())
                    clean_raw_prob = float(torch.sigmoid(logits)[0, 0].item())

            t_mc_start = time.perf_counter()
            mean_prob, var_p, entropy, mean_features = self.mc_estimator.estimate(tensor, device=self.device)
            mc_time_ms = round((time.perf_counter() - t_mc_start) * 1000.0, 2)

            t_ood_start = time.perf_counter()
            ood_dist = self.ood_estimator.compute_distance(mean_features)
            is_ood = self.ood_estimator.is_ood(ood_dist)
            ood_time_ms = round((time.perf_counter() - t_ood_start) * 1000.0, 2)

            unc_signals = UncertaintySignals(
                mean_probability=mean_prob,
                epistemic_variance=var_p,
                entropy=entropy,
                mc_samples=10,
                ood_distance=ood_dist,
                is_ood=is_ood,
                is_uncertain=var_p > self.decision_policy.max_uncertainty_variance,
            )

            verdict, confidence, code, reason = self.decision_policy.decide(
                mean_prob, quality_signals, unc_signals
            )
            decision = "REJECT" if verdict == "Reject" else ("FAKE" if verdict == "Deepfake" else "REAL")
            total_time_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)

            stage_latencies = {
                "preprocessing_ms": prep_time_ms,
                "mc_dropout_inference_ms": mc_time_ms,
                "ood_computation_ms": ood_time_ms,
                "total_ms": total_time_ms,
            }

            diagnostics = {
                "ood_status": "Within validated range" if not is_ood else f"Elevated feature distance ({ood_dist:.2f} > {self.decision_policy.ood_distance_threshold}) [Diagnostic only]",
                "uncertainty_status": "Low epistemic uncertainty" if not unc_signals.is_uncertain else f"Elevated variance ({var_p:.4f} > {self.decision_policy.max_uncertainty_variance})",
                "quality_status": "Acceptable face detection" if quality_signals.is_acceptable else f"Quality issue: {quality_signals.rejection_reason}",
            }
            if branch_scores:
                diagnostics["branch_scores"] = branch_scores

            meta = prep_result.to_metadata_dict()
            meta["raw_probability"] = round(clean_raw_prob, 4)
            meta["calibrated_probability"] = round(mean_prob, 4)
            if branch_scores:
                meta["branch_scores"] = branch_scores

            return PredictionResult(
                video_path=str(video_p.resolve()),
                verdict=verdict,
                probability=round(mean_prob, 4),
                raw_logit=round(clean_raw_logit, 4),
                confidence=round(confidence, 4),
                threshold=self.decision_policy.classification_threshold,
                frames_analyzed=total_frames,
                faces_detected=detected_frames,
                device=str(self.device),
                model_version=self.model_version,
                latency_ms=stage_latencies,
                error=None,
                metadata=meta,
                decision=decision,
                rejection_code=code,
                rejection_reason=reason,
                uncertainty=asdict(unc_signals),
                quality=asdict(quality_signals),
                model_info=self._get_model_info_dict(),
                calibration_info=self.calibration_metadata,
                diagnostics=diagnostics,
            )

        t_model_start = time.perf_counter()
        branch_scores = None
        with torch.inference_mode():
            if hasattr(self.model, "kinematic_branch"):
                out = self.model(tensor, mode="dual_stream")
                raw_logit = float(out.primary_logits[0, 0].item())
                prob = float(out.p_fused[0, 0].item())
                branch_scores = {
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                    "p_kin": round(float(out.p_kin[0, 0].item()), 4),
                    "gate_values": round(float(out.gate_values[0, 0].item()), 4) if hasattr(out, "gate_values") else 0.5,
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                }
            elif hasattr(self.model, "res_branch") or hasattr(self.model, "srm_layer"):
                out = self.model(tensor)
                raw_logit = float(out.primary_logits[0, 0].item())
                prob = float(out.p_fused[0, 0].item())
                branch_scores = {
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                    "p_res": round(float(out.p_res[0, 0].item()), 4),
                    "p_composite": round(float(out.p_composite[0, 0].item()), 4),
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                }
            elif hasattr(self.model, "fusion_head"):
                out = self.model(tensor, return_dict=True)
                raw_logit = float(out.primary_logits[0, 0].item())
                prob = float(out.p_fused[0, 0].item())
                branch_scores = {
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                    "p_composite": round(float(out.p_composite[0, 0].item()), 4),
                    "p_freq": round(float(out.p_freq[0, 0].item()), 4),
                    "p_texture": round(float(out.p_texture[0, 0].item()), 4),
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                }
            elif hasattr(self.model, "aux_head"):
                out = self.model(tensor, return_dict=True)
                raw_logit = float(out.primary_logits[0, 0].item())
                prob = float(out.p_fused[0, 0].item())
                branch_scores = {
                    "p_primary": round(float(out.p_primary[0, 0].item()), 4),
                    "p_aux_fake": round(float(out.p_aux_fake[0, 0].item()), 4),
                    "p_fused": round(float(out.p_fused[0, 0].item()), 4),
                }
            else:
                logits = self.model(tensor)
                raw_logit = float(logits[0, 0].item())
                prob = float(torch.sigmoid(logits)[0, 0].item())
        model_time_ms = round((time.perf_counter() - t_model_start) * 1000.0, 2)
        total_time_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)

        verdict = "Deepfake" if prob >= active_threshold else "Real"
        confidence = float(prob if verdict == "Deepfake" else 1.0 - prob)

        stage_latencies = {
            "preprocessing_ms": prep_time_ms,
            "model_forward_ms": model_time_ms,
            "total_ms": total_time_ms,
        }

        meta = prep_result.to_metadata_dict()
        if branch_scores:
            meta["branch_scores"] = branch_scores

        return PredictionResult(
            video_path=str(video_p.resolve()),
            verdict=verdict,
            probability=round(prob, 4),
            raw_logit=round(raw_logit, 4),
            confidence=round(confidence, 4),
            threshold=active_threshold,
            frames_analyzed=total_frames,
            faces_detected=detected_frames,
            device=str(self.device),
            model_version=self.model_version,
            latency_ms=stage_latencies,
            error=None,
            metadata=meta,
            decision="FAKE" if verdict == "Deepfake" else "REAL",
            model_info=self._get_model_info_dict(),
        )

    def predict_batch(
        self,
        video_paths: List[Union[str, Path]],
        threshold: Optional[float] = None,
        calibrated: bool = False,
    ) -> List[PredictionResult]:
        """Run sequential inference over a batch of video paths."""
        return [self.predict_video(p, threshold=threshold, calibrated=calibrated) for p in video_paths]


class EnsemblePredictor:
    """Decision-level evidence fusion predictor (Milestone 22).

    Combines Champion M13.5 (static frequency & texture) and Baseline M18
    (3D differential kinematics) using validation-calibrated log-odds Platt scaling:
        logit(p_fused) = w1 * logit(p_13) + w2 * logit(p_18) + b*
        p_fused = sigmoid(logit(p_fused))
    Decision threshold: tau_95* = 0.9191 (operating policy for high clean specificity).
    """

    def __init__(
        self,
        predictor_m13: VideoPredictor,
        predictor_m18: VideoPredictor,
        fusion_config_path: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.predictor_m13 = predictor_m13
        self.predictor_m18 = predictor_m18
        self.device = predictor_m13.device
        self.config = predictor_m13.config

        cfg_p = Path(fusion_config_path) if fusion_config_path else Path("checkpoints/m22_ensemble/fusion_config.json")
        self.checkpoint_path = cfg_p
        self.model_version = "m22-ensemble"

        # Defaults from Milestone 22 pre-registered audit
        self.w1 = 0.4092
        self.w2 = 0.3590
        self.bias = 1.4219
        self.default_threshold = 0.9191
        self.strategy_name = "strategy_2_log_odds_platt"

        if cfg_p.is_file():
            try:
                with open(cfg_p, "r") as f:
                    data = json.load(f)
                strat = data.get("best_validation_strategy", {})
                params = strat.get("parameters", {})
                self.w1 = float(params.get("w1", self.w1))
                self.w2 = float(params.get("w2", self.w2))
                self.bias = float(params.get("b", self.bias))
                self.default_threshold = float(strat.get("tau_95", self.default_threshold))
                self.strategy_name = strat.get("strategy_name", self.strategy_name)
            except Exception:
                pass

        self.threshold = float(threshold) if threshold is not None else self.default_threshold
        self.calibration_metadata = {
            "version": "m22_validation_platt",
            "strategy": self.strategy_name,
            "weights": {"w1_m13": self.w1, "w2_m18": self.w2, "bias": self.bias},
            "thresholds": {"classification_threshold": self.threshold, "tau_95": self.default_threshold},
        }

    def _get_model_info_dict(self) -> Dict[str, Any]:
        return {
            "name": "m22-ensemble",
            "version": "m22",
            "architecture": "Calibrated Log-Odds Platt Fusion [Champion M13.5 + Baseline M18]",
            "device": str(self.device),
            "fusion_strategy": self.strategy_name,
            "fusion_weights": {"w1_m13": self.w1, "w2_m18": self.w2, "bias": self.bias},
            "operating_threshold_tau_95": self.default_threshold,
        }

    def _fuse_probabilities(self, p1: float, p2: float) -> Tuple[float, float]:
        """Compute fused log-odds and fused probability."""
        eps = 1e-5
        p1_c = min(max(p1, eps), 1.0 - eps)
        p2_c = min(max(p2, eps), 1.0 - eps)
        lo1 = math.log(p1_c / (1.0 - p1_c))
        lo2 = math.log(p2_c / (1.0 - p2_c))
        z_fused = self.w1 * lo1 + self.w2 * lo2 + self.bias
        p_fused = 1.0 / (1.0 + math.exp(-z_fused))
        return z_fused, p_fused

    def predict_video(
        self,
        video_path: Union[str, Path],
        threshold: Optional[float] = None,
        video_id: Optional[str] = None,
        calibrated: bool = True,
    ) -> PredictionResult:
        """Run sequential inference across M13.5 and M18, then apply calibrated log-odds fusion."""
        active_thresh = float(threshold) if threshold is not None else self.threshold

        # Run M13.5 inference
        res_m13 = self.predictor_m13.predict_video(
            video_path=video_path,
            threshold=threshold,
            video_id=video_id,
            calibrated=calibrated,
        )
        if res_m13.verdict in ("Reject", "Error"):
            res_m13.model_version = self.model_version
            res_m13.model_info = self._get_model_info_dict()
            res_m13.calibration_info = self.calibration_metadata
            return res_m13

        # Run M18 inference
        res_m18 = self.predictor_m18.predict_video(
            video_path=video_path,
            threshold=threshold,
            video_id=video_id,
            calibrated=calibrated,
        )
        if res_m18.verdict in ("Reject", "Error"):
            res_m18.model_version = self.model_version
            res_m18.model_info = self._get_model_info_dict()
            res_m18.calibration_info = self.calibration_metadata
            return res_m18

        p13 = float(res_m13.probability)
        p18 = float(res_m18.probability)
        z_fused, p_fused = self._fuse_probabilities(p13, p18)

        verdict = "Deepfake" if p_fused >= active_thresh else "Real"
        confidence = float(p_fused if verdict == "Deepfake" else 1.0 - p_fused)

        total_lat = (
            (res_m13.latency_ms.get("total_ms", 0.0) if res_m13.latency_ms else 0.0)
            + (res_m18.latency_ms.get("total_ms", 0.0) if res_m18.latency_ms else 0.0)
        )

        m13_scores = (res_m13.metadata or {}).get("branch_scores", {})
        m18_scores = (res_m18.metadata or {}).get("branch_scores", {})

        signals = {
            "m13_5_probability": round(p13, 4),
            "m18_probability": round(p18, 4),
            "fused_probability": round(p_fused, 4),
            "static_freq_score": m13_scores.get("p_freq", 0.0),
            "static_texture_score": m13_scores.get("p_texture", 0.0),
            "kinematic_motion_score": m18_scores.get("p_kin", 0.0),
            "m18_gate_alpha": m18_scores.get("gate_values", 0.5),
            "fusion_weights": {"w1_m13": self.w1, "w2_m18": self.w2, "bias": self.bias},
            "decision_threshold_tau_95": self.default_threshold,
        }

        meta = {
            "ensemble": True,
            "strategy": self.strategy_name,
            "signals": signals,
            "m13_result": res_m13.to_dict(),
            "m18_result": res_m18.to_dict(),
        }

        return PredictionResult(
            video_path=res_m13.video_path,
            verdict=verdict,
            probability=round(p_fused, 4),
            raw_logit=round(z_fused, 4),
            confidence=round(confidence, 4),
            threshold=active_thresh,
            frames_analyzed=res_m13.frames_analyzed,
            faces_detected=res_m13.faces_detected,
            device=str(self.device),
            model_version=self.model_version,
            latency_ms={
                "m13_latency_ms": res_m13.latency_ms.get("total_ms", 0.0) if res_m13.latency_ms else 0.0,
                "m18_latency_ms": res_m18.latency_ms.get("total_ms", 0.0) if res_m18.latency_ms else 0.0,
                "total_ms": round(total_lat, 2),
            },
            error=None,
            metadata=meta,
            decision="FAKE" if verdict == "Deepfake" else "REAL",
            uncertainty=res_m13.uncertainty,
            quality=res_m13.quality,
            model_info=self._get_model_info_dict(),
            calibration_info=self.calibration_metadata,
            diagnostics={
                "ensemble_concordance": "Concordant" if (p13 >= 0.5) == (p18 >= 0.5) else "Discordant",
                "static_vs_kinematic": f"M13={p13:.3f}, M18={p18:.3f}",
            },
        )

    def predict_tensor(
        self,
        tensor: torch.Tensor,
        threshold: Optional[float] = None,
        video_id: Optional[str] = None,
        calibrated: bool = True,
    ) -> PredictionResult:
        """Run tensor inference across M13.5 and M18, then apply calibrated log-odds fusion."""
        active_thresh = float(threshold) if threshold is not None else self.threshold
        res_m13 = self.predictor_m13.predict_tensor(tensor, threshold=threshold, video_id=video_id, calibrated=calibrated)
        res_m18 = self.predictor_m18.predict_tensor(tensor, threshold=threshold, video_id=video_id, calibrated=calibrated)

        p13 = float(res_m13.probability)
        p18 = float(res_m18.probability)
        z_fused, p_fused = self._fuse_probabilities(p13, p18)

        verdict = "Deepfake" if p_fused >= active_thresh else "Real"
        confidence = float(p_fused if verdict == "Deepfake" else 1.0 - p_fused)

        total_lat = (
            (res_m13.latency_ms.get("total_ms", 0.0) if res_m13.latency_ms else 0.0)
            + (res_m18.latency_ms.get("total_ms", 0.0) if res_m18.latency_ms else 0.0)
        )

        m13_scores = (res_m13.metadata or {}).get("branch_scores", {})
        m18_scores = (res_m18.metadata or {}).get("branch_scores", {})

        signals = {
            "m13_5_probability": round(p13, 4),
            "m18_probability": round(p18, 4),
            "fused_probability": round(p_fused, 4),
            "static_freq_score": m13_scores.get("p_freq", 0.0),
            "static_texture_score": m13_scores.get("p_texture", 0.0),
            "kinematic_motion_score": m18_scores.get("p_kin", 0.0),
            "m18_gate_alpha": m18_scores.get("gate_values", 0.5),
            "fusion_weights": {"w1_m13": self.w1, "w2_m18": self.w2, "bias": self.bias},
            "decision_threshold_tau_95": self.default_threshold,
        }

        meta = {
            "ensemble": True,
            "strategy": self.strategy_name,
            "signals": signals,
            "m13_result": res_m13.to_dict(),
            "m18_result": res_m18.to_dict(),
        }

        return PredictionResult(
            video_path=res_m13.video_path,
            verdict=verdict,
            probability=round(p_fused, 4),
            raw_logit=round(z_fused, 4),
            confidence=round(confidence, 4),
            threshold=active_thresh,
            frames_analyzed=res_m13.frames_analyzed,
            faces_detected=res_m13.faces_detected,
            device=str(self.device),
            model_version=self.model_version,
            latency_ms={"total_ms": round(total_lat, 2)},
            error=None,
            metadata=meta,
            decision="FAKE" if verdict == "Deepfake" else "REAL",
            model_info=self._get_model_info_dict(),
            calibration_info=self.calibration_metadata,
            diagnostics={
                "ensemble_concordance": "Concordant" if (p13 >= 0.5) == (p18 >= 0.5) else "Discordant",
                "static_vs_kinematic": f"M13={p13:.3f}, M18={p18:.3f}",
            },
        )


