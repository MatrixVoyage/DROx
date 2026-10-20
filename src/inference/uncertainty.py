"""Calibrated uncertainty estimation, out-of-distribution (OOD) detection, and three-state decision policy.

Provides:
- MCDropoutEstimator: Epistemic uncertainty estimation via Monte Carlo Dropout.
- OODFeatureEstimator: Penultimate feature distance against in-distribution reference.
- QualityAssessor: Extraction and thresholding of video quality and face detection signals.
- CalibratedDecisionPolicy: Strict three-state decision policy (REAL, FAKE, REJECT).
- CalibratedPredictionResult: Strongly typed diagnostic result container.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.detector import DeepfakeDetector
from src.utils.logger import setup_logger

logger = setup_logger("uncertainty")


@dataclass
class QualitySignals:
    """Diagnostic signals regarding video quality and face extraction."""

    total_frames: int
    detected_frames: int
    fallback_frames: int
    failed_frames: int
    fallback_ratio: float
    is_acceptable: bool
    rejection_reason: Optional[str] = None


@dataclass
class UncertaintySignals:
    """Predictive uncertainty and distribution metrics."""

    mean_probability: float
    epistemic_variance: float
    entropy: float
    mc_samples: int
    ood_distance: float
    is_ood: bool
    is_uncertain: bool
    rejection_reason: Optional[str] = None


@dataclass
class CalibratedPredictionResult:
    """Authoritative three-state prediction result.

    Statistical Definitions:
    - probability: Calibrated predictive expectation of FAKE after temperature scaling:
      P(Deepfake | x) = mean(sigmoid(logits / T)) in [0.0, 1.0].
    - confidence: Posterior confidence in the selected verdict class:
      P(Verdict | x) = (probability if verdict=='Deepfake' else 1 - probability) in [0.5, 1.0]
      when accepted; strictly 0.0 when rejected.
    - epistemic_variance: Empirical sample variance of predicted probabilities across
      stochastic MC-Dropout forward passes: var(sigmoid(logits / T)).
    - predictive_entropy: Binary Shannon entropy (in bits) computed from calibrated probability:
      -p * log2(p) - (1-p) * log2(1-p).
    - ood_distance: Standardized Mahalanobis/Euclidean distance from validation feature centroid.
    """

    video_path: Optional[str]
    verdict: str  # "Real", "Deepfake", or "Reject"
    probability: float  # Calibrated predictive expectation of FAKE in [0.0, 1.0]
    confidence: float  # Calibrated confidence in selected verdict: P(Verdict | x) in [0.5, 1.0] or 0.0 on Reject
    rejection_code: Optional[str]  # Compound code: e.g. "LOW_QUALITY_AND_HIGH_UNCERTAINTY"
    rejection_reason: Optional[str]  # Human-readable diagnostic explanation
    quality_signals: QualitySignals
    uncertainty_signals: UncertaintySignals
    device: str
    model_version: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class MCDropoutEstimator:
    """Monte Carlo Dropout estimator for epistemic uncertainty quantification with temperature scaling."""

    def __init__(self, model: Union[DeepfakeDetector, nn.Module], num_samples: int = 15, temperature: float = 1.0):
        self.model = model
        self.num_samples = max(2, num_samples)
        self.temperature = max(1e-4, float(temperature))

    def _enable_dropout(self) -> None:
        """Enable dropout layers while keeping batchnorm and other layers in eval mode."""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def estimate(
        self,
        tensor: torch.Tensor,
        device: torch.device,
    ) -> Tuple[float, float, float, torch.Tensor]:
        """Perform stochastic forward passes to estimate predictive mean and epistemic variance.

        Args:
            tensor: Facial crop sequence of shape (1, T, 3, H, W) or (T, 3, H, W).
            device: Target execution device.

        Returns:
            Tuple of (mean_prob, variance, entropy, mean_temporal_features).
            - mean_prob: Calibrated predictive expectation of FAKE: mean(sigmoid(logits / T)).
            - variance: Epistemic variance across stochastic passes: var(sigmoid(logits / T)).
            - entropy: Binary Shannon entropy (in bits) of the calibrated mean probability.
            - mean_temporal_features: Average penultimate temporal embedding across passes.
        """
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)

        self.model.eval()
        self._enable_dropout()

        probs = []
        feature_list = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                if hasattr(self.model, "kinematic_branch"):
                    out = self.model(tensor, mode="dual_stream")
                    scaled_logits = out.primary_logits / self.temperature
                    prob = torch.sigmoid(scaled_logits).squeeze().item()
                    temporal_features = out.trunk_features
                elif hasattr(self.model, "res_branch") or hasattr(self.model, "srm_layer"):
                    out = self.model(tensor)
                    scaled_logits = out.primary_logits / self.temperature
                    prob = torch.sigmoid(scaled_logits).squeeze().item()
                    temporal_features = out.temporal_features
                elif hasattr(self.model, "fusion_head"):
                    out = self.model(tensor, return_dict=True)
                    scaled_logits = out.primary_logits / self.temperature
                    prob = torch.sigmoid(scaled_logits).squeeze().item()
                    temporal_features = out.temporal_features
                elif hasattr(self.model, "aux_head"):
                    out = self.model(tensor, return_dict=True)
                    scaled_logits = out.primary_logits / self.temperature
                    p_prim = torch.sigmoid(scaled_logits).squeeze().item()
                    p_aux = out.p_aux_fake.squeeze().item()
                    prob = 0.5 * p_prim + 0.5 * p_aux
                    temporal_features = out.temporal_features
                else:
                    logits, temporal_features = self.model(tensor, return_embeddings=True)
                    scaled_logits = logits / self.temperature
                    prob = torch.sigmoid(scaled_logits).squeeze().item()
                probs.append(prob)
                feature_list.append(temporal_features.squeeze(0).cpu())

        # Reset model back to clean eval mode
        self.model.eval()

        probs_arr = np.array(probs, dtype=np.float32)
        mean_p = float(np.mean(probs_arr))
        variance = float(np.var(probs_arr))

        # Binary Shannon entropy (clamped to prevent log(0))
        p_clamped = np.clip(mean_p, 1e-6, 1.0 - 1e-6)
        entropy = float(-p_clamped * np.log2(p_clamped) - (1.0 - p_clamped) * np.log2(1.0 - p_clamped))

        # Average feature embedding
        mean_features = torch.stack(feature_list, dim=0).mean(dim=0)

        return mean_p, variance, entropy, mean_features


class OODFeatureEstimator:
    """Out-of-Distribution detector based on penultimate feature representation distance."""

    def __init__(self, feature_dim: int = 1280):
        self.feature_dim = feature_dim
        self.centroid: Optional[np.ndarray] = None
        self.inv_diag_var: Optional[np.ndarray] = None
        self.threshold: float = 3.5  # Standard deviation distance threshold
        self.is_fitted: bool = False

    def fit(self, features: np.ndarray, std_threshold: float = 3.5) -> None:
        """Fit in-distribution reference statistics on validation features.

        Args:
            features: 2D array of shape (N, D) representing clean validation embeddings.
            std_threshold: Distance threshold in standardized units.
        """
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features of shape (N, {self.feature_dim}), got {features.shape}"
            )

        self.centroid = np.mean(features, axis=0)
        variance = np.var(features, axis=0) + 1e-5  # Regularizer for numerical stability
        self.inv_diag_var = 1.0 / np.sqrt(variance)
        self.threshold = float(std_threshold)
        self.is_fitted = True
        logger.info(f"OODFeatureEstimator fitted on {len(features)} reference embeddings.")

    def compute_distance(self, feature: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute standardized Euclidean distance from the in-distribution centroid."""
        if not self.is_fitted or self.centroid is None or self.inv_diag_var is None:
            # If not fitted yet, default to distance 0.0 (permissive)
            return 0.0

        if isinstance(feature, torch.Tensor):
            feat_arr = feature.detach().cpu().numpy().flatten()
        else:
            feat_arr = np.asarray(feature).flatten()

        diff = (feat_arr - self.centroid) * self.inv_diag_var
        # Normalized root-mean-square standardized distance
        distance = float(np.sqrt(np.mean(diff ** 2)))
        return distance

    def is_ood(self, distance: float) -> bool:
        """Check if computed distance exceeds the OOD threshold."""
        return distance > self.threshold

    def save(self, file_path: Union[str, Path]) -> None:
        """Save reference statistics to file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted OODFeatureEstimator.")
        data = {
            "feature_dim": self.feature_dim,
            "threshold": self.threshold,
            "centroid": self.centroid.tolist() if self.centroid is not None else [],
            "inv_diag_var": self.inv_diag_var.tolist() if self.inv_diag_var is not None else [],
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
        logger.info(f"OOD reference statistics saved to {file_path}")

    def load(self, file_path: Union[str, Path]) -> None:
        """Load reference statistics from file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        self.feature_dim = data["feature_dim"]
        self.threshold = data["threshold"]
        self.centroid = np.array(data["centroid"], dtype=np.float32)
        self.inv_diag_var = np.array(data["inv_diag_var"], dtype=np.float32)
        self.is_fitted = True
        logger.info(f"OOD reference statistics loaded from {file_path}")


class QualityAssessor:
    """Evaluates video quality and face detection extraction metrics."""

    def __init__(
        self,
        max_fallback_ratio: float = 0.50,
        min_detected_frames: int = 8,
    ):
        self.max_fallback_ratio = max_fallback_ratio
        self.min_detected_frames = min_detected_frames

    def assess(
        self,
        total_frames: int,
        detected_frames: int,
        previous_bbox_frames: int = 0,
        center_crop_frames: int = 0,
        failed_frames: int = 0,
    ) -> QualitySignals:
        """Assess facial quality signals."""
        fallback_frames = previous_bbox_frames + center_crop_frames
        fallback_ratio = fallback_frames / total_frames if total_frames > 0 else 1.0

        is_acceptable = True
        reasons = []

        if failed_frames > 0:
            is_acceptable = False
            reasons.append(f"{failed_frames} corrupted/empty frames")

        if detected_frames < self.min_detected_frames:
            is_acceptable = False
            reasons.append(
                f"Insufficient face detections ({detected_frames}/{total_frames} < {self.min_detected_frames})"
            )

        if fallback_ratio > self.max_fallback_ratio:
            is_acceptable = False
            reasons.append(
                f"Excessive fallback crops ({fallback_ratio:.1%} > {self.max_fallback_ratio:.1%})"
            )

        reason_str = "; ".join(reasons) if not is_acceptable else None
        return QualitySignals(
            total_frames=total_frames,
            detected_frames=detected_frames,
            fallback_frames=fallback_frames,
            failed_frames=failed_frames,
            fallback_ratio=round(fallback_ratio, 4),
            is_acceptable=is_acceptable,
            rejection_reason=reason_str,
        )


class CalibratedDecisionPolicy:
    """Decoupled three-state decision policy implementing REAL, FAKE, REJECT semantics.

    REJECT explicitly indicates that the input falls outside the validated operating
    envelope or that predictive uncertainty is too elevated to support a trustworthy verdict.
    """

    def __init__(
        self,
        classification_threshold: float = 0.50,
        max_uncertainty_variance: float = 0.05,  # std > ~0.22
        ood_distance_threshold: float = 3.5,
        reject_on_ood: bool = True,
    ):
        self.classification_threshold = classification_threshold
        self.max_uncertainty_variance = max_uncertainty_variance
        self.ood_distance_threshold = ood_distance_threshold
        self.reject_on_ood = reject_on_ood

    def decide(
        self,
        probability: float,
        quality: QualitySignals,
        uncertainty: UncertaintySignals,
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """Apply compound three-state decision hierarchy.

        Evaluates quality, out-of-distribution distance, and epistemic uncertainty.
        If multiple conditions trigger, returns a combined reason code
        (e.g., 'LOW_QUALITY_AND_HIGH_UNCERTAINTY') and human-readable diagnostic description.

        Returns:
            Tuple of (verdict, confidence, rejection_code, rejection_reason).
        """
        rejection_codes = []
        detailed_reasons = []

        if not quality.is_acceptable:
            rejection_codes.append("LOW_QUALITY")
            detailed_reasons.append(f"Low Video Quality: {quality.rejection_reason}")

        if self.reject_on_ood and uncertainty.is_ood:
            rejection_codes.append("OOD_HIGH")
            detailed_reasons.append(
                f"Out-of-Distribution: Feature distance {uncertainty.ood_distance:.2f} exceeds threshold {self.ood_distance_threshold:.2f}"
            )

        if uncertainty.is_uncertain:
            rejection_codes.append("HIGH_UNCERTAINTY")
            detailed_reasons.append(
                f"High Epistemic Uncertainty: Model variance {uncertainty.epistemic_variance:.4f} exceeds safety threshold {self.max_uncertainty_variance:.4f}"
            )

        if rejection_codes:
            code_str = "_AND_".join(rejection_codes)
            reason_str = "; ".join(detailed_reasons)
            return "Reject", 0.0, code_str, reason_str

        # Binary decision within validated operating envelope
        if probability >= self.classification_threshold:
            verdict = "Deepfake"
            confidence = probability
        else:
            verdict = "Real"
            confidence = 1.0 - probability

        return verdict, round(float(confidence), 4), None, None
