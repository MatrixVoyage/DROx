"""Empirical Baseline Manager and Z-Score Computation.

Implements rigorous validation for baseline references:
1. Versioned schema and provenance tracking (reference dataset, independent video count >= 100).
2. Finite numeric verification and monotonicity constraints (p05 <= median <= p95).
3. Non-zero variance guards (std <= 0 returns None instead of artificial runaway z-scores).
4. Explicit error handling distinguishing missing from corrupt baseline files.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_z_score(value: float, mean: float, std: float) -> Optional[float]:
    """Computes z-score with strict finite checks and zero-variance guard.

    Returns:
        float z-score if valid, or None if std <= 0 (not estimable from reference).
    Raises:
        ValueError if value, mean, or std is non-finite (NaN or Inf).
    """
    if not all(map(math.isfinite, (value, mean, std))):
        raise ValueError("Baseline and sample values must be finite")
    if std <= 0.0:
        return None
    return float((value - mean) / std)


@dataclass(frozen=True)
class BaselineReference:
    metric_name: str
    feature_version: str
    dataset_name: str
    sample_video_count: int
    mean: float
    std: float
    median: float
    p05: float
    p95: float
    tail_policy: str  # "UPPER_TAIL" | "LOWER_TAIL" | "TWO_TAILED"

    def __post_init__(self) -> None:
        values = (self.mean, self.std, self.median, self.p05, self.p95)
        if not all(map(math.isfinite, values)):
            raise ValueError(f"Non-finite baseline values detected for {self.metric_name}")
        if self.p05 > self.median or self.median > self.p95:
            raise ValueError(f"Invalid quantile ordering for {self.metric_name}: p05={self.p05}, median={self.median}, p95={self.p95}")
        if self.sample_video_count < 100:
            raise ValueError(f"Insufficient reference sample count ({self.sample_video_count} < 100) for {self.metric_name}")


class EmpiricalBaselineManager:
    """Loads and validates versioned empirical baseline reference distributions."""

    def __init__(self, baseline_path: Optional[Path | str] = None) -> None:
        self.baselines: Dict[str, BaselineReference] = {}
        if baseline_path:
            self.load_baselines(baseline_path)

    def load_baselines(self, path: Path | str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Baseline file not found: {p}")

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt or invalid JSON baseline file: {e}") from e

        if not isinstance(data, dict) or "schema_version" not in data or "metrics" not in data:
            raise ValueError("Baseline file missing required 'schema_version' or 'metrics' key")

        for metric_name, m_data in data["metrics"].items():
            ref = BaselineReference(
                metric_name=metric_name,
                feature_version=str(m_data.get("feature_version", "1.0")),
                dataset_name=str(m_data.get("dataset_name", "FaceForensics++")),
                sample_video_count=int(m_data.get("sample_video_count", 0)),
                mean=float(m_data.get("mean", 0.0)),
                std=float(m_data.get("std", 0.0)),
                median=float(m_data.get("median", 0.0)),
                p05=float(m_data.get("p05", 0.0)),
                p95=float(m_data.get("p95", 0.0)),
                tail_policy=str(m_data.get("tail_policy", "TWO_TAILED")),
            )
            self.baselines[metric_name] = ref

    def evaluate_sample(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Evaluates whether a sample value is an anomaly against the calibrated reference."""
        if metric_name not in self.baselines:
            return {
                "metric_name": metric_name,
                "is_calibrated": False,
                "is_anomaly": False,
                "z_score": None,
                "reason": "Metric not found in calibrated baseline distribution",
            }

        ref = self.baselines[metric_name]
        z = compute_z_score(value, ref.mean, ref.std)

        # Tail policy check
        if ref.tail_policy == "UPPER_TAIL":
            is_anomaly = value > ref.p95
        elif ref.tail_policy == "LOWER_TAIL":
            is_anomaly = value < ref.p05
        else:  # TWO_TAILED
            is_anomaly = (value < ref.p05) or (value > ref.p95)

        return {
            "metric_name": metric_name,
            "sample_value": value,
            "is_calibrated": True,
            "is_anomaly": is_anomaly,
            "z_score": z,
            "tail_policy": ref.tail_policy,
            "p05": ref.p05,
            "median": ref.median,
            "p95": ref.p95,
            "sample_video_count": ref.sample_video_count,
        }
