"""Performance profiling utilities for model-only and end-to-end video inference."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from src.inference.predictor import VideoPredictor
from src.utils.logger import setup_logger

logger = setup_logger("latency_profiler")


@dataclass
class LatencyBenchmarkResult:
    """Container for dual-tier latency benchmark metrics."""

    device: str
    model_version: str
    batch_size: int
    num_frames: int
    num_runs: int
    warmup_runs: int
    model_only: Dict[str, float]
    end_to_end: Optional[Dict[str, Any]] = None
    vram_usage_mb: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatencyProfiler:
    """Profiles model-only forward latency and end-to-end video inference latency."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def benchmark_model(
        self,
        model: nn.Module,
        batch_size: int = 1,
        num_frames: int = 16,
        num_runs: int = 30,
        warmup_runs: int = 5,
    ) -> Dict[str, float]:
        """Profile isolated model forward pass latency on synthetic tensor.

        Args:
            model: DeepfakeDetector model instance.
            batch_size: Batch size B (default: 1).
            num_frames: Temporal sequence length T (default: 16).
            num_runs: Timed iteration count.
            warmup_runs: Warmup iteration count.

        Returns:
            Dictionary with mean_ms, median_ms, p95_ms, p99_ms, and model_fps.
        """
        model.eval()
        model.to(self.device)
        is_cuda = self.device.type == "cuda"

        x = torch.randn(batch_size, num_frames, 3, 112, 112, dtype=torch.float32, device=self.device)

        # 1. Warmup
        with torch.inference_mode():
            for _ in range(warmup_runs):
                _ = model(x)
                if is_cuda:
                    torch.cuda.synchronize()

        # 2. Timed runs
        timings_ms: List[float] = []
        with torch.inference_mode():
            for _ in range(num_runs):
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                _ = model(x)

                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                timings_ms.append((t1 - t0) * 1000.0)

        timings_arr = np.array(timings_ms)
        mean_ms = float(np.mean(timings_arr))
        median_ms = float(np.median(timings_arr))
        p95_ms = float(np.percentile(timings_arr, 95))
        p99_ms = float(np.percentile(timings_arr, 99))
        fps = float((batch_size * num_frames) / (mean_ms / 1000.0))

        return {
            "mean_ms": round(mean_ms, 2),
            "median_ms": round(median_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "fps": round(fps, 1),
        }

    def benchmark_end_to_end(
        self,
        predictor: VideoPredictor,
        video_path: Union[str, Path],
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """Profile complete end-to-end video pipeline on a real video file.

        Args:
            predictor: VideoPredictor instance.
            video_path: Path to real physical video.
            num_runs: Repetition count.

        Returns:
            Dictionary with breakdown of preprocessing, model forward, and total latency.
        """
        video_p = Path(video_path)
        if not video_p.is_file():
            raise FileNotFoundError(f"Video file not found: {video_p}")

        runs = []
        for _ in range(num_runs):
            res = predictor.predict_video(video_p)
            if res.error:
                raise RuntimeError(f"Video inference failed during profiling: {res.error}")
            runs.append(res.latency_ms)

        prep_times = [r.get("preprocessing_ms", 0.0) for r in runs]
        model_times = [r.get("model_forward_ms", 0.0) for r in runs]
        total_times = [r.get("total_ms", 0.0) for r in runs]

        return {
            "video_path": str(video_p.resolve()),
            "num_runs": num_runs,
            "mean_preprocessing_ms": round(float(np.mean(prep_times)), 2),
            "mean_model_forward_ms": round(float(np.mean(model_times)), 2),
            "mean_total_ms": round(float(np.mean(total_times)), 2),
        }

    def run_full_profile(
        self,
        predictor: VideoPredictor,
        video_path: Optional[Union[str, Path]] = None,
        num_runs: int = 20,
        warmup_runs: int = 5,
    ) -> LatencyBenchmarkResult:
        """Run complete dual-tier profiling."""
        model_metrics = self.benchmark_model(
            model=predictor.model,
            batch_size=1,
            num_frames=16,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        )

        e2e_metrics = None
        if video_path is not None:
            e2e_metrics = self.benchmark_end_to_end(
                predictor=predictor,
                video_path=video_path,
                num_runs=min(num_runs, 5),
            )

        vram_usage = None
        if self.device.type == "cuda":
            vram_usage = {
                "allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024 ** 2), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024 ** 2), 2),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2), 2),
            }

        return LatencyBenchmarkResult(
            device=str(self.device),
            model_version=predictor.model_version,
            batch_size=1,
            num_frames=16,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            model_only=model_metrics,
            end_to_end=e2e_metrics,
            vram_usage_mb=vram_usage,
        )
