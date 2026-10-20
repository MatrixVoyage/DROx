"""Unit tests for LatencyProfiler."""

import torch
import torch.nn as nn
from src.inference.profiler import LatencyProfiler, LatencyBenchmarkResult


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 112 * 112, 1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.conv(x)
        feat = feat.view(B, T, -1)
        out = self.fc(feat.mean(dim=1))
        return out


def test_profiler_benchmark_model_cpu():
    """Verify LatencyProfiler benchmark_model returns expected metrics."""
    profiler = LatencyProfiler(device="cpu")
    model = DummyModel()
    metrics = profiler.benchmark_model(model, batch_size=1, num_frames=4, num_runs=5, warmup_runs=2)
    assert "mean_ms" in metrics
    assert "median_ms" in metrics
    assert "p95_ms" in metrics
    assert "fps" in metrics
    assert metrics["mean_ms"] > 0
    assert metrics["fps"] > 0


def test_profiler_benchmark_result_dataclass():
    """Verify LatencyBenchmarkResult serialization."""
    res = LatencyBenchmarkResult(
        device="cuda:0",
        model_version="test.pth",
        batch_size=1,
        num_frames=16,
        num_runs=10,
        warmup_runs=2,
        model_only={"mean_ms": 30.5, "median_ms": 30.1, "p95_ms": 32.0, "p99_ms": 33.0, "fps": 524.6},
    )
    d = res.to_dict()
    assert d["device"] == "cuda:0"
    assert d["model_only"]["mean_ms"] == 30.5
