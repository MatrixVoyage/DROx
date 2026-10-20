"""Unit tests for Milestone 7 CLI subcommands (predict, export, benchmark-latency, serve)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.cli import main
from src.inference.predictor import PredictionResult, VideoPredictor


def test_cli_predict_help(capsys):
    """Verify predict subparser exists and prints help."""
    with patch("sys.argv", ["cli.py", "predict", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--video" in captured.out
    assert "--checkpoint" in captured.out
    assert "--threshold" in captured.out


def test_cli_predict_single_video(tmp_path: Path, monkeypatch, capsys):
    """Verify predict command execution with mocked predictor."""
    mock_result = PredictionResult(
        video_path="test_video.mp4",
        verdict="DEEPFAKE",
        probability=0.8175,
        raw_logit=1.5,
        confidence=0.8175,
        threshold=0.5,
        frames_analyzed=16,
        faces_detected=16,
        device="cpu",
        model_version="best.pth",
        latency_ms={"preprocessing_ms": 150.0, "model_forward_ms": 25.0, "total_ms": 175.0},
    )

    with patch.object(VideoPredictor, "__init__", return_value=None), \
         patch.object(VideoPredictor, "predict_video", return_value=mock_result):
        out_json = tmp_path / "pred_out.json"
        test_args = [
            "cli.py",
            "predict",
            "--video", "test_video.mp4",
            "--checkpoint", "dummy.pth",
            "--output", str(out_json),
            "--cpu",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        main()

        assert out_json.exists()
        with open(out_json, "r") as f:
            data = json.load(f)
        assert data["verdict"] == "DEEPFAKE"
        assert data["probability"] == 0.8175


def test_cli_export_help(capsys):
    """Verify export subparser exists and prints help."""
    with patch("sys.argv", ["cli.py", "export", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--checkpoint" in captured.out
    assert "--format" in captured.out
    assert "--output-dir" in captured.out


def test_cli_export_execution(tmp_path: Path, monkeypatch):
    """Verify export command execution with mocked export_checkpoint."""
    mock_export_res = {
        "checkpoint": "dummy.pth",
        "exports": {
            "torchscript": {"exported_path": "dummy.pt", "parity_verified": True, "max_abs_diff": 0.0},
            "onnx": {"exported_path": "dummy.onnx", "parity_verified": True, "max_abs_diff": 1e-7},
        }
    }

    with patch("src.inference.export.export_checkpoint", return_value=mock_export_res):
        out_audit = tmp_path / "export_audit.json"
        test_args = [
            "cli.py",
            "export",
            "--checkpoint", "dummy.pth",
            "--output-dir", str(tmp_path / "export"),
            "--output-json", str(out_audit),
            "--cpu",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        main()

        assert out_audit.exists()
        with open(out_audit, "r") as f:
            data = json.load(f)
        assert "exports" in data
        assert data["exports"]["torchscript"]["parity_verified"] is True


def test_cli_benchmark_help(capsys):
    """Verify benchmark-latency subparser exists and prints help."""
    with patch("sys.argv", ["cli.py", "benchmark-latency", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--checkpoint" in captured.out
    assert "--num-runs" in captured.out
    assert "--output" in captured.out


def test_cli_serve_help(capsys):
    """Verify serve subparser exists and prints help."""
    with patch("sys.argv", ["cli.py", "serve", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--checkpoint" in captured.out
    assert "--port" in captured.out
    assert "--host" in captured.out
