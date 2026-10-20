"""Unit tests for Milestone 6 CLI subcommands."""

import json
from pathlib import Path
import pytest
import torch

from src.cli import main
from src.data.storage import ProcessedDataStorage
from src.data.types import PreprocessResult
from src.models.detector import DeepfakeDetector
from src.training.checkpoint import CheckpointManager
from src.utils.config import AppConfig


def test_cli_manifest_celebdf(tmp_path: Path, monkeypatch):
    """Verify cli manifest-celebdf generates expected manifest."""
    root = tmp_path / "celeb_df"
    (root / "Celeb-real").mkdir(parents=True)
    (root / "Celeb-synthesis").mkdir(parents=True)
    (root / "Celeb-real" / "id0_0000.mp4").touch()
    (root / "Celeb-real" / "id1_0000.mp4").touch()
    (root / "Celeb-synthesis" / "id0_id1_0000.mp4").touch()

    out_manifest = tmp_path / "celeb_manifest.csv"
    test_args = [
        "cli.py",
        "manifest-celebdf",
        "--root", str(root),
        "--output", str(out_manifest),
        "--train-ratio", "0.5",
        "--val-ratio", "0.5",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert out_manifest.exists()
    import pandas as pd
    df = pd.read_csv(out_manifest)
    assert len(df) == 3
    assert (df["dataset_name"] == "Celeb-DF-v2").all()


def test_cli_manifest_dfdc(tmp_path: Path, monkeypatch):
    """Verify cli manifest-dfdc generates expected manifest."""
    root = tmp_path / "dfdc"
    part = root / "part0"
    part.mkdir(parents=True)
    (part / "r.mp4").touch()
    (part / "f.mp4").touch()

    meta = {
        "r.mp4": {"label": "REAL"},
        "f.mp4": {"label": "FAKE", "original": "r.mp4"},
    }
    with open(part / "metadata.json", "w") as f:
        json.dump(meta, f)

    out_manifest = tmp_path / "dfdc_manifest.csv"
    test_args = [
        "cli.py",
        "manifest-dfdc",
        "--root", str(root),
        "--output", str(out_manifest),
        "--train-ratio", "0.5",
        "--val-ratio", "0.5",
        "--test-ratio", "0.0",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert out_manifest.exists()
    import pandas as pd
    df = pd.read_csv(out_manifest)
    assert len(df) == 2
    assert (df["dataset_name"] == "DFDC").all()


def test_cli_generate_report(tmp_path: Path, monkeypatch):
    """Verify cli generate-report produces markdown and JSON."""
    summary_path = tmp_path / "exp_summary.json"
    summary_data = {
        "experiment_name": "test_cli_run",
        "test_metrics": {
            "accuracy": 0.925,
            "precision": 0.930,
            "recall": 0.935,
            "f1": 0.932,
            "roc_auc": 0.960,
            "eer": 0.045,
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f)

    out_dir = tmp_path / "reports"
    test_args = [
        "cli.py",
        "generate-report",
        "--summary", str(summary_path),
        "--output-dir", str(out_dir),
        "--stem", "cli_test_report",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert (out_dir / "cli_test_report.md").exists()
    assert (out_dir / "cli_test_report.json").exists()
