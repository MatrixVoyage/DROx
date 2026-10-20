"""Unit and integration tests for ExperimentRunner and provenance capture."""

import json
from pathlib import Path
import pandas as pd
import torch

from src.data.dataset import ProcessedVideoDataset
from src.data.storage import ProcessedDataStorage
from src.data.types import PreprocessResult
from src.experiments.runner import ExperimentRunner, collect_provenance
from src.utils.config import AppConfig


def test_collect_provenance():
    """Verify collect_provenance returns expected environment and hardware metadata."""
    prov = collect_provenance(seed=123, device=torch.device("cpu"))
    assert prov["seed"] == 123
    assert "python_version" in prov
    assert "torch_version" in prov
    assert "timestamp" in prov
    assert prov["device"] == "cpu"


def test_experiment_runner_lifecycle(tmp_path: Path):
    """Verify ExperimentRunner creates run dir, saves config, trains, and produces summary JSON."""
    storage_dir = tmp_path / "processed"
    storage = ProcessedDataStorage(storage_dir)

    # 4 train samples
    for vid in ["t1", "t2", "t3", "t4"]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(PreprocessResult(video_id=vid, group_id=vid, split="train", status="success", tensor=t, valid_frames=16), split="train")

    # 2 val samples
    for vid in ["v1", "v2"]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(PreprocessResult(video_id=vid, group_id=vid, split="val", status="success", tensor=t, valid_frames=16), split="val")

    # 2 test samples
    for vid in ["te1", "te2"]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(PreprocessResult(video_id=vid, group_id=vid, split="test", status="success", tensor=t, valid_frames=16), split="test")

    manifest_df = pd.DataFrame([
        {"video_id": "t1", "group_id": "g1", "video_path": "a.mp4", "label": 0, "split": "train"},
        {"video_id": "t2", "group_id": "g2", "video_path": "b.mp4", "label": 1, "split": "train"},
        {"video_id": "t3", "group_id": "g3", "video_path": "c.mp4", "label": 0, "split": "train"},
        {"video_id": "t4", "group_id": "g4", "video_path": "d.mp4", "label": 1, "split": "train"},
        {"video_id": "v1", "group_id": "g5", "video_path": "e.mp4", "label": 0, "split": "val"},
        {"video_id": "v2", "group_id": "g6", "video_path": "f.mp4", "label": 1, "split": "val"},
        {"video_id": "te1", "group_id": "g7", "video_path": "g.mp4", "label": 0, "split": "test"},
        {"video_id": "te2", "group_id": "g8", "video_path": "h.mp4", "label": 1, "split": "test"},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    cfg = AppConfig()
    cfg.model.backbone.weights = None
    cfg.model.backbone.freeze_strategy = "layer2"
    cfg.training.num_epochs = 1
    cfg.training.grad_accum_steps = 1
    cfg.training.monitor_metric = "val_auc"
    cfg.storage.processed_dir = str(storage_dir)
    cfg.device = "cpu"

    runs_root = tmp_path / "runs"
    runner = ExperimentRunner(
        config=cfg,
        manifest_path=manifest_path,
        output_root=runs_root,
        experiment_name="test_exp",
        device="cpu",
    )

    summary = runner.run()

    assert summary["experiment_name"] == "test_exp"
    assert "provenance" in summary
    assert "test_metrics" in summary
    assert summary["test_metrics"] is not None
    assert "accuracy" in summary["test_metrics"]

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "experiment_summary.json").exists()
    assert (run_dir / "checkpoints" / "latest.pth").exists()
    assert (run_dir / "checkpoints" / "best.pth").exists()
