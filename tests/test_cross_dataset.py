"""Unit tests for CrossDatasetEvaluator."""

import json
from pathlib import Path
import pandas as pd
import torch

from src.data.storage import ProcessedDataStorage
from src.data.types import PreprocessResult
from src.evaluation.cross_dataset import CrossDatasetEvaluator
from src.models.detector import DeepfakeDetector
from src.training.checkpoint import CheckpointManager
from src.utils.config import AppConfig


def test_cross_dataset_evaluator_lifecycle(tmp_path: Path):
    """Verify CrossDatasetEvaluator performs read-only evaluation across external manifests."""
    storage_dir = tmp_path / "processed"
    storage = ProcessedDataStorage(storage_dir)

    # Preprocessed samples for target benchmark (Celeb-DF)
    for vid, lbl in [("cdf_r1", 0), ("cdf_f1", 1)]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(
            PreprocessResult(video_id=vid, group_id=vid, split="test", status="success", tensor=t, valid_frames=16),
            split="test",
        )

    # External manifest
    manifest_df = pd.DataFrame([
        {"video_id": "cdf_r1", "group_id": "id0", "video_path": "a.mp4", "label": 0, "dataset_name": "Celeb-DF", "split": "test"},
        {"video_id": "cdf_f1", "group_id": "id1", "video_path": "b.mp4", "label": 1, "dataset_name": "Celeb-DF", "split": "test"},
    ])
    manifest_path = tmp_path / "celebdf_test.csv"
    manifest_df.to_csv(manifest_path, index=False)

    cfg = AppConfig()
    cfg.model.backbone.weights = None
    cfg.model.backbone.freeze_strategy = "layer2"
    cfg.storage.processed_dir = str(storage_dir)
    cfg.device = "cpu"

    model = DeepfakeDetector(cfg.model)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, monitor_metric="val_auc")

    # Mock in-domain val metrics
    from src.evaluation.metrics import EvaluationMetrics
    in_domain_metrics = EvaluationMetrics(
        loss=0.3,
        accuracy=0.92,
        precision=0.91,
        recall=0.93,
        f1=0.92,
        roc_auc=0.96,
        eer=0.04,
        confusion_matrix={"tp": 1, "fp": 0, "tn": 1, "fn": 0},
        num_samples=2,
    )
    optimizer = torch.optim.Adam(model.parameters())
    _, best_path, _ = ckpt_mgr.save(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        metrics=in_domain_metrics,
        config=cfg,
    )

    # Initial parameter clone to verify read-only invariant
    initial_weight = model.classifier.net[0].weight.clone()

    evaluator = CrossDatasetEvaluator(
        checkpoint_path=best_path,
        config=cfg,
        device="cpu",
    )

    out_json = tmp_path / "cross_dataset_results.json"
    results = evaluator.evaluate_suite(
        benchmarks={"Celeb-DF": manifest_path},
        split="test",
        output_json=out_json,
    )

    assert "Celeb-DF" in results["benchmarks"]
    cdf_res = results["benchmarks"]["Celeb-DF"]
    assert cdf_res["num_samples"] == 2
    assert "roc_auc" in cdf_res["metrics"]
    assert cdf_res["auc_drop_vs_indomain"] is not None
    assert out_json.exists()

    # Read-only verification: weights unchanged
    assert torch.equal(evaluator.model.classifier.net[0].weight, initial_weight)
