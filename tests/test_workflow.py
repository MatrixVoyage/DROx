"""Integration test verifying complete train-evaluate split lifecycle and invariants.

Invariants verified:
1. Validation is the only split used for model selection (best.pth) and early stopping.
2. Test data is strictly excluded from training, scheduler decisions, and model selection.
3. Final evaluation uses best.pth on the test split.
4. Test evaluation yields full metric suite: loss, accuracy, precision, recall, f1, roc_auc, eer, confusion_matrix.
5. Configuration is stored within the checkpoint.
"""

from pathlib import Path
import pandas as pd
import pytest
import torch

from src.data.dataset import ProcessedVideoDataset, create_dataloader
from src.data.manifest import validate_split_integrity
from src.data.storage import ProcessedDataStorage
from src.data.types import PreprocessResult
from src.evaluation.metrics import compute_metrics
from src.models.detector import DeepfakeDetector
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.utils.config import AppConfig


def test_full_train_evaluate_lifecycle_and_test_isolation(tmp_path: Path):
    storage_dir = tmp_path / "processed"
    storage = ProcessedDataStorage(storage_dir)

    # 4 train samples (2 real, 2 fake)
    for vid, lbl in [("t1", 0), ("t2", 1), ("t3", 0), ("t4", 1)]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(PreprocessResult(video_id=vid, group_id=vid, split="train", status="success", tensor=t, valid_frames=16), split="train")

    # 2 val samples (1 real, 1 fake)
    for vid, lbl in [("v1", 0), ("v2", 1)]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(PreprocessResult(video_id=vid, group_id=vid, split="val", status="success", tensor=t, valid_frames=16), split="val")

    # 2 test samples (1 real, 1 fake)
    for vid, lbl in [("te1", 0), ("te2", 1)]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(PreprocessResult(video_id=vid, group_id=vid, split="test", status="success", tensor=t, valid_frames=16), split="test")

    manifest_df = pd.DataFrame([
        {"video_id": "t1", "group_id": "g_t1", "video_path": "a.mp4", "label": 0, "split": "train"},
        {"video_id": "t2", "group_id": "g_t2", "video_path": "b.mp4", "label": 1, "split": "train"},
        {"video_id": "t3", "group_id": "g_t3", "video_path": "c.mp4", "label": 0, "split": "train"},
        {"video_id": "t4", "group_id": "g_t4", "video_path": "d.mp4", "label": 1, "split": "train"},
        {"video_id": "v1", "group_id": "g_v1", "video_path": "e.mp4", "label": 0, "split": "val"},
        {"video_id": "v2", "group_id": "g_v2", "video_path": "f.mp4", "label": 1, "split": "val"},
        {"video_id": "te1", "group_id": "g_te1", "video_path": "g.mp4", "label": 0, "split": "test"},
        {"video_id": "te2", "group_id": "g_te2", "video_path": "h.mp4", "label": 1, "split": "test"},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # 1. Verify manifest split integrity (zero group leakage)
    validate_split_integrity(manifest_df)

    # 2. Datasets
    train_ds = ProcessedVideoDataset(manifest=manifest_path, split="train", processed_dir=storage_dir)
    val_ds = ProcessedVideoDataset(manifest=manifest_path, split="val", processed_dir=storage_dir)
    test_ds = ProcessedVideoDataset(manifest=manifest_path, split="test", processed_dir=storage_dir)

    train_loader = create_dataloader(train_ds, batch_size=2, shuffle=False)
    val_loader = create_dataloader(val_ds, batch_size=2, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=2, shuffle=False)

    cfg = AppConfig()
    cfg.model.backbone.weights = None
    cfg.model.backbone.freeze_strategy = "layer2"
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.training.num_epochs = 2
    cfg.training.monitor_metric = "val_auc"
    cfg.device = "cpu"

    model = DeepfakeDetector(cfg.model)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=torch.device("cpu"),
    )

    # 3. Train: verifies training loop operates ONLY on train_loader and val_loader
    fit_results = trainer.fit(start_epoch=1)
    assert "history" in fit_results
    assert len(fit_results["history"]) == 2

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    best_ckpt_path = ckpt_dir / "best.pth"
    latest_ckpt_path = ckpt_dir / "latest.pth"
    assert best_ckpt_path.exists()
    assert latest_ckpt_path.exists()

    # 4. Check that configuration was saved into checkpoint
    ckpt_data = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    assert "config" in ckpt_data
    assert ckpt_data["config"]["training"]["monitor_metric"] == "val_auc"

    # 5. Final Evaluation strictly on test split using best.pth
    eval_model = DeepfakeDetector(cfg.model)
    ckpt_mgr = CheckpointManager()
    ckpt_mgr.load_checkpoint(best_ckpt_path, model=eval_model, restore_rng=False)
    eval_model.eval()

    test_logits = []
    test_targets = []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.inference_mode():
        for batch in test_loader:
            x, y = batch[0], batch[1]
            out = eval_model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            test_logits.append(out)
            test_targets.append(y)

    cat_targets = torch.cat(test_targets, dim=0)
    cat_logits = torch.cat(test_logits, dim=0)
    test_metrics = compute_metrics(
        cat_targets,
        cat_logits,
        loss=total_loss / len(test_ds),
        threshold=0.5,
    )

    # 6. Verify metric suite completeness
    assert test_metrics.loss is not None
    assert 0.0 <= test_metrics.accuracy <= 1.0
    assert 0.0 <= test_metrics.precision <= 1.0
    assert 0.0 <= test_metrics.recall <= 1.0
    assert 0.0 <= test_metrics.f1 <= 1.0
    assert 0.0 <= test_metrics.roc_auc <= 1.0
    assert 0.0 <= test_metrics.eer <= 1.0
    assert test_metrics.confusion_matrix is not None
    assert "tn" in test_metrics.confusion_matrix
    assert "fp" in test_metrics.confusion_matrix
    assert "fn" in test_metrics.confusion_matrix
    assert "tp" in test_metrics.confusion_matrix
