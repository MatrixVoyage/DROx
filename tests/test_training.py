"""Unit and integration tests for Trainer, optimizer parameter isolation, and training loop."""

from pathlib import Path
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import ProcessedVideoDataset, create_dataloader
from src.data.storage import ProcessedDataStorage
from src.data.types import PreprocessResult
from src.models.detector import DeepfakeDetector
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.utils.config import AppConfig


def test_optimizer_only_includes_trainable_parameters():
    """Verify optimizer filters out frozen backbone parameters."""
    cfg = AppConfig()
    cfg.model.backbone.weights = None
    cfg.model.backbone.freeze_strategy = "layer2"
    model = DeepfakeDetector(cfg.model)

    dummy_loader = DataLoader(TensorDataset(torch.randn(2, 16, 3, 112, 112), torch.tensor([[0.0], [1.0]])), batch_size=2)
    trainer = Trainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        config=cfg,
        device=torch.device("cpu"),
    )

    optimizer_params = set(trainer.optimizer.param_groups[0]["params"])

    # Check frozen parameters are excluded
    for stage in [model.backbone.stem, model.backbone.layer1, model.backbone.layer2]:
        for p in stage.parameters():
            assert not p.requires_grad
            assert p not in optimizer_params, "Frozen parameter must not be in optimizer!"

    # Check trainable parameters are included
    for stage in [model.backbone.layer3, model.backbone.layer4, model.attention, model.temporal, model.classifier]:
        for p in stage.parameters():
            assert p.requires_grad
            assert p in optimizer_params, "Trainable parameter must be in optimizer!"


def test_trainer_val_auc_precondition_guard(tmp_path: Path):
    """Verify Trainer raises ValueError when monitor_metric='val_auc' on a single-class validation set."""
    storage_dir = tmp_path / "processed"
    storage = ProcessedDataStorage(storage_dir)

    # Train split: balanced
    for vid, lbl in [("t1", 0), ("t2", 1)]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(PreprocessResult(video_id=vid, group_id=vid, split="train", status="success", tensor=t, valid_frames=16), split="train")

    # Val split: single-class (all 0s)
    for vid, lbl in [("v1", 0), ("v2", 0)]:
        t = torch.randn(16, 3, 112, 112)
        storage.save(PreprocessResult(video_id=vid, group_id=vid, split="val", status="success", tensor=t, valid_frames=16), split="val")

    manifest_df = pd.DataFrame([
        {"video_id": "t1", "group_id": "t1", "video_path": "a.mp4", "label": 0, "split": "train"},
        {"video_id": "t2", "group_id": "t2", "video_path": "b.mp4", "label": 1, "split": "train"},
        {"video_id": "v1", "group_id": "v1", "video_path": "c.mp4", "label": 0, "split": "val"},
        {"video_id": "v2", "group_id": "v2", "video_path": "d.mp4", "label": 0, "split": "val"},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    train_ds = ProcessedVideoDataset(manifest=manifest_path, split="train", processed_dir=storage_dir)
    val_ds = ProcessedVideoDataset(manifest=manifest_path, split="val", processed_dir=storage_dir)

    train_loader = create_dataloader(train_ds, batch_size=2)
    val_loader = create_dataloader(val_ds, batch_size=2)

    cfg = AppConfig()
    cfg.model.backbone.weights = None
    cfg.training.monitor_metric = "val_auc"
    model = DeepfakeDetector(cfg.model)

    with pytest.raises(ValueError, match="monitor_metric='val_auc', but the validation split contains only labels"):
        Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=cfg, device=torch.device("cpu"))


def test_trainer_mini_training_loop_and_resumption(tmp_path: Path):
    """Verify full end-to-end training loop: parameter updates, frozen stability, checkpointing, and resume."""
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



    manifest_df = pd.DataFrame([
        {"video_id": "t1", "group_id": "t1", "video_path": "a.mp4", "label": 0, "split": "train"},
        {"video_id": "t2", "group_id": "t2", "video_path": "b.mp4", "label": 1, "split": "train"},
        {"video_id": "t3", "group_id": "t3", "video_path": "c.mp4", "label": 0, "split": "train"},
        {"video_id": "t4", "group_id": "t4", "video_path": "d.mp4", "label": 1, "split": "train"},
        {"video_id": "v1", "group_id": "v1", "video_path": "e.mp4", "label": 0, "split": "val"},
        {"video_id": "v2", "group_id": "v2", "video_path": "f.mp4", "label": 1, "split": "val"},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    train_ds = ProcessedVideoDataset(manifest=manifest_path, split="train", processed_dir=storage_dir)
    val_ds = ProcessedVideoDataset(manifest=manifest_path, split="val", processed_dir=storage_dir)

    train_loader = create_dataloader(train_ds, batch_size=2, shuffle=False)
    val_loader = create_dataloader(val_ds, batch_size=2, shuffle=False)

    cfg = AppConfig()
    cfg.model.backbone.weights = None  # fast offline initialization
    cfg.model.backbone.freeze_strategy = "layer2"
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.training.num_epochs = 2
    cfg.training.grad_accum_steps = 2
    cfg.training.monitor_metric = "val_loss"
    cfg.training.scheduler.mode = "min"
    cfg.device = "cpu"

    model = DeepfakeDetector(cfg.model)

    # Clone initial parameter values to verify frozen vs trainable behavior
    initial_frozen_val = model.backbone.layer1[0].conv1.weight.clone()
    initial_trainable_val = model.classifier.net[0].weight.clone()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=torch.device("cpu"),
    )

    results = trainer.fit(start_epoch=1)
    assert len(results["history"]) == 2

    # Invariant: Frozen parameters remain exactly unchanged
    assert torch.equal(model.backbone.layer1[0].conv1.weight, initial_frozen_val)

    # Invariant: Trainable parameters demonstrably update
    assert not torch.equal(model.classifier.net[0].weight, initial_trainable_val)

    # Invariant: Checkpoints latest.pth and best.pth were created
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    assert (ckpt_dir / "latest.pth").exists()
    assert (ckpt_dir / "best.pth").exists()

    # Invariant: Verify resumption restores state
    model_resume = DeepfakeDetector(cfg.model)
    trainer_resume = Trainer(
        model=model_resume,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=torch.device("cpu"),
    )
    cp = trainer_resume.checkpoint_manager.load_checkpoint(
        ckpt_dir / "latest.pth",
        model=model_resume,
        optimizer=trainer_resume.optimizer,
        restore_rng=True,
    )
    assert cp["epoch"] == 2
    assert torch.allclose(model.classifier.net[0].weight, model_resume.classifier.net[0].weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP test")
def test_cuda_amp_training_step(tmp_path: Path):
    """Verify CUDA AMP training step executes without error when CUDA is available."""
    cfg = AppConfig()
    cfg.model.backbone.weights = None
    cfg.model.backbone.freeze_strategy = "layer2"
    cfg.training.amp = True
    cfg.device = "cuda"

    model = DeepfakeDetector(cfg.model).cuda()
    inputs = torch.randn(2, 16, 3, 112, 112, device="cuda")
    targets = torch.tensor([[0.0], [1.0]], device="cuda")

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=cfg,
        device=torch.device("cuda"),
    )
    metrics = trainer.train_epoch(epoch=1)
    assert metrics.loss is not None
    assert not torch.isnan(torch.tensor(metrics.loss))


def test_gradient_accumulation_flushes_partial_remainder():
    """Verify gradient accumulation flushes partial window on final batch."""
    cfg = AppConfig()
    cfg.model.backbone.weights = None
    cfg.model.backbone.freeze_strategy = "layer2"
    cfg.training.grad_accum_steps = 2
    cfg.training.amp = False
    cfg.device = "cpu"

    model = DeepfakeDetector(cfg.model)

    # 3 batches of size 2 -> 6 samples total
    inputs = torch.randn(6, 16, 3, 112, 112)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=cfg,
        device=torch.device("cpu"),
    )

    step_count = 0
    original_step = trainer.optimizer.step

    def mock_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = mock_step

    # Run one epoch: batch 0 (no step), batch 1 (step 1), batch 2 (step 2 flushed!)
    trainer.train_epoch(epoch=1)

    assert step_count == 2, f"Expected 2 optimizer steps for 3 batches with grad_accum=2, got {step_count}"

