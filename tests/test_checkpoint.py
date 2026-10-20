"""Unit tests for CheckpointManager, atomic persistence, and resume restoration."""

from pathlib import Path
import random
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation.metrics import EvaluationMetrics
from src.models.detector import DeepfakeDetector
from src.training.checkpoint import CheckpointManager
from src.utils.config import AppConfig, ModelConfig, TemporalConfig, ClassifierConfig


def _create_dummy_metrics(val_auc: float, loss: float = 0.5) -> EvaluationMetrics:
    return EvaluationMetrics(
        loss=loss,
        accuracy=0.8,
        precision=0.8,
        recall=0.8,
        f1=0.8,
        roc_auc=val_auc,
        eer=0.2 if not np.isnan(val_auc) else float("nan"),
        confusion_matrix={"tp": 4, "fp": 1, "tn": 4, "fn": 1},
        num_samples=10,
    )


def test_checkpoint_atomic_save_and_best_tracking(tmp_path: Path):
    """Verify latest.pth is always written, and best.pth updates strictly on improvement."""
    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir=ckpt_dir, monitor_metric="val_auc", mode="max")
    config = AppConfig()

    # Small linear model for fast test
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Epoch 1: val_auc = 0.80 -> should update latest and best
    m1 = _create_dummy_metrics(val_auc=0.80)
    latest_p, best_p, is_best1 = manager.save(
        epoch=1, model=model, optimizer=optimizer, scheduler=None, scaler=None,
        metrics=m1, config=config, early_stopping_counter=0
    )
    assert is_best1 is True
    assert latest_p.exists()
    assert best_p is not None and best_p.exists()
    assert manager.best_metric == 0.80
    assert manager.best_epoch == 1

    # Epoch 2: val_auc = 0.75 (worse) -> latest updated, best untouched
    m2 = _create_dummy_metrics(val_auc=0.75)
    latest_p, best_p2, is_best2 = manager.save(
        epoch=2, model=model, optimizer=optimizer, scheduler=None, scaler=None,
        metrics=m2, config=config, early_stopping_counter=1
    )
    assert is_best2 is False
    assert best_p2 is None
    assert manager.best_metric == 0.80
    assert manager.best_epoch == 1

    # Verify best.pth still contains epoch 1
    cp_best = torch.load(str(ckpt_dir / "best.pth"), weights_only=False)
    assert cp_best["epoch"] == 1
    assert cp_best["best_metric"] == 0.80

    # Epoch 3: val_auc = 0.90 (better) -> best updated to epoch 3
    m3 = _create_dummy_metrics(val_auc=0.90)
    latest_p, best_p3, is_best3 = manager.save(
        epoch=3, model=model, optimizer=optimizer, scheduler=None, scaler=None,
        metrics=m3, config=config, early_stopping_counter=0
    )
    assert is_best3 is True
    assert best_p3 is not None
    assert manager.best_metric == 0.90
    assert manager.best_epoch == 3

    cp_best3 = torch.load(str(ckpt_dir / "best.pth"), weights_only=False)
    assert cp_best3["epoch"] == 3
    assert cp_best3["best_metric"] == 0.90


def test_checkpoint_nan_metric_never_replaces_best(tmp_path: Path):
    """Verify NaN monitored metrics never replace best.pth."""
    ckpt_dir = tmp_path / "checkpoints_nan"
    manager = CheckpointManager(checkpoint_dir=ckpt_dir, monitor_metric="val_auc", mode="max")
    config = AppConfig()

    model = nn.Linear(5, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # First save with valid metric
    m1 = _create_dummy_metrics(val_auc=0.70)
    manager.save(
        epoch=1, model=model, optimizer=optimizer, scheduler=None, scaler=None,
        metrics=m1, config=config
    )
    assert manager.best_metric == 0.70

    # Second save with NaN metric (e.g. single-class val split)
    m_nan = _create_dummy_metrics(val_auc=float("nan"))
    latest_p, best_p, is_best = manager.save(
        epoch=2, model=model, optimizer=optimizer, scheduler=None, scaler=None,
        metrics=m_nan, config=config, early_stopping_counter=1
    )
    assert is_best is False
    assert best_p is None
    assert manager.best_metric == 0.70
    assert manager.best_epoch == 1


def test_checkpoint_resume_restores_state_and_rng(tmp_path: Path):
    """Verify full state restoration of weights, optimizer, and RNG state."""
    ckpt_dir = tmp_path / "checkpoints_resume"
    manager = CheckpointManager(checkpoint_dir=ckpt_dir, monitor_metric="val_auc", mode="max")
    config = AppConfig()

    torch.manual_seed(1234)
    model1 = nn.Linear(4, 1)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)

    # Run one optimizer step so optimizer state is populated
    out = model1(torch.randn(2, 4))
    loss = out.sum()
    loss.backward()
    optimizer1.step()

    m = _create_dummy_metrics(val_auc=0.88)
    latest_p, _, _ = manager.save(
        epoch=5, model=model1, optimizer=optimizer1, scheduler=None, scaler=None,
        metrics=m, config=config, early_stopping_counter=2
    )

    # Instantiate model2 with different weights
    model2 = nn.Linear(4, 1)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

    # Weights should differ initially
    assert not torch.allclose(model1.weight, model2.weight)

    # Restore from latest_p
    resumed_manager = CheckpointManager(checkpoint_dir=ckpt_dir, monitor_metric="val_auc", mode="max")
    cp_dict = resumed_manager.load_checkpoint(
        latest_p, model=model2, optimizer=optimizer2, restore_rng=True
    )

    # Verify model weights restored exactly
    assert torch.allclose(model1.weight, model2.weight)
    assert torch.allclose(model1.bias, model2.bias)

    # Verify epoch, metric, and early stopping state restored
    assert cp_dict["epoch"] == 5
    assert cp_dict["best_metric"] == 0.88
    assert cp_dict["early_stopping_counter"] == 2
    assert cp_dict["config"] is not None
    assert resumed_manager.best_metric == 0.88
