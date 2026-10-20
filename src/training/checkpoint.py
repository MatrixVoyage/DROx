"""Atomic checkpoint management and resumption for reproducible training."""

import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from src.evaluation.metrics import EvaluationMetrics
from src.utils.config import AppConfig


class CheckpointManager:
    """Manages atomic saving, tracking, and resumption of latest.pth and best.pth checkpoints.

    Invariants enforced:
        - Atomic file persistence using temporary file and os.replace.
        - latest.pth is always saved at the end of each epoch.
        - best.pth is updated strictly when the monitored metric improves.
        - NaN monitored metrics never replace best.pth.
        - Captures complete model, optimizer, scheduler, scaler, early-stopping, and RNG states.
        - Restores exact training and RNG state on resume.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        monitor_metric: str = "val_auc",
        mode: str = "max",
    ):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoints are stored.
            monitor_metric: Metric to monitor for best checkpoint (e.g. 'val_auc', 'val_loss').
            mode: Optimization direction ('min' or 'max').
        """
        valid_modes = ("min", "max")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: '{mode}'. Must be 'min' or 'max'.")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_metric = monitor_metric
        self.mode = mode

        self.best_metric: float = -float("inf") if mode == "max" else float("inf")
        self.best_epoch: int = -1

    def _atomic_torch_save(self, destination: Path, payload: Dict[str, Any]) -> None:
        """Atomically persist payload dictionary to destination path."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=str(destination.parent),
            prefix=f".tmp_{destination.stem}_",
            suffix=".tmp",
        )
        os.close(fd)
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, str(destination))
        except Exception:
            import gc
            gc.collect()
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: Optional[Any],
        metrics: EvaluationMetrics,
        config: AppConfig,
        early_stopping_counter: int = 0,
    ) -> Tuple[Path, Optional[Path], bool]:
        """Atomically save checkpoint states for the current epoch.

        Args:
            epoch: Current completed epoch index (1-based).
            model: DeepfakeDetector model instance.
            optimizer: PyTorch optimizer instance.
            scheduler: Learning rate scheduler instance (optional).
            scaler: GradScaler instance for AMP (optional).
            metrics: EvaluationMetrics computed on the validation split.
            config: Complete run configuration.
            early_stopping_counter: Current early stopping counter value.

        Returns:
            Tuple of (latest_path, best_path if is_best else None, is_best).
        """
        metrics_dict = metrics.to_dict()

        # Extract monitored value with alias mapping
        metric_aliases = {
            "val_auc": "roc_auc",
            "auc": "roc_auc",
            "val_acc": "accuracy",
            "acc": "accuracy",
            "val_loss": "loss",
            "val_f1": "f1",
            "val_eer": "eer",
        }
        target_key = metric_aliases.get(self.monitor_metric, self.monitor_metric)
        if target_key.startswith("val_"):
            target_key = target_key[4:]
        val = metrics_dict.get(target_key, metrics_dict.get(self.monitor_metric))


        # Check for improvement
        is_best = False
        if val is not None and not np.isnan(val):
            val_f = float(val)
            if self.mode == "max" and val_f > self.best_metric:
                is_best = True
                self.best_metric = val_f
                self.best_epoch = epoch
            elif self.mode == "min" and val_f < self.best_metric:
                is_best = True
                self.best_metric = val_f
                self.best_epoch = epoch

        # Capture complete RNG state for exact reproducibility
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "early_stopping_counter": early_stopping_counter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None,
            "monitor_metric": self.monitor_metric,
            "mode": self.mode,
            "current_metrics": metrics_dict,
            "config": config.to_dict(),
            "rng_state": rng_state,
        }


        # 1. Save latest.pth always
        latest_path = self.checkpoint_dir / "latest.pth"
        self._atomic_torch_save(latest_path, payload)

        # 2. Save best.pth only if metric improved and is valid
        best_path: Optional[Path] = None
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            self._atomic_torch_save(best_path, payload)

        return latest_path, best_path, is_best

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        """Restore model, optimizer, scheduler, scaler, and RNG state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint .pth file.
            model: Model to load state dict into.
            optimizer: Optional optimizer to restore state dict into.
            scheduler: Optional scheduler to restore state dict into.
            scaler: Optional GradScaler to restore state dict into.
            restore_rng: Whether to restore Python, NumPy, and PyTorch RNG states.

        Returns:
            Complete checkpoint payload dictionary.
        """
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")

        checkpoint = torch.load(str(p), weights_only=False, map_location="cpu")

        # Restore model
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint '{p}' does not contain 'model_state_dict'")
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore scaler
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore best metric tracking
        if "best_metric" in checkpoint:
            self.best_metric = float(checkpoint["best_metric"])
        if "best_epoch" in checkpoint:
            self.best_epoch = int(checkpoint["best_epoch"])

        # Restore RNG states
        if restore_rng and checkpoint.get("rng_state") is not None:
            rng = checkpoint["rng_state"]
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])
            if "numpy" in rng and rng["numpy"] is not None:
                np.random.set_state(rng["numpy"])
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"])
            if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])

        return checkpoint
