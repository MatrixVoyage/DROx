"""Experiment orchestration and lifecycle execution module.

Manages reproducible end-to-end experiment runs:
1. Environment and hardware provenance tracking (Git, PyTorch, GPU, seeds).
2. Experiment directory setup under runs/{exp_name}_{timestamp}/.
3. Training loop execution via Trainer with model selection strictly on validation set.
4. Evaluation of best.pth on test split.
5. Structured experiment artifact output (summary JSON, config YAML, metric history).
"""

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Union
import torch
import yaml

from src.data.dataset import ProcessedVideoDataset, create_dataloader
from src.data.manifest import load_manifest, validate_split_integrity
from src.evaluation.metrics import EvaluationMetrics, compute_metrics
from src.models.detector import DeepfakeDetector
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.utils.config import AppConfig, load_config
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything


def collect_provenance(seed: int, device: torch.device) -> Dict[str, Any]:
    """Capture environmental and software provenance for experiment reproducibility.

    Args:
        seed: Random seed used for run.
        device: PyTorch device.

    Returns:
        Dictionary detailing git, runtime, hardware, and library versions.
    """
    # Git commit hash
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        pass

    # GPU info
    gpu_info = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(device),
            "total_vram_mb": round(torch.cuda.get_device_properties(device).total_memory / (1024 ** 2), 2),
            "cuda_version": torch.version.cuda,
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_hash,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": sys.platform,
        "device": str(device),
        "gpu": gpu_info,
    }


class ExperimentRunner:
    """Orchestrates an isolated, reproducible deepfake detection training & evaluation experiment."""

    def __init__(
        self,
        config: Union[AppConfig, str, Path],
        manifest_path: Union[str, Path],
        output_root: Union[str, Path] = "runs",
        experiment_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize ExperimentRunner.

        Args:
            config: AppConfig object or path to YAML configuration.
            manifest_path: Path to dataset manifest CSV/JSONL.
            output_root: Root directory where experiment run folders are created.
            experiment_name: Descriptive name for run. If None, derived from config.
            device: Device override ("cuda" or "cpu").
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
            self.config_path = str(config)
        else:
            self.config = config
            self.config_path = "in_memory"

        if device:
            self.config.device = device

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {self.manifest_path}")

        # Derive experiment name
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            bb = self.config.model.backbone.name
            attn = self.config.model.attention.type
            temp = self.config.model.temporal.type
            self.experiment_name = f"{bb}_{attn}_{temp}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_root) / f"{self.experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.training.checkpoint_dir = str(self.checkpoint_dir)

        # Logger
        self.logger = setup_logger(
            f"exp_{self.experiment_name}",
            log_file=self.run_dir / "experiment.log",
        )

        # Device determination
        self.device = torch.device(self.config.get_effective_device())

    def run(self) -> Dict[str, Any]:
        """Execute full experiment lifecycle: train, validation, model selection, and test evaluation.

        Returns:
            Dictionary containing comprehensive summary and metrics.
        """
        start_time = time.time()
        self.logger.info(f"=== Starting Experiment: {self.experiment_name} ===")
        self.logger.info(f"Run directory: {self.run_dir.resolve()}")
        self.logger.info(f"Device: {self.device}")

        # Set seed
        seed_everything(self.config.seed)

        # Provenance collection
        prov = collect_provenance(self.config.seed, self.device)
        self.logger.info(f"Git commit: {prov['git_commit']} | Seed: {prov['seed']}")

        # Save active configuration snapshot in run dir
        config_snapshot_path = self.run_dir / "config.yaml"
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False, sort_keys=False)

        # Verify manifest split integrity
        manifest_df = load_manifest(self.manifest_path)
        validate_split_integrity(manifest_df)

        has_train = (manifest_df["split"] == "train").any()
        has_val = (manifest_df["split"] == "val").any()
        has_test = (manifest_df["split"] == "test").any()

        if not (has_train and has_val):
            raise ValueError(
                f"Manifest must contain both 'train' and 'val' splits for experiment run. Found: {manifest_df['split'].unique()}"
            )

        # Build datasets
        train_ds = ProcessedVideoDataset(
            manifest=self.manifest_path,
            split="train",
            processed_dir=self.config.storage.processed_dir,
            expected_num_frames=self.config.data.sampling.num_frames,
        )
        val_ds = ProcessedVideoDataset(
            manifest=self.manifest_path,
            split="val",
            processed_dir=self.config.storage.processed_dir,
            expected_num_frames=self.config.data.sampling.num_frames,
        )

        self.logger.info(f"Dataset loaded: {len(train_ds)} train samples, {len(val_ds)} val samples")

        train_loader = create_dataloader(
            train_ds,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            seed=self.config.seed,
        )
        val_loader = create_dataloader(
            val_ds,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            seed=self.config.seed,
        )

        # Build Model
        model = DeepfakeDetector(self.config.model)

        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            monitor_metric=self.config.training.monitor_metric,
            mode="min" if self.config.training.monitor_metric == "val_loss" else "max",
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            checkpoint_manager=checkpoint_mgr,
            device=self.device,
        )

        # Train model
        fit_results = trainer.fit(start_epoch=1)
        self.logger.info(
            f"Training finished: Best Epoch={fit_results['best_epoch']} "
            f"| Best {self.config.training.monitor_metric}={fit_results['best_metric']:.4f}"
        )

        # Test evaluation (if test split exists) using best.pth strictly
        test_metrics_dict: Optional[Dict[str, Any]] = None
        best_ckpt_path = self.checkpoint_dir / "best.pth"

        if has_test and best_ckpt_path.exists():
            self.logger.info(f"Evaluating best checkpoint ({best_ckpt_path.name}) on test split...")
            test_ds = ProcessedVideoDataset(
                manifest=self.manifest_path,
                split="test",
                processed_dir=self.config.storage.processed_dir,
                expected_num_frames=self.config.data.sampling.num_frames,
            )
            test_loader = create_dataloader(
                test_ds,
                batch_size=self.config.evaluation.batch_size,
                shuffle=False,
                num_workers=self.config.training.num_workers,
                pin_memory=self.config.training.pin_memory,
                seed=self.config.seed,
            )

            eval_model = DeepfakeDetector(self.config.model)
            checkpoint_mgr.load_checkpoint(best_ckpt_path, model=eval_model, restore_rng=False)
            eval_model.to(self.device)
            eval_model.eval()

            test_logits = []
            test_targets = []
            total_loss = 0.0
            criterion = torch.nn.BCEWithLogitsLoss()

            with torch.inference_mode():
                for batch in test_loader:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    out = eval_model(x)
                    loss = criterion(out, y)
                    total_loss += loss.item() * x.size(0)
                    test_logits.append(out.cpu())
                    test_targets.append(y.cpu())

            cat_targets = torch.cat(test_targets, dim=0)
            cat_logits = torch.cat(test_logits, dim=0)
            test_metrics = compute_metrics(
                cat_targets,
                cat_logits,
                loss=total_loss / len(test_ds),
                threshold=self.config.evaluation.threshold,
            )
            test_metrics_dict = asdict(test_metrics)
            self.logger.info("=== Test Split Results ===")
            self.logger.info(test_metrics.summary_string())

        total_runtime = round(time.time() - start_time, 2)

        summary = {
            "experiment_name": self.experiment_name,
            "run_dir": str(self.run_dir.resolve()),
            "provenance": prov,
            "config": self.config.to_dict(),
            "best_epoch": fit_results["best_epoch"],
            "best_metric": fit_results["best_metric"],
            "monitor_metric": self.config.training.monitor_metric,
            "history": fit_results["history"],
            "test_metrics": test_metrics_dict,
            "runtime_seconds": total_runtime,
        }

        # Save summary JSON
        summary_path = self.run_dir / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Experiment summary saved: {summary_path}")
        return summary
