"""Orchestrator for model training, validation, AMP, and optimization."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import EvaluationMetrics, compute_metrics
from src.models.detector import DeepfakeDetector
from src.training.checkpoint import CheckpointManager
from src.training.losses import CompositeDetectionLoss, FocalLoss
from src.data.samplers import SequenceAugmentor
from src.utils.config import AppConfig
from src.utils.logger import setup_logger


class Trainer:
    """Manages the full lifecycle of deepfake detector training and validation.

    Invariants enforced:
        - Only parameters with requires_grad=True are registered in the optimizer.
        - Frozen backbone stages (stem, layer1, layer2) never receive optimizer updates.
        - BCEWithLogitsLoss / FocalLoss strictly receives targets and logits of matching (B, 1) shape.
        - Test data is strictly excluded from training, validation, and checkpoint selection.
        - Automatic CPU fallback: AMP is disabled when device is CPU.
        - Gradient accumulation strictly flushes the final partial window.
        - latest.pth is always written at epoch end; best.pth only on valid improvement.
        - Fails fast if monitor_metric='val_auc' but validation split is single-class.
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: AppConfig,
        device: Optional[torch.device] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """Initialize Trainer.

        Args:
            model: DeepfakeDetector instance.
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
            config: Validated AppConfig instance.
            device: Optional torch.device (defaults to effective device in config).
            checkpoint_manager: Optional CheckpointManager instance.
        """
        self.config = config
        self.logger = setup_logger("trainer")

        effective_dev_str = config.get_effective_device() if device is None else str(device)
        self.device = torch.device(effective_dev_str)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Precondition check: If monitor_metric is val_auc, validation split must contain both classes
        if self.config.training.monitor_metric == "val_auc":
            if hasattr(self.val_loader.dataset, "samples"):
                val_labels = [s["label"] for s in self.val_loader.dataset.samples]
                unique_labels = set(val_labels)
                if len(unique_labels) < 2:
                    raise ValueError(
                        f"Configuration requests monitor_metric='val_auc', but the validation split contains "
                        f"only labels {unique_labels}. ROC-AUC requires both positive (1) and negative (0) classes. "
                        "Please configure a class-balanced validation split or change monitor_metric to 'val_loss'."
                    )

        # Invariant: Only optimize parameters with requires_grad=True
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable_params:
            raise ValueError("Model has zero trainable parameters (all parameters are frozen).")

        opt_cfg = self.config.training.optimizer
        self.optimizer = torch.optim.Adam(
            self.trainable_params,
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )

        # Loss function setup
        loss_type = getattr(self.config.training, "loss_type", "bce")
        if loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=getattr(self.config.training, "focal_alpha", 0.25),
                gamma=getattr(self.config.training, "focal_gamma", 2.0),
            )
            self.use_composite_loss = False
        elif loss_type == "composite":
            self.criterion = CompositeDetectionLoss(
                cls_loss_type="focal",
                alpha=getattr(self.config.training, "focal_alpha", 0.25),
                gamma=getattr(self.config.training, "focal_gamma", 2.0),
                contrastive_weight=getattr(self.config.training, "contrastive_weight", 0.1),
                temperature=getattr(self.config.training, "contrastive_temperature", 0.1),
            )
            self.use_composite_loss = True
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            self.use_composite_loss = False

        # Sequence augmentation
        from src.data.augment import RobustAugmentor
        if hasattr(self.config, "data") and hasattr(self.config.data, "augmentation"):
            self.augmentor = RobustAugmentor(self.config.data.augmentation) if self.config.data.augmentation.enabled else None
        else:
            self.augmentor = None

        # Scheduler
        sched_cfg = self.config.training.scheduler
        if sched_cfg.type == "plateau":
            self.scheduler: Optional[Any] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_cfg.mode,
                factor=sched_cfg.factor,
                patience=sched_cfg.patience,
                min_lr=sched_cfg.min_lr,
            )
        elif sched_cfg.type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=sched_cfg.min_lr,
            )
        else:
            self.scheduler = None

        # AMP and device gating
        self.use_amp = self.config.training.amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.pin_memory = self.config.training.pin_memory and (self.device.type == "cuda")

        # Checkpoint manager
        mode = "min" if self.config.training.monitor_metric == "val_loss" else "max"
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            checkpoint_dir=self.config.training.checkpoint_dir,
            monitor_metric=self.config.training.monitor_metric,
            mode=mode,
        )

        self.grad_accum_steps = self.config.training.grad_accum_steps
        self.early_stopping_counter = 0

    def train_epoch(self, epoch: int) -> EvaluationMetrics:
        """Run one training epoch with gradient accumulation and AMP.

        Args:
            epoch: Current epoch index (1-based).

        Returns:
            EvaluationMetrics computed over the training epoch.
        """
        self.model.train()
        total_loss = 0.0
        all_logits: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        self.optimizer.zero_grad()
        num_batches = len(self.train_loader)

        # Update epoch on sampler if supported
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        if hasattr(self.train_loader, "batch_sampler") and hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(self.device, non_blocking=self.pin_memory)
            targets = targets.to(self.device, non_blocking=self.pin_memory)

            # Strict shape check
            if targets.shape != (inputs.shape[0], 1):
                raise ValueError(
                    f"Target batch shape must be (B, 1), got {tuple(targets.shape)} for batch size {inputs.shape[0]}"
                )

            # Apply sequence augmentation if enabled
            if self.augmentor is not None:
                augmented_batch = []
                for i in range(inputs.shape[0]):
                    aug_seq, _ = self.augmentor.augment_tensor(inputs[i])
                    augmented_batch.append(aug_seq)
                inputs = torch.stack(augmented_batch, dim=0)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                if self.use_composite_loss:
                    logits, embeddings = self.model(inputs, return_embeddings=True)
                    loss, _ = self.criterion(logits, targets, features=embeddings)
                else:
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)

                if targets.shape != logits.shape:
                    raise ValueError(
                        f"Target/logit shape mismatch: targets shape {tuple(targets.shape)} != "
                        f"logits shape {tuple(logits.shape)}. Both must be strictly (B, 1)."
                    )

                loss_scaled = loss / self.grad_accum_steps

            # Backward pass
            self.scaler.scale(loss_scaled).backward()
            total_loss += loss.item() * inputs.size(0)

            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

            # Gradient accumulation flush: step when window completes OR on final partial batch
            is_final_batch = (batch_idx == num_batches - 1)
            if ((batch_idx + 1) % self.grad_accum_steps == 0) or is_final_batch:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_params,
                    self.config.training.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        epoch_loss = total_loss / len(self.train_loader.dataset)
        cat_targets = torch.cat(all_targets, dim=0)
        cat_logits = torch.cat(all_logits, dim=0)

        return compute_metrics(cat_targets, cat_logits, loss=epoch_loss)

    def validate_epoch(self, epoch: int) -> EvaluationMetrics:
        """Run validation epoch in inference mode.

        Args:
            epoch: Current epoch index (1-based).

        Returns:
            EvaluationMetrics computed over the validation split.
        """
        self.model.eval()
        total_loss = 0.0
        all_logits: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.inference_mode():
            for batch in self.val_loader:
                inputs, targets = batch[0], batch[1]
                inputs = inputs.to(self.device, non_blocking=self.pin_memory)
                targets = targets.to(self.device, non_blocking=self.pin_memory)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(inputs)

                    if targets.shape != logits.shape:
                        raise ValueError(
                            f"Target/logit shape mismatch in validation: targets={tuple(targets.shape)} != "
                            f"logits={tuple(logits.shape)}"
                        )

                    if self.use_composite_loss:
                        loss = self.criterion.cls_loss(logits, targets)
                    else:
                        loss = self.criterion(logits, targets)

                total_loss += loss.item() * inputs.size(0)
                all_logits.append(logits.detach().cpu())
                all_targets.append(targets.detach().cpu())

        val_loss = total_loss / len(self.val_loader.dataset)
        cat_targets = torch.cat(all_targets, dim=0)
        cat_logits = torch.cat(all_logits, dim=0)

        return compute_metrics(cat_targets, cat_logits, loss=val_loss)

    def fit(self, start_epoch: int = 1) -> Dict[str, Any]:
        """Execute full training and validation loop.

        Args:
            start_epoch: Starting epoch index (default: 1, or > 1 when resuming).

        Returns:
            Dictionary containing training history, best metrics, and peak VRAM.
        """
        num_epochs = self.config.training.num_epochs
        history: List[Dict[str, Any]] = []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        self.logger.info(
            f"Starting training on {self.device} (AMP={self.use_amp}) for {num_epochs - start_epoch + 1} epochs..."
        )

        for epoch in range(start_epoch, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            # Step scheduler
            sched_cfg = self.config.training.scheduler
            if sched_cfg.type == "plateau" and self.scheduler is not None:
                # Use monitored metric
                val_dict = val_metrics.to_dict()
                metric_key = self.config.training.monitor_metric
                if metric_key.startswith("val_"):
                    metric_key = metric_key[4:]
                step_val = val_dict.get(metric_key, val_metrics.loss)
                self.scheduler.step(step_val)
            elif sched_cfg.type == "cosine" and self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoints
            latest_path, best_path, is_best = self.checkpoint_manager.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                metrics=val_metrics,
                config=self.config,
                early_stopping_counter=self.early_stopping_counter,
            )

            if is_best:
                self.early_stopping_counter = 0
                best_indicator = " [BEST]"
            else:
                self.early_stopping_counter += 1
                best_indicator = ""

            self.logger.info(
                f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_metrics.loss:.4f}, Acc: {train_metrics.accuracy:.4f} | "
                f"Val Loss: {val_metrics.loss:.4f}, Acc: {val_metrics.accuracy:.4f}, AUC: {val_metrics.roc_auc:.4f}, "
                f"EER: {val_metrics.eer:.4f}{best_indicator}"
            )

            history.append({
                "epoch": epoch,
                "train_metrics": train_metrics.to_dict(),
                "val_metrics": val_metrics.to_dict(),
                "is_best": is_best,
            })

            # Check early stopping
            if self.early_stopping_counter >= self.config.training.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {self.early_stopping_counter} epochs without improvement."
                )
                break

        peak_vram_mb = 0.0
        if self.device.type == "cuda":
            peak_vram_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))
            peak_vram_gb = peak_vram_mb / 1024.0
            self.logger.info(
                f"Peak VRAM during training: {peak_vram_mb:.2f} MB ({peak_vram_gb:.3f} GB / 5.5 GB limit)"
            )
            assert peak_vram_gb <= 5.5, f"Peak VRAM exceeded limit: {peak_vram_gb:.2f} GB > 5.5 GB"

        return {
            "history": history,
            "best_metric": self.checkpoint_manager.best_metric,
            "best_epoch": self.checkpoint_manager.best_epoch,
            "peak_vram_mb": peak_vram_mb,
        }
