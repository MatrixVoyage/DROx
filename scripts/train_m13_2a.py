"""Training script for Milestone 13.2a: Penultimate Auxiliary Manipulation Head.

Executes two controlled experimental ablations:
1. Ablation 1 (M13.2a-Frozen):
   - M12 backbone and temporal Transformer trunk are completely frozen.
   - Only the AuxiliaryManipulationHead is trained on seen training classes.
   - Strictly isolates whether the existing 1280-D frozen representation can be
     decoded into an out-of-family detector without representation drift.
   - Checkpoint: checkpoints/m13_2a_frozen/best.pth

2. Ablation 2 (M13.2a-Joint):
   - Initializes from M12 checkpoint.
   - Backbone spatial stages remain frozen per M12 policy.
   - Temporal Transformer, primary classifier, and auxiliary head are jointly trained
     with multi-task loss: L_bce + lambda_aux * L_ce.
   - Tests whether multi-family gradient feedback aligns representations for generalization.
   - Checkpoint: checkpoints/m13_2a_joint/best.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProcessedVideoDataset, create_dataloader
from src.data.samplers import BalancedDatasetBatchSampler
from src.models.detector import DeepfakeDetector
from src.models.multi_head_detector import MultiHeadDeepfakeDetector
from src.utils.config import AppConfig, load_config
from src.utils.logger import setup_logger

logger = setup_logger("m13_2a_train")

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "m13_2a_aux_penultimate.yaml"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
PROCESSED_DIR = PROJECT_ROOT / "processed" / "m12_full"
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
FREEZE_MANIFEST_PATH = PROJECT_ROOT / "reports" / "m12_baseline_freeze.json"


def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_m12_integrity() -> str:
    """Verify that M12 baseline best.pth has not been modified."""
    with open(FREEZE_MANIFEST_PATH, "r") as f:
        freeze_data = json.load(f)
    expected_hash = freeze_data["fingerprints"]["m12_checkpoint"]["sha256"]
    current_hash = sha256_file(M12_CHECKPOINT_PATH)
    assert current_hash == expected_hash, f"CRITICAL: M12 checkpoint modified! {current_hash} != {expected_hash}"
    logger.info(f"M12 integrity verified. Hash: {current_hash[:16]}...")
    return current_hash


def filter_manifest_to_existing(manifest_df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    """Filter manifest for train and val to only include rows that have preprocessed tensors."""
    valid_rows = []
    skipped_count = 0
    for _, row in manifest_df.iterrows():
        split = str(row["split"])
        vid = str(row["video_id"])
        if split in ("train", "val"):
            tensor_path = processed_dir / split / "tensors" / f"{vid}.pt"
            meta_path = processed_dir / split / "metadata" / f"{vid}.json"
            if tensor_path.exists() and meta_path.exists():
                valid_rows.append(row)
            else:
                skipped_count += 1
        else:
            valid_rows.append(row)

    if skipped_count > 0:
        logger.warning(f"Filtered out {skipped_count} manifest rows whose tensors were missing.")
    return pd.DataFrame(valid_rows)


def train_ablation_frozen(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: AppConfig,
    device: torch.device,
    epochs: int = 15,
) -> Path:
    """Train Ablation 1: Frozen M12 Trunk + Auxiliary Manipulation Head."""
    logger.info("=" * 70)
    logger.info("Starting Ablation 1: M13.2a-Frozen (Frozen Trunk + Auxiliary Head)")
    logger.info("=" * 70)

    # 1. Load M12 base detector
    ckpt = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    m12_config = ckpt["config"]
    if hasattr(m12_config, "model"):
        base_model_cfg = m12_config.model
    else:
        base_model_cfg = config.model

    base_detector = DeepfakeDetector(config=base_model_cfg).to(device)
    base_detector.load_state_dict(ckpt["model_state_dict"])
    base_detector.eval()

    # 2. Build MultiHead model and freeze base trunk
    model = MultiHeadDeepfakeDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=config.model.auxiliary.hidden_dim,
        aux_dropout=config.model.auxiliary.dropout,
    ).to(device)
    model.freeze_trunk()

    # Optimizer: only aux_head parameters
    trainable_params = [p for p in model.aux_head.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters in aux_head: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=0.001,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )
    criterion_ce = nn.CrossEntropyLoss()

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "m13_2a_frozen"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pth"

    best_val_auc = -1.0

    for epoch in range(1, epochs + 1):
        model.aux_head.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            tensors, targets, vids, family_targets = batch
            tensors = tensors.to(device, dtype=torch.float32)
            family_targets = family_targets.to(device)

            with torch.no_grad():
                # Extract temporal features through frozen trunk
                _, temporal_features = model.base_detector(tensors, return_embeddings=True)

            aux_logits = model.aux_head(temporal_features)
            loss = criterion_ce(aux_logits, family_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

        avg_train_loss = total_loss / max(1, total_samples)

        # Validation
        model.aux_head.eval()
        val_targets_list = []
        val_p_fake_list = []
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                tensors, targets, vids, family_targets = batch
                tensors = tensors.to(device, dtype=torch.float32)
                family_targets = family_targets.to(device)

                _, temporal_features = model.base_detector(tensors, return_embeddings=True)
                aux_logits = model.aux_head(temporal_features)
                loss = criterion_ce(aux_logits, family_targets)

                p_fake = model.aux_head.compute_fake_probability(aux_logits).squeeze(-1)

                val_loss += loss.item() * len(targets)
                val_samples += len(targets)

                val_targets_list.extend(targets.squeeze(-1).cpu().numpy())
                val_p_fake_list.extend(p_fake.cpu().numpy())

        avg_val_loss = val_loss / max(1, val_samples)
        val_auc = roc_auc_score(val_targets_list, val_p_fake_list)
        val_pred = (np.array(val_p_fake_list) >= 0.5).astype(int)
        val_f1 = f1_score(val_targets_list, val_pred)

        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC (1-P(Real)): {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f} | LR: {lr:.6f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_auc": best_val_auc,
                    "config": config,
                    "ablation": "m13_2a_frozen",
                },
                best_ckpt_path,
            )
            logger.info(f"  -> Saved new best checkpoint (Val AUC: {best_val_auc:.4f}) to {best_ckpt_path}")

    logger.info(f"Ablation 1 training completed. Best Val AUC: {best_val_auc:.4f}")
    return best_ckpt_path


def train_ablation_joint(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: AppConfig,
    device: torch.device,
    epochs: int = 15,
) -> Path:
    """Train Ablation 2: Joint Multi-Task Fine-Tuning (Temporal Trunk + Heads)."""
    logger.info("=" * 70)
    logger.info("Starting Ablation 2: M13.2a-Joint (Multi-Task Fine-Tuning)")
    logger.info("=" * 70)

    # 1. Load M12 base detector
    ckpt = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    m12_config = ckpt["config"]
    if hasattr(m12_config, "model"):
        base_model_cfg = m12_config.model
    else:
        base_model_cfg = config.model

    base_detector = DeepfakeDetector(config=base_model_cfg).to(device)
    base_detector.load_state_dict(ckpt["model_state_dict"])

    # 2. Build MultiHead model and configure trainable stages
    model = MultiHeadDeepfakeDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=config.model.auxiliary.hidden_dim,
        aux_dropout=config.model.auxiliary.dropout,
    ).to(device)
    model.unfreeze_temporal_and_heads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters (Temporal + Heads): {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=0.0001,  # Lower LR for fine-tuning
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()
    lambda_aux = float(config.model.auxiliary.loss_weight)

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "m13_2a_joint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pth"

    best_val_auc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            tensors, targets, vids, family_targets = batch
            tensors = tensors.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            family_targets = family_targets.to(device)

            prim_logits, aux_logits = model(tensors)

            loss_bce = criterion_bce(prim_logits, targets)
            loss_ce = criterion_ce(aux_logits, family_targets)
            loss = loss_bce + lambda_aux * loss_ce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

        avg_train_loss = total_loss / max(1, total_samples)

        # Validation
        model.eval()
        val_targets_list = []
        val_p_primary_list = []
        val_p_fake_list = []
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                tensors, targets, vids, family_targets = batch
                tensors = tensors.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                family_targets = family_targets.to(device)

                prim_logits, aux_logits = model(tensors)
                loss_bce = criterion_bce(prim_logits, targets)
                loss_ce = criterion_ce(aux_logits, family_targets)
                loss = loss_bce + lambda_aux * loss_ce

                p_primary = torch.sigmoid(prim_logits).squeeze(-1)
                p_fake = model.aux_head.compute_fake_probability(aux_logits).squeeze(-1)

                val_loss += loss.item() * len(targets)
                val_samples += len(targets)

                val_targets_list.extend(targets.squeeze(-1).cpu().numpy())
                val_p_primary_list.extend(p_primary.cpu().numpy())
                val_p_fake_list.extend(p_fake.cpu().numpy())

        avg_val_loss = val_loss / max(1, val_samples)
        # Compute composite validation AUC
        fused_probs = 0.5 * np.array(val_p_primary_list) + 0.5 * np.array(val_p_fake_list)
        val_auc = roc_auc_score(val_targets_list, fused_probs)
        val_pred = (fused_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_targets_list, val_pred)

        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC (Fused): {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f} | LR: {lr:.6f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_auc": best_val_auc,
                    "config": config,
                    "ablation": "m13_2a_joint",
                },
                best_ckpt_path,
            )
            logger.info(f"  -> Saved new best checkpoint (Val AUC: {best_val_auc:.4f}) to {best_ckpt_path}")

    logger.info(f"Ablation 2 training completed. Best Val AUC: {best_val_auc:.4f}")
    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="M13.2a Auxiliary Manipulation Head Training")
    parser.add_argument("--ablation", choices=["frozen", "joint", "both"], default="both",
                        help="Which ablation to run: frozen (trunk frozen), joint (fine-tuned), or both")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    args = parser.parse_args()

    # Pre-execution integrity check
    verify_m12_integrity()

    # Load config and setup device
    config = load_config(str(CONFIG_PATH))
    device = torch.device(config.get_effective_device())
    logger.info(f"Using compute device: {device}")

    # Load datasets
    manifest_df = pd.read_csv(MANIFEST_PATH)
    filtered_df = filter_manifest_to_existing(manifest_df, PROCESSED_DIR)

    train_dataset = ProcessedVideoDataset(
        manifest=filtered_df,
        split="train",
        processed_dir=PROCESSED_DIR,
        expected_num_frames=config.data.sampling.num_frames,
        return_family_label=True,
    )
    val_dataset = ProcessedVideoDataset(
        manifest=filtered_df,
        split="val",
        processed_dir=PROCESSED_DIR,
        expected_num_frames=config.data.sampling.num_frames,
        return_family_label=True,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Balanced batch sampler
    train_labels = [s["label"] for s in train_dataset.samples]
    vid_to_dataset = dict(zip(filtered_df["video_id"].astype(str), filtered_df["dataset"]))
    train_datasets = [vid_to_dataset[s["video_id"]] for s in train_dataset.samples]

    train_batch_sampler = BalancedDatasetBatchSampler(
        labels=train_labels,
        dataset_ids=train_datasets,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    if args.ablation in ("frozen", "both"):
        train_ablation_frozen(train_loader, val_loader, config, device, epochs=args.epochs)
        verify_m12_integrity()

    if args.ablation in ("joint", "both"):
        train_ablation_joint(train_loader, val_loader, config, device, epochs=args.epochs)
        verify_m12_integrity()

    logger.info("All requested M13.2a training runs completed successfully.")


if __name__ == "__main__":
    main()
