"""Training script for Milestone 13.2b: Stage-5 Mid-Level Auxiliary Head.

Tests whether intermediate spatial features (Stage 5, C=80) provide complementary
information before temporal aggregation:
Stage 5 ──→ mid-level auxiliary head
     │
     ↓
Temporal Transformer ──→ primary & auxiliary heads
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
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProcessedVideoDataset
from src.data.samplers import BalancedDatasetBatchSampler
from src.models.detector import DeepfakeDetector
from src.models.multi_head_detector import MultiHeadDeepfakeDetector
from src.utils.config import AppConfig, _dict_to_config, load_config
from src.utils.logger import setup_logger

logger = setup_logger("m13_2b_train")

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "m13_2b_aux_stage5.yaml"
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


def main():
    parser = argparse.ArgumentParser(description="M13.2b Stage-5 Auxiliary Head Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
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

    # Load M12 base detector
    ckpt = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    raw_cfg = ckpt["config"]
    m12_config = _dict_to_config(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    base_detector = DeepfakeDetector(config=m12_config.model).to(device)
    base_detector.load_state_dict(ckpt["model_state_dict"])

    # Build MultiHead model with Stage-5 Head enabled
    model = MultiHeadDeepfakeDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=config.model.auxiliary.hidden_dim,
        aux_dropout=config.model.auxiliary.dropout,
        enable_mid_head=True,
        mid_in_channels=112,
        mid_hidden_dim=128,
    ).to(device)
    model.unfreeze_temporal_and_heads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters (Temporal + Heads + Stage-5 Head): {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.training.optimizer.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()
    lambda_aux = float(config.model.auxiliary.loss_weight)
    lambda_mid = 0.3

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pth"

    best_val_auc = -1.0
    epochs = args.epochs

    logger.info("=" * 70)
    logger.info("Starting M13.2b Training: Stage-5 Mid-Level Auxiliary Head")
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            tensors, targets, vids, family_targets = batch
            tensors = tensors.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            family_targets = family_targets.to(device)

            out = model(tensors, return_dict=True)

            loss_prim = criterion_bce(out.primary_logits, targets)
            loss_aux = criterion_ce(out.aux_family_logits, family_targets)
            loss_mid = criterion_bce(out.mid_logits, targets) if out.mid_logits is not None else torch.tensor(0.0)

            loss = loss_prim + lambda_aux * loss_aux + lambda_mid * loss_mid

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
        val_p_composite_list = []
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                tensors, targets, vids, family_targets = batch
                tensors = tensors.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                family_targets = family_targets.to(device)

                out = model(tensors, return_dict=True)
                loss_prim = criterion_bce(out.primary_logits, targets)
                loss_aux = criterion_ce(out.aux_family_logits, family_targets)
                loss_mid = criterion_bce(out.mid_logits, targets) if out.mid_logits is not None else torch.tensor(0.0)

                loss = loss_prim + lambda_aux * loss_aux + lambda_mid * loss_mid

                p_prim = torch.sigmoid(out.primary_logits).squeeze(-1)
                p_aux = out.p_aux_fake.squeeze(-1)
                p_mid = torch.sigmoid(out.mid_logits).squeeze(-1) if out.mid_logits is not None else p_prim

                # Tri-head composite probability
                p_composite = 0.4 * p_prim + 0.4 * p_aux + 0.2 * p_mid

                val_loss += loss.item() * len(targets)
                val_samples += len(targets)

                val_targets_list.extend(targets.squeeze(-1).cpu().numpy())
                val_p_composite_list.extend(p_composite.cpu().numpy())

        avg_val_loss = val_loss / max(1, val_samples)
        val_auc = roc_auc_score(val_targets_list, val_p_composite_list)
        val_pred = (np.array(val_p_composite_list) >= 0.5).astype(int)
        val_f1 = f1_score(val_targets_list, val_pred)

        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC (Composite): {val_auc:.4f} | "
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
                    "ablation": "m13_2b_stage5",
                },
                best_ckpt_path,
            )
            logger.info(f"  -> Saved new best checkpoint (Val AUC: {best_val_auc:.4f}) to {best_ckpt_path}")

    logger.info(f"M13.2b training completed. Best Val AUC: {best_val_auc:.4f}")

    # Post-training integrity check
    verify_m12_integrity()
    logger.info("M13.2b execution finished successfully.")


if __name__ == "__main__":
    main()
