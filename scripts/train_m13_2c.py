"""Training script for Milestone 13.2c: Identity/Manipulation Disentanglement.

Supports three controlled configurations:
1. true_group: Full disentanglement with genuine group_id identity contrastive loss.
2. shuffled_group: Negative control with randomly permuted group assignments.
3. orth_only: Subspace orthogonality constraint without contrastive loss.
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
from src.models.disentangled_detector import DisentangledDeepfakeDetector
from src.models.heads.disentangled import SubspaceOrthogonalityLoss, SupervisedContrastiveLoss
from src.utils.config import AppConfig, _dict_to_config, load_config
from src.utils.logger import setup_logger

logger = setup_logger("m13_2c_train")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
PROCESSED_DIR = PROJECT_ROOT / "processed" / "m12_full"
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
FREEZE_MANIFEST_PATH = PROJECT_ROOT / "reports" / "m12_baseline_freeze.json"
M13_2B_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m13_2b_stage5" / "best.pth"


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
    """Filter manifest for train and val to only include rows with preprocessed tensors."""
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
    parser = argparse.ArgumentParser(description="M13.2c Disentanglement Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="true_group",
        choices=["true_group", "shuffled_group", "orth_only", "proj_only", "contrast_only"],
        help="Disentanglement training mode",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_aux", type=float, default=0.5, help="Auxiliary family CE loss weight")
    parser.add_argument("--lambda_mid", type=float, default=0.3, help="Stage-5 mid-level loss weight")
    parser.add_argument("--lambda_orth", type=float, default=0.2, help="Subspace orthogonality loss weight")
    parser.add_argument("--lambda_con_id", type=float, default=0.1, help="Identity contrastive loss weight")
    parser.add_argument("--lambda_con_manip", type=float, default=0.1, help="Manipulation contrastive loss weight")
    args = parser.parse_args()

    # Pre-execution integrity check
    verify_m12_integrity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} | Mode: {args.mode}")

    # Set mode-specific hyperparameters and paths
    if args.mode == "proj_only":
        lambda_orth = 0.0
        lambda_con_id = 0.0
        lambda_con_manip = 0.0
        output_dir = PROJECT_ROOT / "checkpoints" / "m13_2c_proj_only"
    elif args.mode == "orth_only":
        lambda_orth = args.lambda_orth
        lambda_con_id = 0.0
        lambda_con_manip = 0.0
        output_dir = PROJECT_ROOT / "checkpoints" / "m13_2c_orth_only"
    elif args.mode == "contrast_only":
        lambda_orth = 0.0
        lambda_con_id = args.lambda_con_id
        lambda_con_manip = args.lambda_con_manip
        output_dir = PROJECT_ROOT / "checkpoints" / "m13_2c_contrast_only"
    elif args.mode == "shuffled_group":
        lambda_orth = args.lambda_orth
        lambda_con_id = args.lambda_con_id
        lambda_con_manip = args.lambda_con_manip
        output_dir = PROJECT_ROOT / "checkpoints" / "m13_2c_shuffled"
    else:  # true_group
        lambda_orth = args.lambda_orth
        lambda_con_id = args.lambda_con_id
        lambda_con_manip = args.lambda_con_manip
        output_dir = PROJECT_ROOT / "checkpoints" / "m13_2c_true_group"

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best.pth"

    # Load datasets
    manifest_df = pd.read_csv(MANIFEST_PATH)
    filtered_df = filter_manifest_to_existing(manifest_df, PROCESSED_DIR)

    train_dataset = ProcessedVideoDataset(
        manifest=filtered_df,
        split="train",
        processed_dir=PROCESSED_DIR,
        expected_num_frames=16,
        return_family_label=True,
        return_group_id=True,
    )
    val_dataset = ProcessedVideoDataset(
        manifest=filtered_df,
        split="val",
        processed_dir=PROCESSED_DIR,
        expected_num_frames=16,
        return_family_label=True,
        return_group_id=True,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Balanced batch sampler
    train_labels = [s["label"] for s in train_dataset.samples]
    vid_to_dataset = dict(zip(filtered_df["video_id"].astype(str), filtered_df["dataset"]))
    train_datasets = [vid_to_dataset[s["video_id"]] for s in train_dataset.samples]

    train_batch_sampler = BalancedDatasetBatchSampler(
        labels=train_labels,
        dataset_ids=train_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Load base detector weights from M12 baseline
    m12_ckpt = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    raw_cfg = m12_ckpt["config"]
    m12_config = _dict_to_config(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    base_detector = DeepfakeDetector(config=m12_config.model).to(device)
    base_detector.load_state_dict(m12_ckpt["model_state_dict"])

    # Build Disentangled model with Stage-5 Spatial Head
    model = DisentangledDeepfakeDetector(
        base_detector=base_detector,
        num_families=5,
        subspace_dim=128,
        projection_hidden_dim=512,
        projection_dropout=0.2,
        enable_mid_head=True,
        mid_in_channels=112,
        mid_hidden_dim=128,
    ).to(device)

    # Initialize Stage-5 head and temporal weights from M13.2b if available
    if M13_2B_CHECKPOINT_PATH.exists():
        logger.info(f"Initializing Stage-5 head and temporal weights from M13.2b ({M13_2B_CHECKPOINT_PATH})...")
        b_ckpt = torch.load(M13_2B_CHECKPOINT_PATH, map_location=device, weights_only=False)
        b_state = b_ckpt["model_state_dict"]
        filtered_state = {k: v for k, v in b_state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        logger.info(f"Successfully loaded {len(filtered_state)} parameter tensors from M13.2b into Disentangled model.")

    model.unfreeze_temporal_and_heads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_orth = SubspaceOrthogonalityLoss(cov_weight=1.0)
    criterion_con = SupervisedContrastiveLoss(temperature=0.1)

    lambda_aux = args.lambda_aux
    lambda_mid = args.lambda_mid

    best_val_auc = -1.0
    epochs = args.epochs

    logger.info("=" * 70)
    logger.info(f"Starting M13.2c Training [{args.mode.upper()}]")
    logger.info(f"Loss weights: aux={lambda_aux}, mid={lambda_mid}, orth={lambda_orth}, con_id={lambda_con_id}, con_manip={lambda_con_manip}")
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        rng_shuffle = np.random.default_rng(42 + epoch)

        for batch in train_loader:
            tensors, targets, vids, family_targets, group_ids = batch
            tensors = tensors.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            family_targets = family_targets.to(device)

            # Map group_id strings to integer labels in batch
            if args.mode == "shuffled_group":
                # Negative control: randomly permute group labels
                shuffled_groups = rng_shuffle.permutation(list(group_ids))
                unique_groups, group_indices_arr = np.unique(shuffled_groups, return_inverse=True)
            else:
                unique_groups, group_indices_arr = np.unique(list(group_ids), return_inverse=True)

            group_indices = torch.tensor(group_indices_arr, dtype=torch.long, device=device)

            out = model(tensors, return_dict=True)

            loss_prim = criterion_bce(out.primary_logits, targets)
            loss_aux = criterion_ce(out.aux_family_logits, family_targets)
            loss_mid = criterion_bce(out.mid_logits, targets) if out.mid_logits is not None else torch.tensor(0.0)
            loss_orth = criterion_orth(out.z_id, out.z_manip)

            loss_id_con = criterion_con(out.z_id, group_indices) if lambda_con_id > 0 else torch.tensor(0.0)
            loss_manip_con = criterion_con(out.z_manip, family_targets) if lambda_con_manip > 0 else torch.tensor(0.0)

            loss = (
                loss_prim
                + lambda_aux * loss_aux
                + lambda_mid * loss_mid
                + lambda_orth * loss_orth
                + lambda_con_id * loss_id_con
                + lambda_con_manip * loss_manip_con
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

        avg_train_loss = total_loss / max(1, total_samples)

        # Validation loop
        model.eval()
        val_targets_list = []
        val_p_fused_list = []
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                tensors, targets, vids, family_targets, group_ids = batch
                tensors = tensors.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                family_targets = family_targets.to(device)

                out = model(tensors, return_dict=True)

                loss_prim = criterion_bce(out.primary_logits, targets)
                loss_aux = criterion_ce(out.aux_family_logits, family_targets)
                loss_mid = criterion_bce(out.mid_logits, targets) if out.mid_logits is not None else torch.tensor(0.0)
                loss_orth = criterion_orth(out.z_id, out.z_manip)

                loss = loss_prim + lambda_aux * loss_aux + lambda_mid * loss_mid + lambda_orth * loss_orth

                val_loss += loss.item() * len(targets)
                val_samples += len(targets)

                val_targets_list.extend(targets.squeeze(-1).cpu().numpy())
                val_p_fused_list.extend(out.p_fused.squeeze(-1).cpu().numpy())

        avg_val_loss = val_loss / max(1, val_samples)
        val_auc = roc_auc_score(val_targets_list, val_p_fused_list)
        val_pred = (np.array(val_p_fused_list) >= 0.5).astype(int)
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
                    "mode": args.mode,
                    "config": {
                        "lambda_aux": lambda_aux,
                        "lambda_mid": lambda_mid,
                        "lambda_orth": lambda_orth,
                        "lambda_con_id": lambda_con_id,
                        "lambda_con_manip": lambda_con_manip,
                    },
                },
                best_ckpt_path,
            )
            logger.info(f"  -> Saved new best checkpoint (Val AUC: {best_val_auc:.4f}) to {best_ckpt_path}")

    logger.info(f"M13.2c [{args.mode}] training complete. Best Val AUC: {best_val_auc:.4f}")

    # Post-training integrity check
    verify_m12_integrity()
    logger.info("M12 integrity verified post-training.")


if __name__ == "__main__":
    main()
