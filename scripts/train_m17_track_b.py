"""Training Script for Milestone 17 Track B: Learned Temporal/Facial Dynamics.

Implements pre-registered M17 Track B training protocol:
1. Pinned Application Track (v5-family-invariant) verified and untouched.
2. Training split: FF++ C23 disjoint split (Deepfakes, FaceSwap, NeuralTextures, and clean reals).
   Zero Face2Face and zero Celeb-DF in training or validation splits.
3. 100% disjoint actor group_id partition between train and val.
4. Balanced sampling: exactly 50% Real and 50% Fake via BalancedBatchSampler.
5. Parameter freezing: Stages 0 to 4 frozen; Stage 5, RegionDynamicsBranch, gating, and heads trainable.
6. Loss: M17TrackBLoss (primary BCE + 0.20 dynamics BCE + 0.30 auxiliary CE + 0.05 gate L2).
7. RTX 3050 VRAM boundary: <= 5.2 GB peak VRAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProcessedVideoDataset
from src.data.samplers import BalancedBatchSampler
from src.models.detector import DeepfakeDetector
from src.models.m17_detector import (
    M17TrackBDetector,
    M17TrackBLoss,
    M17TrackBOutput,
)
from src.utils.config import _dict_to_config
from src.utils.logger import setup_logger

logger = setup_logger("m17_track_b_train")

PROCESSED_DIRS = [
    PROJECT_ROOT / "processed" / "m12_full",
    PROJECT_ROOT / "processed" / "m13_3_lofo",
]
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
V5_CHAMPION_PATH = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "lofo_f2f.csv"

EXPECTED_M12_HASH = "c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56"
EXPECTED_V5_HASH = "4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7"


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_baseline_and_champion_integrity() -> None:
    """Verify that M12 baseline and v5 champion checkpoints are 100% bitwise immutable."""
    assert M12_CHECKPOINT_PATH.exists(), f"M12 checkpoint missing: {M12_CHECKPOINT_PATH}"
    assert V5_CHAMPION_PATH.exists(), f"v5 champion missing: {V5_CHAMPION_PATH}"

    m12_hash = sha256_file(M12_CHECKPOINT_PATH)
    assert m12_hash == EXPECTED_M12_HASH, (
        f"CRITICAL: M12 baseline modified! {m12_hash} != {EXPECTED_M12_HASH}"
    )

    v5_hash = sha256_file(V5_CHAMPION_PATH)
    assert v5_hash == EXPECTED_V5_HASH, (
        f"CRITICAL: v5 champion modified! {v5_hash} != {EXPECTED_V5_HASH}"
    )

    logger.info(f"Verified immutable hashes. M12: {m12_hash[:12]}... | v5: {v5_hash[:12]}...")


def prepare_manifest(manifest_path: Path, processed_dirs: List[Path]) -> pd.DataFrame:
    """Filter manifest strictly to existing FF++ sequences (excluding Face2Face and Celeb-DF in train/val)."""
    df = pd.read_csv(manifest_path)
    ffpp_mask = df["dataset"] == "ffpp"
    df = df[ffpp_mask].copy().reset_index(drop=True)

    valid_rows = []
    skipped = 0
    for _, row in df.iterrows():
        split = str(row["split"])
        vid = str(row["video_id"])
        if split in ("train", "val"):
            found = any((pdir / split / "tensors" / f"{vid}.pt").exists() for pdir in processed_dirs)
            if found:
                valid_rows.append(row)
            else:
                skipped += 1
        else:
            valid_rows.append(row)

    if skipped > 0:
        logger.warning(f"Filtered out {skipped} rows without preprocessed tensors.")

    filtered_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    # Verify zero Face2Face in train/val
    train_f2f = filtered_df[
        (filtered_df["split"] == "train") & (filtered_df["manipulation_type"] == "Face2Face")
    ]
    val_f2f = filtered_df[
        (filtered_df["split"] == "val") & (filtered_df["manipulation_type"] == "Face2Face")
    ]
    assert len(train_f2f) == 0, "CRITICAL LEAKAGE: Face2Face present in training set!"
    assert len(val_f2f) == 0, "CRITICAL LEAKAGE: Face2Face present in validation set!"

    return filtered_df


def evaluate_val(
    model: M17TrackBDetector,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation loader."""
    model.eval()
    all_targets: List[float] = []
    all_p_fused: List[float] = []
    all_p_dyn: List[float] = []
    all_p_composite: List[float] = []
    all_gates: List[float] = []

    with torch.no_grad():
        for batch in loader:
            tensors, targets, vids, _ = batch
            tensors = tensors.to(device, dtype=torch.float32)

            with torch.amp.autocast("cuda"):
                out: M17TrackBOutput = model(tensors)

            all_targets.extend(targets.cpu().numpy().flatten().tolist())
            all_p_fused.extend(out.p_fused.cpu().numpy().flatten().tolist())
            all_p_dyn.extend(out.p_dyn.cpu().numpy().flatten().tolist())
            all_p_composite.extend(out.p_composite.cpu().numpy().flatten().tolist())
            all_gates.extend(out.gate_values.mean(dim=-1).cpu().numpy().flatten().tolist())

    y_true = np.array(all_targets)
    p_comp = np.nan_to_num(np.array(all_p_composite), nan=0.5)
    p_fused = np.nan_to_num(np.array(all_p_fused), nan=0.5)
    p_dyn = np.nan_to_num(np.array(all_p_dyn), nan=0.5)

    val_auc = float(roc_auc_score(y_true, p_comp)) if len(np.unique(y_true)) > 1 else 0.5
    fused_auc = float(roc_auc_score(y_true, p_fused)) if len(np.unique(y_true)) > 1 else 0.5
    dyn_auc = float(roc_auc_score(y_true, p_dyn)) if len(np.unique(y_true)) > 1 else 0.5
    y_pred = (p_comp >= 0.5).astype(int)
    acc = float(np.mean(y_true == y_pred))
    mean_gate = float(np.mean(all_gates))

    return {
        "val_auc": val_auc,
        "fused_auc": fused_auc,
        "dyn_auc": dyn_auc,
        "val_acc": acc,
        "mean_gate": mean_gate,
    }


def main():
    parser = argparse.ArgumentParser(description="M17 Track B (Facial Dynamics) Training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=8, help="Max training epochs (default: 8)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per forward pass (default: 8)")
    parser.add_argument("--lr-heads", type=float, default=1e-4, help="Learning rate for heads & dynamics stream")
    parser.add_argument("--lr-stage5", type=float, default=5e-5, help="Learning rate for Stage-5 backbone")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (default: 4)")
    args = parser.parse_args()

    # Pre-flight integrity check
    verify_baseline_and_champion_integrity()

    # Reproducibility seed setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Seed: {args.seed} | Track: B (Facial Dynamics)")

    # Prepare datasets
    manifest_df = prepare_manifest(MANIFEST_PATH, PROCESSED_DIRS)

    train_dataset = ProcessedVideoDataset(
        manifest=manifest_df,
        split="train",
        processed_dir=PROCESSED_DIRS,
        return_family_label=True,
    )
    val_dataset = ProcessedVideoDataset(
        manifest=manifest_df,
        split="val",
        processed_dir=PROCESSED_DIRS,
        return_family_label=True,
    )

    logger.info(f"Dataset indexed: Train={len(train_dataset)} | Val={len(val_dataset)}")

    # Balanced batch sampling (50% Real / 50% Fake)
    train_labels = [train_dataset.samples[i]["label"] for i in range(len(train_dataset))]
    train_batch_sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Initialize Base Detector from frozen M12 baseline
    ckpt = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    raw_cfg = ckpt["config"]
    m12_config = _dict_to_config(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    base_detector = DeepfakeDetector(config=m12_config.model).to(device)
    base_detector.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded base detector weights from frozen M12 baseline.")

    # Build M17 Track B Detector
    model = M17TrackBDetector(
        base_detector=base_detector,
        num_families=5,
        dyn_dim=128,
        mid_in_channels=112,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        gate_bottleneck=128,
    ).to(device)

    # Freeze stages 0-4
    model.freeze_stages_0_to_4()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Parameter groups
    stage5_params = [p for p in model.base_detector.backbone.features[5].parameters() if p.requires_grad] if hasattr(model.base_detector.backbone, "features") else []
    branch_params = [
        p
        for m in [model.dynamics_branch, model.dyn_head, model.gate_mlp, model.dyn_proj, model.aux_head, model.fused_head, model.fused_norm]
        for p in m.parameters()
        if p.requires_grad
    ]

    optimizer_groups = [{"params": branch_params, "lr": args.lr_heads, "weight_decay": 1e-4}]
    if stage5_params:
        optimizer_groups.append({"params": stage5_params, "lr": args.lr_stage5, "weight_decay": 1e-4})

    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = M17TrackBLoss(lambda_dyn=0.20, lambda_aux=0.30, lambda_gate=0.05).to(device)
    scaler = torch.amp.GradScaler("cuda", init_scale=8192.0)

    trainable_param_list = [p for group in optimizer.param_groups for p in group["params"]]

    ckpt_dir = PROJECT_ROOT / "checkpoints" / f"m17_track_b_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = -1.0
    best_epoch = 0
    patience_counter = 0

    logger.info(f"Starting M17 Track B training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in train_loader:
            tensors, targets, vids, families = batch
            tensors = tensors.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            families = families.to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                out: M17TrackBOutput = model(tensors)
                loss_dict = loss_fn(out, targets, families)
                loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                logger.warning(f"Epoch {epoch} non-finite loss detected ({loss.item()}), skipping batch.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_param_list, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_time = time.time() - t0
        avg_loss = epoch_loss / max(1, num_batches)

        # Validation
        val_metrics = evaluate_val(model, val_loader, device)
        val_auc = val_metrics["val_auc"]

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{train_time:.1f}s] - "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val AUC (Comp): {val_auc:.4f} | "
            f"Val AUC (Fused): {val_metrics['fused_auc']:.4f} | "
            f"Val AUC (Dyn): {val_metrics['dyn_auc']:.4f} | "
            f"Mean Gate: {val_metrics['mean_gate']:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_path = ckpt_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "val_metrics": val_metrics,
                    "config": ckpt["config"],
                },
                best_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {best_path} (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={args.patience})")
                break

    logger.info(f"Training completed. Best Epoch: {best_epoch} | Best Val AUC: {best_val_auc:.4f}")
    return best_val_auc


if __name__ == "__main__":
    main()
