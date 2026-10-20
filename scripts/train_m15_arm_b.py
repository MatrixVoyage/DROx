"""Training Script for Milestone 15 Arm B: Temporal-Spatial Cross-Attention Detector (M15-TS-CrossAttn).

Preregistered Protocol Reference: context/m15_preregistration.md
Governing Standard: Rule 2 (Source-of-Truth), Rule 6 & 10 (Research Integrity), Rule 8 (RTX 3050 6GB VRAM), Rule 9 (Dataset Discipline)

Key Operational Constraints:
1. Pinned Application Track (v5-family-invariant) remains untouched and bitwise verified.
2. Training split: FF++ C23 only (Deepfakes, FaceSwap, NeuralTextures, and clean reals).
   Zero Face2Face and zero Celeb-DF in training or validation.
3. 100% disjoint group_id partition between train and val (train_groups ∩ val_groups = ∅).
4. Balanced sampling: exactly 50% Real and 50% Fake via BalancedDatasetBatchSampler.
5. Parameter budgets: Total <= 31.0M, Trainable <= 6.5M.
6. Loss formulation: L_total = L_BCE + 0.30 * L_CE(aux) + 0.05 * L_sparsity(A_cross).
7. Differential learning rates: 5e-5 (Stage 5 backbone), 1e-4 (cross-attention fusion and heads).

Usage:
    .venv\\Scripts\\python.exe scripts/train_m15_arm_b.py --seed 42 --epochs 10 --batch-size 8
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
from src.models.m15_cross_attention_detector import (
    M15CrossAttentionDetector,
    M15DetectorOutput,
    M15Loss,
)
from src.utils.config import _dict_to_config
from src.utils.logger import setup_logger

logger = setup_logger("m15_arm_b_train")

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

    logger.info(f"Immutable checkpoint hashes verified. M12: {m12_hash[:12]}... | v5: {v5_hash[:12]}...")


def prepare_m15_manifest(manifest_path: Path, processed_dirs: List[Path]) -> pd.DataFrame:
    """Filter manifest strictly to existing FF++ sequences (excluding Face2Face and Celeb-DF)."""
    df = pd.read_csv(manifest_path)

    # Strict isolation: FF++ only in train and val
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
    train_f2f = filtered_df[(filtered_df["split"] == "train") & (filtered_df.get("manipulation_type", "") == "Face2Face")]
    val_f2f = filtered_df[(filtered_df["split"] == "val") & (filtered_df.get("manipulation_type", "") == "Face2Face")]
    assert len(train_f2f) == 0, "DATA LEAKAGE: Face2Face detected in M15 training split!"
    assert len(val_f2f) == 0, "DATA LEAKAGE: Face2Face detected in M15 validation split!"

    # Verify disjoint group_ids
    train_groups = set(filtered_df[filtered_df["split"] == "train"]["group_id"])
    val_groups = set(filtered_df[filtered_df["split"] == "val"]["group_id"])
    assert len(train_groups.intersection(val_groups)) == 0, (
        "DATA LEAKAGE: Group ID overlap between M15 train and val splits!"
    )

    logger.info(
        f"M15 Manifest ready: Train={len(filtered_df[filtered_df['split']=='train'])} | "
        f"Val={len(filtered_df[filtered_df['split']=='val'])} | "
        f"Train groups={len(train_groups)}, Val groups={len(val_groups)} (Disjoint: 100%)"
    )
    return filtered_df


def train_one_epoch(
    model: M15CrossAttentionDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: M15Loss,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    grad_accum_steps: int = 2,
) -> Dict[str, float]:
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_aux = 0.0
    total_sparsity = 0.0
    total_samples = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        tensors, targets, vids, family_targets = batch
        tensors = tensors.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        family_targets = family_targets.to(device, dtype=torch.long)
        b_size = tensors.size(0)

        with torch.amp.autocast("cuda"):
            out: M15DetectorOutput = model(tensors, return_dict=True)
            losses = loss_fn(out, targets, family_targets)
            loss = losses["loss_total"] / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += losses["loss_total"].item() * b_size
        total_bce += losses["loss_bce"].item() * b_size
        total_aux += losses["loss_aux"].item() * b_size
        total_sparsity += losses["loss_sparsity"].item() * b_size
        total_samples += b_size

    return {
        "loss": total_loss / max(1, total_samples),
        "loss_bce": total_bce / max(1, total_samples),
        "loss_aux": total_aux / max(1, total_samples),
        "loss_sparsity": total_sparsity / max(1, total_samples),
    }


def evaluate(
    model: M15CrossAttentionDetector,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation loader."""
    model.eval()
    all_targets: List[float] = []
    all_p_fused: List[float] = []
    all_p_composite: List[float] = []
    all_gates: List[float] = []

    with torch.no_grad():
        for batch in loader:
            tensors, targets, vids, _ = batch
            tensors = tensors.to(device, dtype=torch.float32)

            with torch.amp.autocast("cuda"):
                out: M15DetectorOutput = model(tensors, return_dict=True)

            all_targets.extend(targets.cpu().numpy().flatten().tolist())
            all_p_fused.extend(out.p_fused.cpu().numpy().flatten().tolist())
            all_p_composite.extend(out.p_composite.cpu().numpy().flatten().tolist())
            all_gates.extend(out.gate_values.mean(dim=-1).cpu().numpy().flatten().tolist())

    y_true = np.array(all_targets)
    p_comp = np.array(all_p_composite)
    p_fused = np.array(all_p_fused)

    val_auc = float(roc_auc_score(y_true, p_comp)) if len(np.unique(y_true)) > 1 else 0.5
    fused_auc = float(roc_auc_score(y_true, p_fused)) if len(np.unique(y_true)) > 1 else 0.5
    y_pred = (p_comp >= 0.5).astype(int)
    acc = float(np.mean(y_true == y_pred))
    mean_gate = float(np.mean(all_gates))

    return {
        "val_auc": val_auc,
        "fused_auc": fused_auc,
        "val_acc": acc,
        "mean_gate": mean_gate,
    }


def main():
    parser = argparse.ArgumentParser(description="M15 Arm B (M15-TS-CrossAttn) Training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per forward pass (default: 8)")
    parser.add_argument("--lr-heads", type=float, default=1e-4, help="Learning rate for cross-attention & heads")
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
    logger.info(f"Using device: {device} | Seed: {args.seed} | Arm: B (M15-TS-CrossAttn)")

    # Prepare datasets
    manifest_df = prepare_m15_manifest(MANIFEST_PATH, PROCESSED_DIRS)

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
    train_labels = [s["label"] for s in train_dataset.samples]

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

    # Build M15 Cross-Attention Detector
    model = M15CrossAttentionDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        d_attn=256,
        nhead=4,
        gate_bottleneck=128,
        mid_in_channels=112,
        enable_local_refinement=False,
    ).to(device)

    # Apply preregistered parameter freezing strategy
    model.freeze_stages_0_to_4()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params:,} (Budget: <= 31,000,000)")
    logger.info(f"Trainable Parameters: {trainable_params:,} (Budget: <= 6,500,000)")

    # Differential learning rate parameter groups
    stage5_params = [p for p in model.base_detector.backbone.features[5].parameters() if p.requires_grad]
    head_params = [
        p
        for m in [model.cross_attn_fusion, model.aux_head, model.primary_head]
        for p in m.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": stage5_params, "lr": args.lr_stage5, "weight_decay": 1e-4},
            {"params": head_params, "lr": args.lr_heads, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    loss_fn = M15Loss(lambda_aux=0.30, lambda_gate=0.05).to(device)
    scaler = torch.amp.GradScaler("cuda")

    # Destination directory
    ckpt_dir = PROJECT_ROOT / "checkpoints" / f"m15_arm_b_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pth"
    history_path = ckpt_dir / "training_history.json"

    logger.info(f"Output directory: {ckpt_dir}")
    logger.info("Beginning M15 Arm B training loop...")

    best_val_auc = 0.0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_batch_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            grad_accum_steps=2,
        )

        val_metrics = evaluate(model=model, loader=val_loader, device=device)
        scheduler.step()

        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_metrics['loss']:.4f} (BCE: {train_metrics['loss_bce']:.4f}, Aux: {train_metrics['loss_aux']:.4f}, Sparsity: {train_metrics['loss_sparsity']:.4f}) | "
            f"Val AUC: {val_metrics['val_auc']:.4f} | Fused AUC: {val_metrics['fused_auc']:.4f} | "
            f"Val Acc: {val_metrics['val_acc']*100:.2f}% | Mean Gate: {val_metrics['mean_gate']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "elapsed_seconds": round(elapsed, 2),
            **train_metrics,
            **val_metrics,
            "lr_stage5": optimizer.param_groups[0]["lr"],
            "lr_heads": optimizer.param_groups[1]["lr"],
        }
        history.append(epoch_record)

        # Early stopping and checkpointing based on Validation AUC
        if val_metrics["val_auc"] > best_val_auc:
            best_val_auc = val_metrics["val_auc"]
            patience_counter = 0

            checkpoint_payload = {
                "epoch": epoch,
                "arm": "Arm_B_TS_CrossAttn",
                "seed": args.seed,
                "val_auc": best_val_auc,
                "model_state_dict": model.state_dict(),
                "config": m12_config,
                "architecture": {
                    "type": "M15CrossAttentionDetector",
                    "num_families": 5,
                    "d_attn": 256,
                    "nhead": 4,
                    "gate_bottleneck": 128,
                    "mid_in_channels": 112,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                },
                "preregistration": "context/m15_preregistration.md",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            torch.save(checkpoint_payload, best_ckpt_path)
            logger.info(f" Saved new best checkpoint -> {best_ckpt_path} (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_hash = sha256_file(best_ckpt_path)
    logger.info(f"Training completed successfully. Best Val AUC: {best_val_auc:.4f}")
    logger.info(f"Best checkpoint SHA-256: {best_hash}")


if __name__ == "__main__":
    main()
