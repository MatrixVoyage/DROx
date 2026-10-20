"""Training Script for Milestone 13.3: Family-Invariant Artifact Representation.

Supports:
1. Training across any fold of the 4-Fold LOFO Matrix (lofo_f2f, lofo_nt, lofo_fs, lofo_df).
2. Multi-seed stability runs (e.g. seed=42, 1337, 2026).
3. Joint training of Multi-Domain Invariant Fusion, Local Frequency Branch, Local Texture ROI Branch,
   Stage-5 Spatial Head, and Multi-Family Auxiliary Head.
4. Cryptographic pre- and post-flight verification of frozen M12 baseline integrity.

Usage:
    .venv\\Scripts\\python.exe scripts/train_m13_3_lofo.py --fold f2f --seed 42 --epochs 10
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
from src.data.samplers import BalancedDatasetBatchSampler
from src.models.detector import DeepfakeDetector
from src.models.family_invariant_detector import (
    FamilyInvariantDeepfakeDetector,
    FamilyInvariantOutput,
)
from src.utils.config import AppConfig, _dict_to_config, load_config
from src.utils.logger import setup_logger

logger = setup_logger("m13_3_train")

PROCESSED_DIR = PROJECT_ROOT / "processed" / "m12_full"
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
FREEZE_MANIFEST_PATH = PROJECT_ROOT / "reports" / "m12_baseline_freeze.json"

FAMILY_LABEL_MAP = {
    "original": 0,
    "pristine": 0,
    "Deepfakes": 1,
    "deepfake": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
}


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
    assert current_hash == expected_hash, (
        f"CRITICAL: M12 checkpoint modified! {current_hash} != {expected_hash}"
    )
    logger.info(f"M12 integrity verified. Hash: {current_hash[:16]}...")
    return current_hash


def filter_manifest_to_existing(
    manifest_df: pd.DataFrame,
    processed_dirs: List[Path],
) -> pd.DataFrame:
    """Filter manifest for train and val to only include rows with preprocessed tensors."""
    valid_rows = []
    skipped_count = 0
    for _, row in manifest_df.iterrows():
        split = str(row["split"])
        vid = str(row["video_id"])
        if split in ("train", "val"):
            found = False
            for pdir in processed_dirs:
                tensor_path = pdir / split / "tensors" / f"{vid}.pt"
                meta_path = pdir / split / "metadata" / f"{vid}.json"
                if tensor_path.exists() and meta_path.exists():
                    found = True
                    break
            if found:
                valid_rows.append(row)
            else:
                skipped_count += 1
        else:
            valid_rows.append(row)

    if skipped_count > 0:
        logger.warning(f"Filtered out {skipped_count} manifest rows whose tensors were missing.")
    return pd.DataFrame(valid_rows)


# Using standard ProcessedVideoDataset with return_family_label=True


def train_one_epoch(
    model: FamilyInvariantDeepfakeDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 2,
    lambda_aux: float = 0.5,
    lambda_mid: float = 0.3,
    lambda_freq: float = 0.2,
    lambda_texture: float = 0.2,
) -> Dict[str, float]:
    """Train for one epoch with multi-domain loss."""
    model.train()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_prim = 0.0
    total_aux = 0.0
    total_freq = 0.0
    total_tex = 0.0
    total_samples = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        tensors, targets, vids, family_targets = batch
        tensors = tensors.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        family_targets = family_targets.to(device)

        out: FamilyInvariantOutput = model(tensors, return_dict=True)

        l_prim = criterion_bce(out.primary_logits, targets)
        l_aux = criterion_ce(out.aux_family_logits, family_targets)
        l_freq = criterion_bce(out.freq_logits, targets)
        l_tex = criterion_bce(out.texture_logits, targets)

        loss = l_prim + lambda_aux * l_aux + lambda_freq * l_freq + lambda_texture * l_tex

        if out.mid_logits is not None:
            l_mid = criterion_bce(out.mid_logits, targets)
            loss = loss + lambda_mid * l_mid

        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        b_size = tensors.size(0)
        total_loss += loss.item() * grad_accum_steps * b_size
        total_prim += l_prim.item() * b_size
        total_aux += l_aux.item() * b_size
        total_freq += l_freq.item() * b_size
        total_tex += l_tex.item() * b_size
        total_samples += b_size

    return {
        "loss": total_loss / max(1, total_samples),
        "loss_primary": total_prim / max(1, total_samples),
        "loss_aux": total_aux / max(1, total_samples),
        "loss_freq": total_freq / max(1, total_samples),
        "loss_texture": total_tex / max(1, total_samples),
    }


def evaluate(
    model: FamilyInvariantDeepfakeDetector,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation loader."""
    model.eval()
    all_targets = []
    all_p_fused = []
    all_p_composite = []
    all_p_freq = []
    all_p_texture = []

    with torch.no_grad():
        for batch in loader:
            tensors, targets, vids, _ = batch
            tensors = tensors.to(device, dtype=torch.float32)
            out: FamilyInvariantOutput = model(tensors, return_dict=True)

            all_targets.extend(targets.cpu().numpy().flatten().tolist())
            all_p_fused.extend(out.p_fused.cpu().numpy().flatten().tolist())
            all_p_composite.extend(out.p_composite.cpu().numpy().flatten().tolist())
            all_p_freq.extend(out.p_freq.cpu().numpy().flatten().tolist())
            all_p_texture.extend(out.p_texture.cpu().numpy().flatten().tolist())

    y_true = np.array(all_targets)
    p_comp = np.array(all_p_composite)
    p_fused = np.array(all_p_fused)

    val_auc = float(roc_auc_score(y_true, p_comp)) if len(np.unique(y_true)) > 1 else 0.5
    fused_auc = float(roc_auc_score(y_true, p_fused)) if len(np.unique(y_true)) > 1 else 0.5

    y_pred = (p_comp >= 0.5).astype(int)
    acc = float(np.mean(y_true == y_pred))

    return {
        "val_auc": val_auc,
        "fused_auc": fused_auc,
        "val_acc": acc,
    }


def main():
    parser = argparse.ArgumentParser(description="M13.3 Family-Invariant Detector Training")
    parser.add_argument(
        "--fold",
        type=str,
        default="f2f",
        choices=["f2f", "nt", "fs", "df"],
        help="LOFO fold to train on (default: f2f)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per forward step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Pre-execution integrity check
    verify_m12_integrity()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} | Fold: {args.fold.upper()} | Seed: {args.seed}")

    manifest_path = PROJECT_ROOT / "manifests" / f"lofo_{args.fold}.csv"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}. Run build_lofo_manifests.py first."

    manifest_df = pd.read_csv(manifest_path)
    processed_dirs = [PROCESSED_DIR, PROJECT_ROOT / "processed" / "m13_3_lofo"]
    filtered_df = filter_manifest_to_existing(manifest_df, processed_dirs)

    train_dataset = ProcessedVideoDataset(
        manifest=filtered_df,
        split="train",
        processed_dir=processed_dirs,
        return_family_label=True,
    )
    val_dataset = ProcessedVideoDataset(
        manifest=filtered_df,
        split="val",
        processed_dir=processed_dirs,
        return_family_label=True,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Balanced batch sampler
    train_labels = [s["label"] for s in train_dataset.samples]
    vid_to_dataset = dict(zip(filtered_df["video_id"].astype(str), filtered_df["dataset"]))
    train_datasets = [vid_to_dataset.get(s["video_id"], "ffpp") for s in train_dataset.samples]

    train_batch_sampler = BalancedDatasetBatchSampler(
        labels=train_labels,
        dataset_ids=train_datasets,
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

    # Load frozen M12 base detector
    ckpt = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    raw_cfg = ckpt["config"]
    m12_config = _dict_to_config(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    base_detector = DeepfakeDetector(config=m12_config.model).to(device)
    base_detector.load_state_dict(ckpt["model_state_dict"])

    # Instantiate Family-Invariant Detector
    model = FamilyInvariantDeepfakeDetector(
        base_detector=base_detector,
        num_families=5,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        freq_dim=64,
        texture_dim=64,
        fusion_dim=512,
        enable_mid_head=True,
        mid_in_channels=112,
        mid_hidden_dim=128,
    ).to(device)

    model.unfreeze_temporal_and_heads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters (Temporal + Heads + Frequency + Texture + Fusion): {num_trainable:,}")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / f"m13_3_{args.fold}_seed{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pth"

    best_val_auc = -1.0
    best_metrics = {}

    print("=" * 75)
    print(f"Starting M13.3 Training: Fold {args.fold.upper()} (Seed: {args.seed})")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=2,
            lambda_aux=0.5,
            lambda_mid=0.3,
            lambda_freq=0.2,
            lambda_texture=0.2,
        )
        val_res = evaluate(model=model, loader=val_loader, device=device)
        scheduler.step(val_res["val_auc"])

        epoch_time = time.time() - t0
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_res['loss']:.4f} (prim: {train_res['loss_primary']:.4f}, "
            f"freq: {train_res['loss_freq']:.4f}, tex: {train_res['loss_texture']:.4f}) | "
            f"Val AUC: {val_res['val_auc']:.4f} (fused: {val_res['fused_auc']:.4f}) | "
            f"Val Acc: {val_res['val_acc']:.4f}"
        )

        if val_res["val_auc"] > best_val_auc:
            best_val_auc = val_res["val_auc"]
            best_metrics = {**val_res, "epoch": epoch}
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_auc": best_val_auc,
                    "val_metrics": best_metrics,
                    "fold": args.fold,
                    "seed": args.seed,
                },
                best_ckpt_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val AUC: {best_val_auc:.4f})")

    # Post-execution integrity check
    verify_m12_integrity()

    logger.info("=" * 75)
    logger.info(f"M13.3 Training Complete: Fold {args.fold.upper()} | Best Val AUC: {best_val_auc:.4f}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
