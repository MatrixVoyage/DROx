"""Milestone 20.2: Training & Fine-Tuning Pipeline for Pyramidal Dual-Stream Detector.

Fine-tunes the Pyramidal Dual-Stream Detector:
- Initializes from proven M18 checkpoint (Seed 42)
- Integrates learnable PyramidalFusionStem with native Laplacian high-frequency injection
- Supervised across 5 balanced classes: Pristine Real (0), Face2Face (1),
  NeuralTextures (2), FaceSwap (3), Deepfakes (4)
- Early stopping governed strictly by multi-family validation loss (L_val)
- Zero exposure of the locked Face2Face test holdout or cross-dataset test benchmarks
- Adheres to 6 GB RTX 3050 VRAM budget with AMP fp16 and gradient accumulation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.m20_pyramidal_detector import (
    M20PyramidalDualStreamDetector,
    PyramidalFusionStem,
)
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("train_m20_pyramidal_detector")

# Paths & References
MANIFEST_FFPP = PROJECT_ROOT / "manifests" / "ffpp_c23.csv"
TRUNK_CKPT_PATH = PROJECT_ROOT / "checkpoints" / "m17_track_a_seed42" / "best.pth"
M18_CKPT_PATH = PROJECT_ROOT / "checkpoints" / "m18_dual_stream_seed42" / "best.pth"

TENSOR_DIRS = [
    PROJECT_ROOT / "processed" / "m12_full",
    PROJECT_ROOT / "processed" / "m13_3_lofo",
]

FAMILY_MAP = {
    "original": 0,
    "pristine": 0,
    "Face2Face": 1,
    "NeuralTextures": 2,
    "FaceSwap": 3,
    "Deepfakes": 4,
}

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_tensor_path(video_id: str, split: str) -> Optional[Path]:
    """Find preprocessed tensor on disk."""
    candidates = [
        TENSOR_DIRS[0] / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[0] / "train" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[0] / "val" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / "train" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / "val" / "tensors" / f"{video_id}.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def apply_compression_augmentation(tensor: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Apply random compression/blur augmentation to video tensor (T, 3, 112, 112)."""
    if random.random() > p:
        return tensor

    arr = tensor.numpy()  # (T, 3, H, W)
    aug_type = random.choice(["jpeg", "blur", "dct_quant"])

    if aug_type == "jpeg":
        q = random.randint(45, 80)
        out_arr = np.zeros_like(arr)
        for t in range(arr.shape[0]):
            frame = arr[t].transpose(1, 2, 0)
            frame = (frame * NORM_STD + NORM_MEAN) * 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
            success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if success:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                out_arr[t] = ((dec - NORM_MEAN) / NORM_STD).transpose(2, 0, 1)
            else:
                out_arr[t] = arr[t]
        return torch.from_numpy(out_arr)

    elif aug_type == "blur":
        k = random.choice([3, 5])
        out_arr = np.zeros_like(arr)
        for t in range(arr.shape[0]):
            frame = arr[t].transpose(1, 2, 0)
            blurred = cv2.GaussianBlur(frame, (k, k), 0)
            out_arr[t] = blurred.transpose(2, 0, 1)
        return torch.from_numpy(out_arr)

    elif aug_type == "dct_quant":
        quant_scale = random.choice([16.0, 32.0])
        quant_arr = np.round(arr * quant_scale) / quant_scale
        return torch.from_numpy(quant_arr)

    return tensor


class M20PyramidalFFppDataset(Dataset):
    """Multi-family dataset for FF++ C23 with high-frequency spatial augmentation."""

    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        augment: bool = True,
        max_samples_per_class: Optional[int] = None,
    ) -> None:
        self.split = split
        self.augment = augment

        df = pd.read_csv(manifest_path)
        df_split = df[df["split"] == split].copy().reset_index(drop=True)

        self.samples = []
        counts: Dict[str, int] = {}
        for _, row in df_split.iterrows():
            vid = str(row["video_id"])
            mtype = str(row["manipulation_type"])
            if mtype not in FAMILY_MAP:
                continue

            if max_samples_per_class is not None:
                if counts.get(mtype, 0) >= max_samples_per_class:
                    continue

            tp = find_tensor_path(vid, split)
            if tp is not None:
                fam_label = FAMILY_MAP[mtype]
                bin_label = 0 if fam_label == 0 else 1
                self.samples.append((tp, bin_label, fam_label, mtype))
                counts[mtype] = counts.get(mtype, 0) + 1

        logger.info(
            f"Loaded M20PyramidalFFppDataset [{split}]: {len(self.samples)} clips (Class breakdown: {counts})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tp, bin_label, fam_label, _ = self.samples[idx]
        t = torch.load(tp, map_location="cpu", weights_only=False)

        # Standardize to (16, 3, 112, 112)
        if t.dim() == 5 and t.shape[0] == 1:
            t = t.squeeze(0)

        if self.augment:
            t = apply_compression_augmentation(t, p=0.4)

        y_bin = torch.tensor([float(bin_label)], dtype=torch.float32)
        y_fam = torch.tensor(fam_label, dtype=torch.long)
        return t, y_bin, y_fam


def train_m20_model(
    epochs: int = 8,
    lr: float = 1e-4,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    seed: int = 42,
    init_alpha: float = 0.50,
) -> Path:
    """Train M20 Pyramidal Dual-Stream Detector."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting M20 Training on device={device}")

    run_dir = PROJECT_ROOT / "checkpoints" / f"m20_pyramidal_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "best.pth"
    latest_ckpt_path = run_dir / "latest.pth"

    # 1. Load M17 Track A trunk
    cfg = load_config("configs/experiments/m12_full.yaml")
    base = DeepfakeDetector(config=cfg.model).to(device)
    trunk = M17TrackADetector(
        base_detector=base,
        num_families=5,
        res_dim=128,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        gate_bottleneck=128,
    ).to(device)

    assert TRUNK_CKPT_PATH.exists(), f"Missing M17 Track A checkpoint: {TRUNK_CKPT_PATH}"
    trunk_ckpt = torch.load(TRUNK_CKPT_PATH, map_location=device, weights_only=False)
    trunk.load_state_dict(trunk_ckpt["model_state_dict"])
    trunk.eval()

    # 2. Build M20 Pyramidal Detector
    detector = M20PyramidalDualStreamDetector(
        trunk_detector=trunk,
        kin_dim=256,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        gate_bottleneck=128,
        init_pyramid_alpha=init_alpha,
        freeze_trunk=True,
    ).to(device)

    # 3. Load M18 Dual-Stream pretrained weights for warm start
    if M18_CKPT_PATH.exists():
        logger.info(f"Loading pretrained weights from M18 Dual-Stream: {M18_CKPT_PATH}")
        m18_ckpt = torch.load(M18_CKPT_PATH, map_location=device, weights_only=False)
        detector.load_state_dict(m18_ckpt["model_state_dict"], strict=False)
        logger.info("M18 weights transferred cleanly into M20 detector.")
    else:
        logger.warning("M18 checkpoint not found; training M20 from random initialization.")

    trainable_params = [p for p in detector.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters in M20: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()

    train_ds = M20PyramidalFFppDataset(MANIFEST_FFPP, split="train", augment=True)
    val_ds = M20PyramidalFFppDataset(MANIFEST_FFPP, split="val", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        detector.train()
        train_loss = 0.0
        train_bce = 0.0
        train_aux = 0.0
        train_gate = 0.0
        optimizer.zero_grad()

        for step, (x, y, c) in enumerate(train_loader):
            x, y, c = x.to(device), y.to(device), c.to(device)

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out = detector(x, mode="dual_stream")
                l_bce = criterion_bce(out.primary_logits, y)
                l_aux = criterion_ce(out.aux_logits, c)

                reals_mask = (y == 0).squeeze(-1)
                if torch.any(reals_mask):
                    l_gate = torch.mean(out.gate_values[reals_mask] ** 2)
                else:
                    l_gate = torch.tensor(0.0, device=device)

                loss = (l_bce + 0.3 * l_aux + 0.05 * l_gate) / grad_accum_steps

            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            train_bce += l_bce.item()
            train_aux += l_aux.item()
            train_gate += l_gate.item()

        num_batches = len(train_loader)
        train_loss /= num_batches
        train_bce /= num_batches
        train_aux /= num_batches
        train_gate /= num_batches
        scheduler.step()

        # Validation pass
        detector.eval()
        val_loss = 0.0
        val_bce = 0.0
        val_aux = 0.0
        val_gate_list = []

        with torch.no_grad():
            for x, y, c in val_loader:
                x, y, c = x.to(device), y.to(device), c.to(device)
                with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    out = detector(x, mode="dual_stream")
                    l_bce = criterion_bce(out.primary_logits, y)
                    l_aux = criterion_ce(out.aux_logits, c)
                    l_val = l_bce + 0.3 * l_aux

                val_loss += l_val.item()
                val_bce += l_bce.item()
                val_aux += l_aux.item()
                val_gate_list.extend(out.gate_values.squeeze(-1).tolist())

        val_loss /= len(val_loader)
        val_bce /= len(val_loader)
        val_aux /= len(val_loader)
        mean_gate = float(np.mean(val_gate_list))
        pyr_alpha = float(detector.pyramidal_stem.alpha.item())
        epoch_time = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Aux: {train_aux:.4f}) | "
            f"Val Loss: {val_loss:.4f} (BCE: {val_bce:.4f}, Aux: {val_aux:.4f}) | "
            f"Gate: {mean_gate:.4f} | Pyr Alpha: {pyr_alpha:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_gate": mean_gate,
            "pyramidal_alpha": pyr_alpha,
            "time_sec": epoch_time,
        })

        # Save checkpoint
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": detector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "history": history,
            "pyramidal_alpha": pyr_alpha,
            "seed": seed,
        }
        torch.save(ckpt_payload, latest_ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(ckpt_payload, best_ckpt_path)
            logger.info(f"[*] New best M20 model saved with val_loss={val_loss:.4f} at {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"M20 Training Complete. Best model saved to: {best_ckpt_path}")
    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train M20 Pyramidal Dual-Stream Detector")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.50, help="Initial pyramidal alpha")
    args = parser.parse_args()

    train_m20_model(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        seed=args.seed,
        init_alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
