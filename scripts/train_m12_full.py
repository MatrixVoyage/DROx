"""M12 Mixed-Domain Full Training Script.

Executes the M12.3 full scientific training run:
- Architecture: EfficientNet-B0 + Temporal Transformer
- Dataset: 100% eligible mixed manifest (FF++ [held-out Face2Face] + Celeb-DF v2)
- Robust Augmentation enabled (sequence-consistent spatial degradations, temporal perturbations)
- Group-level BalancedDatasetBatchSampler (50/50 dataset mix, 50/50 real/fake)
- Strict validation-only calibration and checkpointing via trainer.fit()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProcessedVideoDataset, create_dataloader
from src.data.samplers import BalancedDatasetBatchSampler
from src.models.detector import DeepfakeDetector
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("m12_full_train")

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "m12_full.yaml"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
PROCESSED_DIR = PROJECT_ROOT / "processed" / "m12_full"


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
        logger.warning(
            f"Filtered out {skipped_count} manifest rows whose preprocessed tensors were missing."
        )
    return pd.DataFrame(valid_rows)


def main() -> None:
    print("=" * 70)
    print("M12.3 Mixed-Domain Full Training Run")
    print("=" * 70)

    # 1. Load config
    logger.info(f"Loading configuration from {CONFIG_PATH}")
    config = load_config(str(CONFIG_PATH))

    device = torch.device(config.get_effective_device())
    logger.info(f"Using device: {device}")

    # 2. Setup datasets
    logger.info(f"Loading full manifest from {MANIFEST_PATH}")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    filtered_df = filter_manifest_to_existing(manifest_df, PROCESSED_DIR)

    logger.info("Initializing ProcessedVideoDataset (train)")
    train_dataset = ProcessedVideoDataset(
        manifest=filtered_df,
        split="train",
        processed_dir=PROCESSED_DIR,
        expected_num_frames=config.data.sampling.num_frames,
    )

    logger.info("Initializing ProcessedVideoDataset (val)")
    val_dataset = ProcessedVideoDataset(
        manifest=filtered_df,
        split="val",
        processed_dir=PROCESSED_DIR,
        expected_num_frames=config.data.sampling.num_frames,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 3. Setup Balanced Sampler for training
    train_labels = [s["label"] for s in train_dataset.samples]
    vid_to_dataset = dict(zip(filtered_df["video_id"].astype(str), filtered_df["dataset"]))
    train_datasets = [vid_to_dataset[s["video_id"]] for s in train_dataset.samples]

    logger.info(
        f"Creating BalancedDatasetBatchSampler with batch_size={config.training.batch_size}"
    )
    train_batch_sampler = BalancedDatasetBatchSampler(
        labels=train_labels,
        dataset_ids=train_datasets,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    # 5. Initialize Model
    logger.info(
        f"Initializing DeepfakeDetector:\n"
        f"  - Backbone:  {config.model.backbone.name} ({config.model.backbone.freeze_strategy})\n"
        f"  - Temporal:  {config.model.temporal.type}\n"
        f"  - Attention: {config.model.attention.type}"
    )
    model = DeepfakeDetector(config=config.model)
    model = model.to(device)

    # 6. Initialize Trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # 7. Start Training Loop
    logger.info(f"Starting full training for {config.training.num_epochs} epochs using trainer.fit()")
    
    fit_results = trainer.fit()
    
    best_ckpt = Path(config.training.checkpoint_dir) / "best.pth"
    if best_ckpt.exists():
        logger.info(f"M12.3 Full Training Complete. Best checkpoint saved to {best_ckpt}")
    else:
        logger.error("Training completed but best checkpoint was not found!")
    
    print("\nTraining completed.")


if __name__ == "__main__":
    main()
