"""M12 Mixed-Domain Pilot Training Script.

Executes the M12.2 pilot run:
- EfficientNet-B0 + Temporal Transformer
- 20% stratified mixed manifest (FF++ [held-out Face2Face] + Celeb-DF v2)
- Robust Augmentation enabled (sequence-consistent)
- Group-level BalancedDatasetBatchSampler (50/50 dataset mix, 50/50 real/fake)
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

logger = setup_logger("m12_pilot_train")

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "m12_pilot.yaml"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_train_p20_seed42.csv"
PROCESSED_DIR = PROJECT_ROOT / "processed" / "m12_pilot"


def main() -> None:
    print("=" * 70)
    print("M12.2 Mixed-Domain Pilot Training")
    print("=" * 70)

    # 1. Load config
    logger.info(f"Loading configuration from {CONFIG_PATH}")
    config = load_config(str(CONFIG_PATH))

    device = torch.device(config.get_effective_device())
    logger.info(f"Using device: {device}")

    # 2. Setup datasets
    logger.info(f"Loading pilot manifest from {MANIFEST_PATH}")
    manifest_df = pd.read_csv(MANIFEST_PATH)

    logger.info("Initializing ProcessedVideoDataset (train)")
    train_dataset = ProcessedVideoDataset(
        manifest=manifest_df,
        split="train",
        processed_dir=PROCESSED_DIR,
        expected_num_frames=config.data.sampling.num_frames,
    )

    logger.info("Initializing ProcessedVideoDataset (val)")
    val_dataset = ProcessedVideoDataset(
        manifest=manifest_df,
        split="val",
        processed_dir=PROCESSED_DIR,
        expected_num_frames=config.data.sampling.num_frames,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 3. Setup Balanced Sampler for training
    # The BalancedDatasetBatchSampler requires (labels, dataset_ids) for all train samples
    # We must extract these directly from the dataset's valid `.samples` list
    train_labels = [s["label"] for s in train_dataset.samples]
    
    # We need the dataset_id (ffpp vs celebdf). It's in the manifest. 
    # Let's map video_id -> dataset_id
    vid_to_dataset = dict(zip(manifest_df["video_id"].astype(str), manifest_df["dataset"]))
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
    # Note: `batch_sampler` is mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`
    # We bypass `create_dataloader` since it doesn't gracefully handle `batch_sampler` vs `batch_size`.
    # Wait, `create_dataloader` has `batch_sampler` argument. But if we pass `batch_size`, it crashes.
    # Let's just create standard DataLoaders.
    
    # For training, use the batch_sampler
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    # For validation, standard inference dataloader
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
    logger.info(f"Starting pilot training for {config.training.num_epochs} epochs using trainer.fit()")
    
    fit_results = trainer.fit()
    
    best_ckpt = Path(config.training.checkpoint_dir) / "best.pth"
    if best_ckpt.exists():
        logger.info(f"M12.2 Pilot Training Complete. Best checkpoint saved to {best_ckpt}")
    else:
        logger.error("Training completed but best checkpoint was not found!")
    
    print("\nTraining completed.")


if __name__ == "__main__":
    main()
