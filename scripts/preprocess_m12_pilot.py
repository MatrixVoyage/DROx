"""Preprocess videos from the M12 mixed pilot manifest.

Processes all videos in the pilot manifest that don't already have
preprocessed tensors. Outputs go to processed/m12_pilot/{split}/tensors/.

Usage:
    python scripts/preprocess_m12_pilot.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.data.pipeline import VideoFacePipeline
from src.data.storage import ProcessedDataStorage
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("m12_preprocess")

PILOT_MANIFEST = Path("manifests/mixed_robust_train_p20_seed42.csv")
OUTPUT_DIR = Path("processed/m12_pilot")
CONFIG_PATH = Path("configs/default.yaml")


def main() -> None:
    print("=" * 70)
    print("M12 Pilot Preprocessing")
    print("=" * 70)

    # Load config and pilot manifest
    config = load_config(str(CONFIG_PATH))
    pilot = pd.read_csv(PILOT_MANIFEST)

    print(f"Pilot manifest: {len(pilot)} videos")
    print(f"Splits: {dict(pilot['split'].value_counts())}")

    # Initialize pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.hardware.device = device
    pipeline = VideoFacePipeline(config=config)
    storage = ProcessedDataStorage(str(OUTPUT_DIR))

    # Track statistics
    stats = {"success": 0, "skipped": 0, "failed": 0, "total": len(pilot)}
    start_time = time.time()

    for idx, row in pilot.iterrows():
        vid = str(row["video_id"])
        split = str(row["split"])
        label = int(row["label"])
        video_path = str(row["video_path"])
        group_id = str(row["group_id"])

        # Check if already preprocessed
        tensor_path = OUTPUT_DIR / split / "tensors" / f"{vid}.pt"
        if tensor_path.exists():
            stats["skipped"] += 1
            continue

        # Verify source video exists
        if not Path(video_path).is_file():
            logger.warning(f"Video not found: {video_path}")
            stats["failed"] += 1
            continue

        # Process
        try:
            result = pipeline.process_video(
                video_path=video_path,
                video_id=vid,
                group_id=group_id,
                split=split,
            )

            if result.status == "success" and result.tensor is not None:
                # Save tensor and metadata using the updated API
                storage.save(
                    result=result,
                    split=split,
                    config_snapshot=None
                )
                stats["success"] += 1
            else:
                logger.warning(
                    f"[{vid}] Status: {result.status} — {result.error_message}"
                )
                stats["failed"] += 1

        except Exception as e:
            logger.error(f"[{vid}] Exception: {e}")
            stats["failed"] += 1

        # Progress update
        processed = stats["success"] + stats["skipped"] + stats["failed"]
        if processed % 20 == 0 or processed == stats["total"]:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (stats["total"] - processed) / rate if rate > 0 else 0
            print(
                f"  [{processed}/{stats['total']}] "
                f"success={stats['success']} skipped={stats['skipped']} "
                f"failed={stats['failed']} "
                f"({rate:.1f} vid/s, ~{remaining/60:.0f}m remaining)"
            )

    elapsed = time.time() - start_time
    print(f"\nPreprocessing complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Total:   {stats['total']}")

    # Save config used
    config_out = OUTPUT_DIR / "config.yaml"
    config_out.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(CONFIG_PATH), str(config_out))

    print(f"\nConfig saved to: {config_out}")


if __name__ == "__main__":
    main()
