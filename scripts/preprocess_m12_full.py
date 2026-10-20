"""Preprocess videos from the M12 mixed full manifest.

Processes videos in the full manifest that don't already have
preprocessed tensors. Outputs go to processed/m12_full/{split}/tensors/.

Usage:
    .venv\\Scripts\\python.exe scripts/preprocess_m12_full.py [--splits train val]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import VideoFacePipeline
from src.data.storage import ProcessedDataStorage
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("m12_preprocess_full")

FULL_MANIFEST = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "m12_full"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess full M12 manifest")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Splits to preprocess (default: train val)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("M12 Full Dataset Preprocessing")
    print(f"Target splits: {args.splits}")
    print("=" * 70)

    # Load config and full manifest
    config = load_config(str(CONFIG_PATH))
    df = pd.read_csv(FULL_MANIFEST)

    # Filter to requested splits
    df = df[df["split"].isin(args.splits)].reset_index(drop=True)
    print(f"Filtered manifest: {len(df)} videos across {args.splits}")
    print(f"Splits breakdown: {dict(df['split'].value_counts())}")

    # Initialize pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.hardware.device = device
    logger.info(f"Initializing VideoFacePipeline on {device}")
    pipeline = VideoFacePipeline(config=config)
    storage = ProcessedDataStorage(str(OUTPUT_DIR))

    # Track statistics
    stats = {"success": 0, "skipped": 0, "failed": 0, "total": len(df)}
    start_time = time.time()

    for idx, row in df.iterrows():
        vid = str(row["video_id"])
        split = str(row["split"])
        label = int(row["label"])
        video_path = str(row["video_path"])
        group_id = str(row["group_id"])

        # Check if already preprocessed
        tensor_path = OUTPUT_DIR / split / "tensors" / f"{vid}.pt"
        meta_path = OUTPUT_DIR / split / "metadata" / f"{vid}.json"
        if tensor_path.exists() and meta_path.exists():
            stats["skipped"] += 1
            continue

        # Verify source video exists
        if not Path(video_path).is_file():
            logger.warning(f"Video not found: {video_path}")
            stats["failed"] += 1
            continue

        # Process video
        try:
            result = pipeline.process_video(
                video_path=video_path,
                video_id=vid,
                group_id=group_id,
                split=split,
            )

            if result.status == "success" and result.tensor is not None:
                storage.save(
                    result=result,
                    split=split,
                    config_snapshot=None,
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

        # Periodic progress update
        processed = stats["success"] + stats["skipped"] + stats["failed"]
        if processed % 25 == 0 or processed == stats["total"]:
            elapsed = time.time() - start_time
            # Only calculate rate based on non-skipped items to give realistic ETA
            actual_processed = stats["success"] + stats["failed"]
            rate = actual_processed / elapsed if elapsed > 0 else 0
            items_left = stats["total"] - processed
            remaining_sec = items_left / rate if rate > 0 else 0
            print(
                f"  [{processed}/{stats['total']}] "
                f"success={stats['success']} skipped={stats['skipped']} "
                f"failed={stats['failed']} "
                f"({rate:.2f} new vid/s, ETA: ~{remaining_sec/60:.1f}m)"
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
    shutil.copy(str(CONFIG_PATH), str(config_out))
    print(f"\nConfig saved to: {config_out}")


if __name__ == "__main__":
    main()
