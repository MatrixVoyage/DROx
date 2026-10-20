"""Preprocessing script for LOFO Face2Face train and val sequences.

Extracts 16-frame face tensors for Face2Face train (404) and val (298) sequences
from Datasets/FaceForensics++_C23/Face2Face into processed/m13_3_lofo/{split}/tensors/.

Usage:
    .venv\\Scripts\\python.exe scripts/preprocess_m13_4_lofo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import VideoFacePipeline
from src.data.storage import ProcessedDataStorage
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("m13_4_preprocess")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "lofo_nt.csv"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "m13_3_lofo"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def main():
    print("=" * 70)
    print("Milestone 13.4: Preprocessing Face2Face Train & Val Sequences for LOFO")
    print("=" * 70)

    config = load_config(str(CONFIG_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.hardware.device = device
    logger.info(f"Initializing VideoFacePipeline on {device}")
    pipeline = VideoFacePipeline(config=config)
    storage = ProcessedDataStorage(str(OUTPUT_DIR))

    df = pd.read_csv(MANIFEST_PATH)
    f2f_df = df[(df["manipulation_type"] == "Face2Face") & (df["split"].isin(["train", "val"]))].reset_index(drop=True)
    print(f"Target Face2Face sequences: {len(f2f_df)} ({dict(f2f_df['split'].value_counts())})")

    stats = {"success": 0, "skipped": 0, "failed": 0, "total": len(f2f_df)}
    start_time = time.time()

    for idx, row in f2f_df.iterrows():
        vid = str(row["video_id"])
        split = str(row["split"])
        video_path = str(row["video_path"])
        group_id = str(row["group_id"])

        tensor_path = OUTPUT_DIR / split / "tensors" / f"{vid}.pt"
        meta_path = OUTPUT_DIR / split / "metadata" / f"{vid}.json"
        if tensor_path.exists() and meta_path.exists():
            stats["skipped"] += 1
            continue

        if not Path(video_path).is_file():
            # Check relative to project root or dataset directory
            alt_path = PROJECT_ROOT / video_path
            if alt_path.is_file():
                video_path = str(alt_path)
            else:
                logger.warning(f"Video not found: {video_path}")
                stats["failed"] += 1
                continue

        try:
            result = pipeline.process_video(
                video_path=video_path,
                video_id=vid,
                group_id=group_id,
                split=split,
            )

            if result.status == "success" and result.tensor is not None:
                storage.save(result=result, split=split, config_snapshot=None)
                stats["success"] += 1
            else:
                logger.warning(f"[{vid}] Status: {result.status} - {result.error_message}")
                stats["failed"] += 1

        except Exception as e:
            logger.error(f"[{vid}] Exception: {e}")
            stats["failed"] += 1

        processed = stats["success"] + stats["skipped"] + stats["failed"]
        if processed % 50 == 0 or processed == stats["total"]:
            elapsed = time.time() - start_time
            actual_processed = stats["success"] + stats["failed"]
            rate = actual_processed / elapsed if elapsed > 0 else 0
            items_left = stats["total"] - processed
            remaining_sec = items_left / rate if rate > 0 else 0
            print(
                f"  [{processed}/{stats['total']}] "
                f"success={stats['success']} skipped={stats['skipped']} failed={stats['failed']} "
                f"({rate:.2f} vid/s, ETA: ~{remaining_sec/60:.1f}m)"
            )

    elapsed = time.time() - start_time
    print(f"\nFace2Face Preprocessing complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Total:   {stats['total']}")


if __name__ == "__main__":
    main()
