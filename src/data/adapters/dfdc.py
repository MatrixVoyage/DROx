"""DFDC (Deepfake Detection Challenge) dataset adapter for metadata parsing and manifest generation."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from src.data.manifest import split_manifest_by_group, validate_manifest, validate_split_integrity


def extract_dfdc_group_id(filename_or_key: str, entry: Dict[str, Union[str, int]]) -> str:
    """Extract source identity group_id from a DFDC video entry.

    For DFDC:
    - Fake videos contain an 'original' field pointing to the source video (e.g. 'orig.mp4').
      Their group_id is the stem of that original video.
    - Real videos do not have an 'original' field (or original is null/same).
      Their group_id is their own stem.

    This ensures that a fake video and its pristine original share the identical group_id,
    strictly preventing group leakage across train, val, and test splits.

    Args:
        filename_or_key: Video filename (e.g. 'abc.mp4') or stem.
        entry: Dictionary containing metadata fields ('label', 'original', etc.).

    Returns:
        The extracted source identity string.
    """
    stem = Path(filename_or_key).stem.strip()
    original = entry.get("original")
    if original and isinstance(original, str) and original.strip():
        return Path(original).stem.strip()
    return stem


def build_dfdc_manifest_from_metadata(
    dataset_root: Union[str, Path],
    metadata_filename: str = "metadata.json",
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Construct a validated manifest DataFrame from DFDC directory structure and metadata.json files.

    DFDC datasets (full or preview) typically contain one or more folders (e.g. 'dfdc_train_part_0/'),
    each containing .mp4 files and a 'metadata.json' mapping filenames to labels and source videos.

    Args:
        dataset_root: Path to the DFDC root or partition directory.
        metadata_filename: Name of the JSON metadata file (default 'metadata.json').
        split_ratios: (train, val, test) split ratios if split is not predefined in metadata.
        seed: Random seed for group splitting.

    Returns:
        DataFrame conforming to the manifest schema.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"DFDC dataset root not found: {root}")

    # Discover all metadata.json files recursively
    meta_files = list(root.rglob(metadata_filename))
    if not meta_files:
        raise FileNotFoundError(f"No '{metadata_filename}' found under DFDC root: {root}")

    records: List[Dict[str, Union[str, int]]] = []

    for meta_path in meta_files:
        part_dir = meta_path.parent
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data: Dict[str, Dict[str, Union[str, int]]] = json.load(f)

        for filename, entry in meta_data.items():
            stem = Path(filename).stem
            video_path = part_dir / filename
            label_str = str(entry.get("label", "FAKE")).upper().strip()
            label = 1 if label_str == "FAKE" else 0
            group_id = extract_dfdc_group_id(stem, entry)
            manip = "deepfake" if label == 1 else "pristine"
            split = str(entry.get("split", "")).lower().strip()
            if split not in {"train", "val", "test"}:
                split = ""

            records.append({
                "video_id": stem,
                "group_id": group_id,
                "video_path": str(video_path.resolve()) if video_path.exists() else str(video_path),
                "label": label,
                "dataset_name": "DFDC",
                "manipulation_type": manip,
                "split": split,
            })

    if not records:
        raise ValueError(f"No records found in DFDC metadata files under: {root}")

    df = pd.DataFrame(records)
    validate_manifest(df, check_paths=False)

    # Check if splits are already populated across all records
    unique_splits = set(df["split"].unique())
    has_valid_splits = unique_splits.issubset({"train", "val", "test"}) and len(unique_splits) > 1 and ("" not in unique_splits)

    if not has_valid_splits:
        df = split_manifest_by_group(df, ratios=split_ratios, seed=seed)

    validate_split_integrity(df)
    return df
