"""Celeb-DF (v2) dataset adapter for group_id extraction, test-list parsing, and manifest generation."""

from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.data.manifest import split_manifest_by_group, validate_manifest, validate_split_integrity


def extract_celebdf_group_id(filename_or_stem: Union[str, Path]) -> str:
    """Extract celebrity source identity group_id from a Celeb-DF v2 video path or filename.

    In Celeb-DF v2:
    - Celeb-real: 'id<source_id>_<clip_id>.mp4' (e.g. 'id0_0000.mp4' -> group_id 'id0')
    - Celeb-synthesis: 'id<source_id>_id<target_id>_<clip_id>.mp4' (e.g. 'id0_id16_0000.mp4' -> group_id 'id0')
    - YouTube-real: '<clip_id>.mp4' (e.g. '00000.mp4' -> group_id 'yt_00000')

    Mapping both real and synthesized derivatives to the primary source identity 'id<source_id>'
    ensures zero group leakage across train, val, and test splits.

    Args:
        filename_or_stem: Filename, stem, or Path of the video.

    Returns:
        The extracted source identity string.
    """
    stem = Path(filename_or_stem).stem.strip()
    if not stem:
        raise ValueError(f"Empty filename or stem: '{filename_or_stem}'")

    if stem.startswith("id"):
        parts = stem.split("_")
        group_id = parts[0].strip()
        if not group_id:
            raise ValueError(f"Unable to extract Celeb-DF group_id from: '{filename_or_stem}'")
        return group_id

    # YouTube-real or non-celebrity numbered clips
    return f"yt_{stem}"


def parse_celebdf_test_list(test_list_path: Union[str, Path]) -> Set[str]:
    """Parse official Celeb-DF v2 List_of_testing_videos.txt.

    Official file format typically contains lines formatted as:
        '1 Celeb-real/id0_0000.mp4'
        '0 Celeb-synthesis/id0_id16_0000.mp4'
        or simply:
        'Celeb-real/id0_0000.mp4'

    Args:
        test_list_path: Path to the testing videos text file.

    Returns:
        Set of video stems that belong to the predefined test set.
    """
    path = Path(test_list_path)
    if not path.is_file():
        raise FileNotFoundError(f"Celeb-DF test list file not found: {path}")

    test_stems: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            parts = line_str.split()
            # If line starts with binary flag (e.g. '1 Celeb-real/id0_0000.mp4'), second token is the path
            rel_path = parts[-1]
            stem = Path(rel_path).stem.strip()
            if stem:
                test_stems.add(stem)

    return test_stems


def build_celebdf_manifest(
    dataset_root: Union[str, Path],
    test_list_path: Optional[Union[str, Path]] = None,
    split_ratios: Tuple[float, float] = (0.85, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Construct a validated manifest DataFrame from a Celeb-DF v2 directory structure.

    Directory structure supported:
        dataset_root/
            Celeb-real/
            Celeb-synthesis/
            YouTube-real/ (optional)

    If test_list_path is provided, test videos are assigned split='test' first.
    The remaining non-test videos are then partitioned into 'train' and 'val'
    with indivisible source group stratification (zero group leakage).

    Args:
        dataset_root: Path to the Celeb-DF v2 root directory.
        test_list_path: Optional path to List_of_testing_videos.txt.
        split_ratios: (train_ratio, val_ratio) for non-test videos (must sum to 1.0).
        seed: Random seed for group splitting.

    Returns:
        DataFrame conforming to the manifest schema.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Celeb-DF dataset root not found: {root}")

    test_stems = parse_celebdf_test_list(test_list_path) if test_list_path else set()

    records = []
    # Search for videos (.mp4, .avi, .mov)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    categories = [
        ("Celeb-real", 0, "pristine"),
        ("YouTube-real", 0, "pristine"),
        ("Celeb-synthesis", 1, "deepfake"),
    ]

    for subfolder, label, manip in categories:
        folder_path = root / subfolder
        if not folder_path.is_dir():
            continue

        for p in folder_path.iterdir():
            if p.is_file() and p.suffix.lower() in video_extensions:
                stem = p.stem
                group_id = extract_celebdf_group_id(stem)
                split = "test" if stem in test_stems else ""

                records.append({
                    "video_id": stem,
                    "group_id": group_id,
                    "video_path": str(p.resolve()),
                    "label": label,
                    "dataset_name": "Celeb-DF-v2",
                    "manipulation_type": manip,
                    "split": split,
                })

    if not records:
        raise ValueError(f"No video files found in Celeb-DF root: {root}")

    df = pd.DataFrame(records)
    validate_manifest(df, check_paths=False)

    # Split non-test records into train and val if test_stems was used
    if test_stems and (df["split"] == "test").any():
        test_mask = df["split"] == "test"
        test_groups = set(df[test_mask]["group_id"].unique())

        # Any non-test video sharing an identity group with the official test set
        # cannot appear in train or val without causing group leakage.
        # Leave these with split="" to preserve zero group leakage and conform to manifest schema.
        holdout_mask = (~test_mask) & (df["group_id"].isin(test_groups))
        df.loc[holdout_mask, "split"] = ""

        # Eligible for train/val are only non-test videos strictly disjoint from test groups
        eligible_mask = (~test_mask) & (~holdout_mask)
        if eligible_mask.any():
            eligible_groups = sorted(list(df[eligible_mask]["group_id"].unique()))
            rng = np.random.RandomState(seed)
            rng.shuffle(eligible_groups)

            total_r = split_ratios[0] + split_ratios[1]
            train_frac = split_ratios[0] / total_r if total_r > 0 else 0.85
            n_train = int(len(eligible_groups) * train_frac)
            # Ensure at least one group in train and val if >= 2 groups exist
            if n_train == 0 and len(eligible_groups) > 1:
                n_train = 1
            elif n_train == len(eligible_groups) and len(eligible_groups) > 1:
                n_train = len(eligible_groups) - 1

            train_groups = set(eligible_groups[:n_train])
            val_groups = set(eligible_groups[n_train:])

            df.loc[eligible_mask & df["group_id"].isin(train_groups), "split"] = "train"
            df.loc[eligible_mask & df["group_id"].isin(val_groups), "split"] = "val"
    elif not test_stems:
        # Default 70/15/15 split
        df = split_manifest_by_group(df, ratios=(0.70, 0.15, 0.15), seed=seed)

    # Validate split integrity on active evaluation splits
    active_splits_df = df[df["split"].isin(["train", "val", "test"])]
    if not active_splits_df.empty:
        validate_split_integrity(active_splits_df)
    return df
