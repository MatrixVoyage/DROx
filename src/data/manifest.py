"""Dataset manifest management, validation, and indivisible group-stratified splitting."""

import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["video_id", "group_id", "video_path", "label"]


def validate_manifest(df: pd.DataFrame, check_paths: bool = False) -> None:
    """Validate dataset manifest schema, uniqueness, and constraints.

    Args:
        df: The manifest DataFrame.
        check_paths: If True, asserts each video_path exists on filesystem.

    Raises:
        ValueError: If validation rules are violated.
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Manifest missing required column: '{col}'")

    if df["video_id"].duplicated().any():
        dups = df["video_id"][df["video_id"].duplicated()].tolist()
        raise ValueError(f"Manifest contains duplicate video_id values: {dups[:5]}")

    if ((df["group_id"].isna()) | (df["group_id"].astype(str).str.strip() == "")).any():
        raise ValueError("Manifest contains null or empty group_id values")

    invalid_labels = df[~df["label"].isin([0, 1])]["label"].tolist()
    if invalid_labels:
        raise ValueError(f"Manifest labels must be in {{0, 1}}, found: {set(invalid_labels)}")

    if "split" in df.columns:
        valid_splits = {"train", "val", "test", ""}
        found_splits = set(df["split"].fillna("").astype(str))
        invalid_splits = found_splits - valid_splits
        if invalid_splits:
            raise ValueError(f"Manifest contains invalid split values: {invalid_splits}. Must be in {valid_splits}")

    if check_paths:
        for idx, row in df.iterrows():
            p = Path(row["video_path"])
            if not p.exists():
                raise ValueError(f"Video path does not exist for video_id '{row['video_id']}': {p}")


def split_manifest_by_group(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    check_paths: bool = False,
) -> pd.DataFrame:
    """Perform indivisible group-level stratified splitting across train, val, and test.

    Every group_id is strictly indivisible (all derivative videos sharing a group_id
    are placed into exactly one split), guaranteeing ZERO cross-split data leakage.
    Stratification balances class distribution across splits.

    Args:
        df: Validated or unvalidated manifest DataFrame.
        ratios: Tuple of (train_ratio, val_ratio, test_ratio). Must sum to 1.0.
        seed: Random seed for deterministic group shuffling.
        check_paths: If True, checks file paths during manifest validation.

    Returns:
        DataFrame with an updated 'split' column ('train', 'val', 'test').
    """
    validate_manifest(df, check_paths=check_paths)

    total_ratio = sum(ratios)
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
        raise ValueError(f"Split ratios must sum to 1.0, got sum({ratios}) = {total_ratio}")

    train_r, val_r, test_r = ratios
    split_names = ["train", "val", "test"]
    split_target_ratios = [train_r, val_r, test_r]

    # Compute group-level statistics
    group_stats: List[Dict[str, Any]] = []
    for group_id, grp in df.groupby("group_id"):
        real_count = int((grp["label"] == 0).sum())
        fake_count = int((grp["label"] == 1).sum())
        total_count = len(grp)
        fake_ratio = fake_count / total_count if total_count > 0 else 0.0
        group_stats.append({
            "group_id": group_id,
            "total_count": total_count,
            "real_count": real_count,
            "fake_count": fake_count,
            "fake_ratio": fake_ratio,
        })

    rng = np.random.RandomState(seed)
    rng.shuffle(group_stats)

    # Sort groups primarily by fake_ratio category to facilitate stratified balancing
    # Discretize fake_ratio into bins: purely real (0.0), mixed, purely fake (1.0)
    def strat_key(item: Dict[str, Any]) -> Tuple[int, int]:
        fr = item["fake_ratio"]
        if fr == 0.0:
            cat = 0
        elif fr == 1.0:
            cat = 2
        else:
            cat = 1
        return (cat, -item["total_count"])

    group_stats.sort(key=strat_key)

    # Track allocations per split
    split_counts = {s: {"total": 0, "real": 0, "fake": 0} for s in split_names}
    group_to_split: Dict[Any, str] = {}

    total_manifest_videos = len(df)
    target_counts = {
        s: {
            "total": total_manifest_videos * r,
            "real": int((df["label"] == 0).sum()) * r,
            "fake": int((df["label"] == 1).sum()) * r,
        }
        for s, r in zip(split_names, split_target_ratios)
    }

    # Greedy allocation of indivisible groups
    for g in group_stats:
        best_split = None
        best_score = float("inf")

        for s in split_names:
            curr = split_counts[s]
            tgt = target_counts[s]

            # Projected error if group assigned to split s
            proj_total = curr["total"] + g["total_count"]
            proj_real = curr["real"] + g["real_count"]
            proj_fake = curr["fake"] + g["fake_count"]

            err_total = (proj_total - tgt["total"]) / max(1.0, tgt["total"])
            err_real = (proj_real - tgt["real"]) / max(1.0, tgt["real"])
            err_fake = (proj_fake - tgt["fake"]) / max(1.0, tgt["fake"])

            score = (err_total ** 2) + 1.5 * (err_real ** 2 + err_fake ** 2)

            if score < best_score:
                best_score = score
                best_split = s

        group_to_split[g["group_id"]] = best_split
        split_counts[best_split]["total"] += g["total_count"]
        split_counts[best_split]["real"] += g["real_count"]
        split_counts[best_split]["fake"] += g["fake_count"]

    # Assign split to copy of DataFrame
    result_df = df.copy()
    result_df["split"] = result_df["group_id"].map(group_to_split)

    # Verification: Assert zero group leakage between splits via explicit ValueError
    validate_split_integrity(result_df)

    return result_df


def validate_split_integrity(df: pd.DataFrame) -> None:
    """Validate zero group leakage across train, val, and test splits.

    Guarantees that:
        train_groups ∩ val_groups == ∅
        train_groups ∩ test_groups == ∅
        val_groups ∩ test_groups == ∅

    Raises:
        ValueError: If any group_id appears in multiple splits.
    """
    if "split" not in df.columns or "group_id" not in df.columns:
        raise ValueError("DataFrame must contain 'split' and 'group_id' columns for split integrity validation")

    train_groups: Set[Any] = set(df[df["split"] == "train"]["group_id"].dropna().unique())
    val_groups: Set[Any] = set(df[df["split"] == "val"]["group_id"].dropna().unique())
    test_groups: Set[Any] = set(df[df["split"] == "test"]["group_id"].dropna().unique())

    leakage_errors = []
    train_val = train_groups & val_groups
    if train_val:
        leakage_errors.append(f"train and val share {len(train_val)} groups: {sorted(list(train_val))[:5]}")
    train_test = train_groups & test_groups
    if train_test:
        leakage_errors.append(f"train and test share {len(train_test)} groups: {sorted(list(train_test))[:5]}")
    val_test = val_groups & test_groups
    if val_test:
        leakage_errors.append(f"val and test share {len(val_test)} groups: {sorted(list(val_test))[:5]}")

    if leakage_errors:
        raise ValueError(f"Group leakage detected across splits: {'; '.join(leakage_errors)}")



def load_manifest(path: Union[str, Path]) -> pd.DataFrame:
    """Load manifest from CSV or JSONL, preserving video_id and group_id as strings."""
    p = Path(path)
    str_cols = {"video_id": str, "group_id": str}
    if p.suffix.lower() == ".jsonl":
        return pd.read_json(str(p), lines=True, dtype=str_cols)
    return pd.read_csv(str(p), dtype=str_cols)



def save_manifest(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save manifest to CSV or JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        df.to_json(str(p), orient="records", lines=True)
    else:
        df.to_csv(str(p), index=False)
