"""Tests for dataset manifest schema validation and indivisible group-stratified splitting."""

import pandas as pd
import pytest

from src.data.manifest import validate_manifest, split_manifest_by_group


def test_manifest_validation_valid():
    """Verify compliant DataFrame passes validation."""
    df = pd.DataFrame({
        "video_id": ["v1", "v2"],
        "group_id": ["g1", "g2"],
        "video_path": ["/path/v1.mp4", "/path/v2.mp4"],
        "label": [0, 1],
        "split": ["train", "val"],
    })
    validate_manifest(df, check_paths=False)


def test_manifest_validation_duplicate_video_id():
    """Verify duplicate video_id raises ValueError."""
    df = pd.DataFrame({
        "video_id": ["v1", "v1"],
        "group_id": ["g1", "g1"],
        "video_path": ["/path/v1.mp4", "/path/v1.mp4"],
        "label": [0, 0],
    })
    with pytest.raises(ValueError, match="duplicate video_id"):
        validate_manifest(df, check_paths=False)


def test_manifest_validation_empty_group_id():
    """Verify null or empty group_id raises ValueError."""
    df = pd.DataFrame({
        "video_id": ["v1"],
        "group_id": [""],
        "video_path": ["/path/v1.mp4"],
        "label": [0],
    })
    with pytest.raises(ValueError, match="null or empty group_id"):
        validate_manifest(df, check_paths=False)


def test_manifest_validation_invalid_label():
    """Verify label outside {0, 1} raises ValueError."""
    df = pd.DataFrame({
        "video_id": ["v1"],
        "group_id": ["g1"],
        "video_path": ["/path/v1.mp4"],
        "label": [2],
    })
    with pytest.raises(ValueError, match="Manifest labels must be in {0, 1}"):
        validate_manifest(df, check_paths=False)


def test_indivisible_group_stratified_split_zero_leakage():
    """Verify group-stratified split preserves indivisible groups and guarantees zero leakage."""
    # Build synthetic dataset representing 20 source groups
    # Each group has 1 real video and 3 fake videos (like FF++)
    records = []
    for g_idx in range(20):
        gid = f"group_{g_idx:03d}"
        # Real video
        records.append({
            "video_id": f"{gid}_real",
            "group_id": gid,
            "video_path": f"/dummy/{gid}_real.mp4",
            "label": 0,
        })
        # 3 Fake videos derived from same source
        for f_idx in range(3):
            records.append({
                "video_id": f"{gid}_fake_{f_idx}",
                "group_id": gid,
                "video_path": f"/dummy/{gid}_fake_{f_idx}.mp4",
                "label": 1,
            })

    manifest_df = pd.DataFrame(records)
    assert len(manifest_df) == 80  # 20 real + 60 fake

    split_df = split_manifest_by_group(manifest_df, ratios=(0.70, 0.15, 0.15), seed=42)

    # 1. Assert all rows have a valid split
    assert set(split_df["split"]) == {"train", "val", "test"}

    # 2. Assert groups are indivisible (each group appears in exactly ONE split)
    for gid, group_rows in split_df.groupby("group_id"):
        unique_splits = set(group_rows["split"])
        assert len(unique_splits) == 1, f"Group '{gid}' was split across multiple splits: {unique_splits}"

    # 3. Assert zero leakage between splits
    train_groups = set(split_df[split_df["split"] == "train"]["group_id"])
    val_groups = set(split_df[split_df["split"] == "val"]["group_id"])
    test_groups = set(split_df[split_df["split"] == "test"]["group_id"])

    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0

    # 4. Assert stratification preserved class balance in each split
    for s in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == s]
        reals = (sub["label"] == 0).sum()
        fakes = (sub["label"] == 1).sum()
        assert reals > 0, f"Split {s} has no real videos!"
        assert fakes > 0, f"Split {s} has no fake videos!"
        # The true ratio is 1 real : 3 fakes (0.25 real)
        assert abs(reals / len(sub) - 0.25) < 0.10
