"""Unit tests for DFDC dataset adapter."""

import json
from pathlib import Path
import pytest

from src.data.adapters.dfdc import (
    build_dfdc_manifest_from_metadata,
    extract_dfdc_group_id,
)
from src.data.manifest import validate_split_integrity


def test_extract_dfdc_group_id():
    """Verify group_id extraction links fakes to their original pristine video."""
    # Fake with original
    fake_entry = {"label": "FAKE", "original": "orig_source.mp4"}
    assert extract_dfdc_group_id("fake_video.mp4", fake_entry) == "orig_source"

    # Real without original
    real_entry = {"label": "REAL"}
    assert extract_dfdc_group_id("orig_source.mp4", real_entry) == "orig_source"

    # Fake without original (fallback to stem)
    orphan_fake = {"label": "FAKE"}
    assert extract_dfdc_group_id("orphan_fake.mp4", orphan_fake) == "orphan_fake"


def test_build_dfdc_manifest_from_metadata(tmp_path: Path):
    """Verify end-to-end DFDC manifest building across chunk folders."""
    dataset_root = tmp_path / "dfdc"
    part0 = dataset_root / "dfdc_train_part_0"
    part1 = dataset_root / "dfdc_train_part_1"
    part0.mkdir(parents=True)
    part1.mkdir(parents=True)

    # Touch mock videos
    for name in ["r0.mp4", "f0_1.mp4", "f0_2.mp4"]:
        (part0 / name).touch()
    for name in ["r1.mp4", "f1_1.mp4"]:
        (part1 / name).touch()

    meta0 = {
        "r0.mp4": {"label": "REAL"},
        "f0_1.mp4": {"label": "FAKE", "original": "r0.mp4"},
        "f0_2.mp4": {"label": "FAKE", "original": "r0.mp4"},
    }
    meta1 = {
        "r1.mp4": {"label": "REAL"},
        "f1_1.mp4": {"label": "FAKE", "original": "r1.mp4"},
    }

    with open(part0 / "metadata.json", "w") as f:
        json.dump(meta0, f)
    with open(part1 / "metadata.json", "w") as f:
        json.dump(meta1, f)

    df = build_dfdc_manifest_from_metadata(dataset_root, split_ratios=(0.5, 0.5, 0.0), seed=42)

    assert len(df) == 5
    assert (df["dataset_name"] == "DFDC").all()
    # Check that f0_1 and r0 share the same group_id
    r0_group = df[df["video_id"] == "r0"]["group_id"].iloc[0]
    f0_1_group = df[df["video_id"] == "f0_1"]["group_id"].iloc[0]
    assert r0_group == f0_1_group == "r0"

    validate_split_integrity(df)
