"""Tests for FaceForensics++ dataset-specific adapter and source identity extraction."""

import pytest
from src.data.adapters.ffpp import extract_ffpp_group_id, build_ffpp_manifest_from_paths


def test_extract_ffpp_group_id():
    """Verify source identity extraction for pristine and manipulated filenames."""
    assert extract_ffpp_group_id("000_003.mp4") == "000"
    assert extract_ffpp_group_id("000.mp4") == "000"
    assert extract_ffpp_group_id("123_456") == "123"
    assert extract_ffpp_group_id("999") == "999"


def test_extract_ffpp_group_id_invalid():
    """Verify empty or invalid names raise ValueError."""
    with pytest.raises(ValueError):
        extract_ffpp_group_id("")


def test_build_ffpp_manifest_from_paths(tmp_path):
    """Verify manifest dataframe generation from video paths."""
    paths = [
        tmp_path / "001.mp4",
        tmp_path / "001_002.mp4",
        tmp_path / "003.mp4",
    ]
    for p in paths:
        p.touch()

    df = build_ffpp_manifest_from_paths(paths)
    assert len(df) == 3
    assert set(df["group_id"]) == {"001", "003"}

    # 001.mp4 should be real (0), 001_002.mp4 should be fake (1)
    row_001 = df[df["video_id"] == "001"].iloc[0]
    assert row_001["label"] == 0
    assert row_001["group_id"] == "001"

    row_001_002 = df[df["video_id"] == "001_002"].iloc[0]
    assert row_001_002["label"] == 1
    assert row_001_002["group_id"] == "001"


def test_build_ffpp_manifest(tmp_path):
    """Verify build_ffpp_manifest creates valid manifest with mutual pair group isolation."""
    from src.data.adapters.ffpp import build_ffpp_manifest
    from src.data.manifest import validate_manifest, validate_split_integrity

    root = tmp_path / "ffpp"
    (root / "original").mkdir(parents=True)
    (root / "Deepfakes").mkdir(parents=True)
    (root / "Face2Face").mkdir(parents=True)

    # Create paired videos: 000 and 003
    (root / "original" / "000.mp4").touch()
    (root / "original" / "003.mp4").touch()
    (root / "Deepfakes" / "000_003.mp4").touch()
    (root / "Deepfakes" / "003_000.mp4").touch()
    (root / "Face2Face" / "000_003.mp4").touch()

    # Create another pair: 001 and 002
    (root / "original" / "001.mp4").touch()
    (root / "original" / "002.mp4").touch()
    (root / "Deepfakes" / "001_002.mp4").touch()

    df = build_ffpp_manifest(root, split_ratios=(0.5, 0.25, 0.25), seed=42)
    assert len(df) == 8
    validate_manifest(df, check_paths=True)
    validate_split_integrity(df)

    # Check that pair (000, 003) maps to the same group_id
    grp_000 = df[df["video_id"] == "orig_000"]["group_id"].iloc[0]
    grp_003 = df[df["video_id"] == "orig_003"]["group_id"].iloc[0]
    grp_df = df[df["video_id"] == "df_000_003"]["group_id"].iloc[0]
    assert grp_000 == grp_003 == grp_df == "000"

