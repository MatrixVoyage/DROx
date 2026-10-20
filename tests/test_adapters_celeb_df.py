"""Unit tests for Celeb-DF v2 dataset adapter."""

from pathlib import Path
import pytest

from src.data.adapters.celeb_df import (
    build_celebdf_manifest,
    extract_celebdf_group_id,
    parse_celebdf_test_list,
)
from src.data.manifest import validate_split_integrity


def test_extract_celebdf_group_id():
    """Verify group_id extraction maps real and fake derivatives to the source identity."""
    # Celeb-real
    assert extract_celebdf_group_id("id0_0000.mp4") == "id0"
    assert extract_celebdf_group_id("id15_0012.mp4") == "id15"
    assert extract_celebdf_group_id(Path("/path/to/id3_0001.mp4")) == "id3"

    # Celeb-synthesis (synthesized derivatives)
    assert extract_celebdf_group_id("id0_id16_0000.mp4") == "id0"
    assert extract_celebdf_group_id("id15_id0_0005.mp4") == "id15"

    # YouTube-real
    assert extract_celebdf_group_id("00000.mp4") == "yt_00000"
    assert extract_celebdf_group_id("sample123.mp4") == "yt_sample123"

    # Edge cases
    with pytest.raises(ValueError):
        extract_celebdf_group_id("   ")


def test_parse_celebdf_test_list(tmp_path: Path):
    """Verify parsing of official Celeb-DF List_of_testing_videos.txt."""
    test_list_file = tmp_path / "List_of_testing_videos.txt"
    test_list_file.write_text(
        "# Header comment\n"
        "1 Celeb-real/id0_0000.mp4\n"
        "1 Celeb-real/id0_0001.mp4\n"
        "0 Celeb-synthesis/id0_id16_0000.mp4\n"
        "YouTube-real/00001.mp4\n"
        "\n"
    )

    stems = parse_celebdf_test_list(test_list_file)
    assert stems == {"id0_0000", "id0_0001", "id0_id16_0000", "00001"}


def test_build_celebdf_manifest_with_test_list(tmp_path: Path):
    """Verify end-to-end Celeb-DF manifest creation with test list integration and zero leakage."""
    dataset_root = tmp_path / "celeb_df"
    real_dir = dataset_root / "Celeb-real"
    fake_dir = dataset_root / "Celeb-synthesis"
    yt_dir = dataset_root / "YouTube-real"

    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    yt_dir.mkdir(parents=True)

    # Group id0: 2 real, 2 fake
    (real_dir / "id0_0000.mp4").touch()
    (real_dir / "id0_0001.mp4").touch()
    (fake_dir / "id0_id1_0000.mp4").touch()
    (fake_dir / "id0_id1_0001.mp4").touch()

    # Group id1: 2 real, 2 fake
    (real_dir / "id1_0000.mp4").touch()
    (real_dir / "id1_0001.mp4").touch()
    (fake_dir / "id1_id0_0000.mp4").touch()
    (fake_dir / "id1_id0_0001.mp4").touch()

    # Group id2: 2 real
    (real_dir / "id2_0000.mp4").touch()
    (real_dir / "id2_0001.mp4").touch()

    # Test list contains id2 videos and one id1 fake
    test_list_file = tmp_path / "List_of_testing_videos.txt"
    test_list_file.write_text(
        "1 Celeb-real/id2_0000.mp4\n"
        "1 Celeb-real/id2_0001.mp4\n"
    )

    df = build_celebdf_manifest(dataset_root, test_list_path=test_list_file, split_ratios=(0.5, 0.5), seed=42)

    assert len(df) == 10
    assert (df["split"] == "test").sum() == 2
    assert (df["split"] == "train").sum() > 0
    assert (df["split"] == "val").sum() > 0

    # Ensure zero group leakage
    validate_split_integrity(df)
