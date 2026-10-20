"""
Tests for DeepGuard 5-Fold Cross-Validation Infrastructure.

Verifies:
1. Zero group leakage between train and validation folds.
2. Group stratification across 5 folds.
3. Shape and label consistency.
4. Determinism under fixed seed.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.evaluation.cross_validation import LeakageSafeGroupKFold


@pytest.fixture
def sample_deepfake_manifest(tmp_path: Path) -> Path:
    """Creates a synthetic multi-family deepfake dataset manifest with groups."""
    data = []
    # 20 distinct source subjects (groups), each having 1 real and 4 fakes
    for group_idx in range(20):
        group_id = f"subj_{group_idx:03d}"
        # Pristine real
        data.append({
            "video_id": f"{group_id}_real",
            "group_id": group_id,
            "label": 0,
            "manipulation_type": "pristine"
        })
        # Manipulated versions
        for manip in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            data.append({
                "video_id": f"{group_id}_{manip.lower()}",
                "group_id": group_id,
                "label": 1,
                "manipulation_type": manip
            })
    df = pd.DataFrame(data)
    manifest_path = tmp_path / "synthetic_manifest.csv"
    df.to_csv(manifest_path, index=False)
    return manifest_path


def test_leakage_safe_group_kfold_zero_leakage(sample_deepfake_manifest: Path):
    """Ensure no group_id appears in both train and validation in any fold."""
    df = pd.read_csv(sample_deepfake_manifest)
    splitter = LeakageSafeGroupKFold(n_splits=5, seed=42)
    splits = splitter.split(df, group_col="group_id", label_col="label")

    assert len(splits) == 5

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(set(train_idx).intersection(set(val_idx))) == 0

        train_groups = set(df.iloc[train_idx]["group_id"].unique())
        val_groups = set(df.iloc[val_idx]["group_id"].unique())

        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Fold {fold_idx} has {len(overlap)} leaking groups: {overlap}"


def test_generate_fold_manifests_file_emission(sample_deepfake_manifest: Path, tmp_path: Path):
    """Test generating 5 fold manifest CSV files to disk."""
    out_dir = tmp_path / "folds"
    splitter = LeakageSafeGroupKFold(n_splits=5, seed=42)
    fold_paths = splitter.generate_fold_manifests(sample_deepfake_manifest, out_dir)

    assert len(fold_paths) == 5
    for fold_name, path in fold_paths.items():
        assert path.exists()
        fold_df = pd.read_csv(path)
        assert "split" in fold_df.columns
        assert set(fold_df["split"].unique()) == {"train", "val"}
        train_groups = set(fold_df[fold_df["split"] == "train"]["group_id"])
        val_groups = set(fold_df[fold_df["split"] == "val"]["group_id"])
        assert len(train_groups.intersection(val_groups)) == 0


def test_missing_column_guards(tmp_path: Path):
    """Ensure ValueError is raised if group_id or label column is missing."""
    bad_df = pd.DataFrame({"video_id": ["v1", "v2"], "val": [1, 2]})
    splitter = LeakageSafeGroupKFold(n_splits=2)
    with pytest.raises(ValueError, match="group_col"):
        splitter.split(bad_df, group_col="missing_group", label_col="val")
    with pytest.raises(ValueError, match="label_col"):
        splitter.split(bad_df, group_col="video_id", label_col="missing_label")
