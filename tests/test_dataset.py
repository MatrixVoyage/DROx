"""Unit and integration tests for ProcessedVideoDataset and DataLoader."""

import json
from pathlib import Path
import pandas as pd
import pytest
import torch

from src.data.dataset import ProcessedVideoDataset, create_dataloader
from src.data.manifest import validate_split_integrity
from src.data.storage import ProcessedDataStorage
from src.data.types import PreprocessResult


def test_validate_split_integrity_success():
    """Verify clean group split passes integrity check."""
    df = pd.DataFrame([
        {"video_id": "v1", "group_id": "g1", "label": 0, "split": "train"},
        {"video_id": "v2", "group_id": "g1", "label": 1, "split": "train"},
        {"video_id": "v3", "group_id": "g2", "label": 0, "split": "val"},
        {"video_id": "v4", "group_id": "g3", "label": 1, "split": "test"},
    ])
    # Should not raise
    validate_split_integrity(df)


def test_validate_split_integrity_leakage():
    """Verify group leakage between any splits raises explicit ValueError."""
    # Leakage between train and val
    df_leak1 = pd.DataFrame([
        {"video_id": "v1", "group_id": "g1", "label": 0, "split": "train"},
        {"video_id": "v2", "group_id": "g1", "label": 1, "split": "val"},
        {"video_id": "v3", "group_id": "g2", "label": 0, "split": "test"},
    ])
    with pytest.raises(ValueError, match="Group leakage detected across splits: train and val share"):
        validate_split_integrity(df_leak1)

    # Leakage between val and test
    df_leak2 = pd.DataFrame([
        {"video_id": "v1", "group_id": "g1", "label": 0, "split": "train"},
        {"video_id": "v2", "group_id": "g2", "label": 1, "split": "val"},
        {"video_id": "v3", "group_id": "g2", "label": 0, "split": "test"},
    ])
    with pytest.raises(ValueError, match="Group leakage detected across splits: val and test share"):
        validate_split_integrity(df_leak2)


def test_dataset_loading_and_shape_contract(tmp_path: Path):
    """Verify ProcessedVideoDataset enforces (T, 3, 112, 112) tensor and (1,) target."""
    storage_dir = tmp_path / "processed"
    storage = ProcessedDataStorage(storage_dir)

    # Create synthetic preprocessed sequences for train split
    tensor1 = torch.randn(16, 3, 112, 112, dtype=torch.float32)
    res1 = PreprocessResult(
        video_id="001",
        group_id="g1",
        split="train",
        status="success",
        tensor=tensor1,
        valid_frames=16,
    )
    storage.save(res1, split="train")

    tensor2 = torch.randn(16, 3, 112, 112, dtype=torch.float32)
    res2 = PreprocessResult(
        video_id="002",
        group_id="g1",
        split="train",
        status="success",
        tensor=tensor2,
        valid_frames=16,
    )
    storage.save(res2, split="train")

    manifest_df = pd.DataFrame([
        {"video_id": "001", "group_id": "g1", "video_path": "fake/path/1.mp4", "label": 0, "split": "train"},
        {"video_id": "002", "group_id": "g1", "video_path": "fake/path/2.mp4", "label": 1, "split": "train"},
        {"video_id": "003", "group_id": "g2", "video_path": "fake/path/3.mp4", "label": 0, "split": "val"},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    dataset = ProcessedVideoDataset(
        manifest=manifest_path,
        split="train",
        processed_dir=storage_dir,
        expected_num_frames=16,
    )
    assert len(dataset) == 2

    sample_tensor, sample_target, vid = dataset[0]
    assert isinstance(sample_tensor, torch.Tensor)
    assert sample_tensor.shape == (16, 3, 112, 112)
    assert sample_tensor.dtype == torch.float32
    assert sample_target.shape == (1,)
    assert sample_target.dtype == torch.float32
    assert sample_target.item() == 0.0
    assert vid == "001"

    # Verify DataLoader batching
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch_tensors, batch_targets, batch_vids = next(iter(loader))
    assert batch_tensors.shape == (2, 16, 3, 112, 112)
    assert batch_tensors.dtype == torch.float32
    assert batch_targets.shape == (2, 1)
    assert batch_targets.dtype == torch.float32
    assert list(batch_vids) == ["001", "002"]



def test_dataset_metadata_cross_validation_rejects_mismatch(tmp_path: Path):
    """Verify that dataset loading strictly rejects metadata mismatch against manifest."""
    storage_dir = tmp_path / "processed"
    storage = ProcessedDataStorage(storage_dir)

    tensor = torch.randn(16, 3, 112, 112, dtype=torch.float32)
    res = PreprocessResult(
        video_id="v_test",
        group_id="g_test",
        split="train",
        status="success",
        tensor=tensor,
        valid_frames=16,
    )
    _, meta_path = storage.save(res, split="train")


    # Tamper with metadata to create a video_id mismatch
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    meta_data["video_id"] = "tampered_id"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    manifest_df = pd.DataFrame([
        {"video_id": "v_test", "group_id": "g_test", "video_path": "fake.mp4", "label": 1, "split": "train"},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="Metadata video_id mismatch"):
        ProcessedVideoDataset(manifest=manifest_path, split="train", processed_dir=storage_dir)



def test_dataset_empty_split_guard(tmp_path: Path):
    """Verify ValueError is raised if requested split has 0 samples in manifest."""
    manifest_df = pd.DataFrame([
        {"video_id": "v1", "group_id": "g1", "video_path": "fake.mp4", "label": 0, "split": "train"},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="contains zero samples"):
        ProcessedVideoDataset(manifest=manifest_path, split="val", processed_dir=tmp_path / "processed")
