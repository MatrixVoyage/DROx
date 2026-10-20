"""Tests for ProcessedDataStorage, atomic writes, reload integrity, and rejection invariants."""

from pathlib import Path
import pytest
import torch

from src.utils.config import AppConfig
from src.data.storage import ProcessedDataStorage
from src.data.types import PreprocessResult, VideoErrorCode


def test_storage_successful_save_and_reload(tmp_path: Path):
    """Verify successful PreprocessResult persists tensor and metadata atomically and reloads correctly."""
    storage = ProcessedDataStorage(base_dir=tmp_path)
    dummy_tensor = torch.randn(16, 3, 112, 112, dtype=torch.float32)

    result = PreprocessResult(
        video_id="test_vid_01",
        group_id="group_01",
        split="train",
        tensor=dummy_tensor,
        frame_indices=list(range(16)),
        detected_frames=14,
        previous_bbox_frames=2,
        center_crop_frames=0,
        failed_frames=0,
        valid_frames=16,
        fallback_ratio=2 / 16,
        failure_ratio=0.0,
        status="success",
    )

    config_snapshot = {"seed": 42, "image_size": 112}
    tensor_path, meta_path = storage.save(result, split="train", config_snapshot=config_snapshot)

    assert tensor_path is not None
    assert tensor_path.exists()
    assert meta_path.exists()

    # Verify paths match expected folder hierarchy
    assert tensor_path == tmp_path / "train" / "tensors" / "test_vid_01.pt"
    assert meta_path == tmp_path / "train" / "metadata" / "test_vid_01.json"

    # Verify tensor reloads with exact shape, dtype, and numerical equality
    reloaded_tensor = storage.load_tensor(tensor_path)
    assert reloaded_tensor.shape == (16, 3, 112, 112)
    assert reloaded_tensor.dtype == torch.float32
    assert torch.equal(reloaded_tensor, dummy_tensor)

    # Verify metadata JSON fields
    meta_dict = storage.load_metadata(meta_path)
    assert meta_dict["video_id"] == "test_vid_01"
    assert meta_dict["group_id"] == "group_01"
    assert meta_dict["split"] == "train"
    assert meta_dict["status"] == "success"
    assert meta_dict["frame_indices"] == list(range(16))
    assert meta_dict["counters"]["detected_frames"] == 14
    assert meta_dict["metrics"]["fallback_ratio"] == pytest.approx(2 / 16)
    assert meta_dict["config"]["seed"] == 42


def test_storage_rejected_sequence_invariant(tmp_path: Path):
    """Verify invariant: rejected sequences are never written as successful tensors."""
    storage = ProcessedDataStorage(base_dir=tmp_path)

    rejected_result = PreprocessResult(
        video_id="bad_vid_02",
        group_id="group_02",
        split="val",
        tensor=None,
        frame_indices=[0, 1, 2],
        failed_frames=10,
        valid_frames=6,
        fallback_ratio=0.5,
        failure_ratio=10 / 16,
        status="rejected_quality",
        error_message="Too many failed crops",
    )

    tensor_path, meta_path = storage.save(rejected_result, split="val")

    # Invariant check: tensor_path must be None and no file created in tensors/
    assert tensor_path is None
    tensor_file = tmp_path / "val" / "tensors" / "bad_vid_02.pt"
    assert not tensor_file.exists()

    # Rejected metadata must be saved in rejected_metadata/
    assert meta_path.exists()
    assert meta_path == tmp_path / "val" / "rejected_metadata" / "bad_vid_02.json"

    meta_dict = storage.load_metadata(meta_path)
    assert meta_dict["status"] == "rejected_quality"
    assert meta_dict["error_message"] == "Too many failed crops"


def test_storage_save_run_config(tmp_path: Path):
    """Verify run-level configuration YAML is saved in base_dir."""
    storage = ProcessedDataStorage(base_dir=tmp_path)
    cfg = AppConfig(seed=77)
    saved_path = storage.save_run_config(cfg)

    assert saved_path.exists()
    assert saved_path == tmp_path / "config.yaml"
