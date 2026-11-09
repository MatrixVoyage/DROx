"""Authoritative PyTorch Dataset and DataLoader for preprocessed video face sequences."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.manifest import load_manifest, validate_manifest, validate_split_integrity
from src.data.storage import ProcessedDataStorage


class ProcessedVideoDataset(Dataset):
    """Manifest-authoritative PyTorch Dataset for preprocessed video sequences.

    Invariants enforced:
        - Manifest is the sole source of truth for split membership and labels.
        - Directory scanning is never used to infer splits.
        - Saved audit metadata is cross-validated against the manifest row.
        - Mismatched or rejected artifacts raise explicit ValueError.
        - Returned tensor shape is strictly (T, 3, 112, 112), dtype float32.
        - Returned target shape is strictly (1,), dtype float32, value in {0.0, 1.0}.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        split: str,
        processed_dir: Union[str, Path, List[Union[str, Path]]] = "processed",
        expected_num_frames: int = 16,
        return_family_label: bool = False,
        return_group_id: bool = False,
    ):
        """Initialize dataset for a specific split.

        Args:
            manifest: Path to manifest file or loaded pandas DataFrame.
            split: Target split ("train", "val", or "test").
            processed_dir: Base directory or list of directories where ProcessedDataStorage persisted artifacts.
            expected_num_frames: Expected sequence length T (default: 16).
            return_family_label: If True, returns family classification target.
            return_group_id: If True, returns group_id string.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(f"Invalid split: '{split}'. Must be one of {valid_splits}.")

        self.split = split
        self.expected_num_frames = expected_num_frames
        self.return_family_label = return_family_label
        self.return_group_id = return_group_id

        if isinstance(processed_dir, (list, tuple)):
            self.base_dirs = [Path(p) for p in processed_dir]
        else:
            self.base_dirs = [Path(processed_dir)]

        self.storage = ProcessedDataStorage(self.base_dirs[0])

        # Load and validate manifest
        if isinstance(manifest, (str, Path)):
            manifest_df = load_manifest(manifest)
        elif isinstance(manifest, pd.DataFrame):
            manifest_df = manifest.copy()
        else:
            raise TypeError(f"manifest must be a Path, str, or pd.DataFrame, got {type(manifest)}")

        validate_manifest(manifest_df, check_paths=False)
        validate_split_integrity(manifest_df)

        # Filter strictly by the requested split
        self.df = manifest_df[manifest_df["split"] == self.split].reset_index(drop=True)

        # Empty split guard
        if len(self.df) == 0:
            raise ValueError(f"Split '{self.split}' contains zero samples in the manifest.")

        # Validate storage paths and cross-validate metadata against manifest
        self.samples: List[Dict[str, Any]] = []
        for idx, row in self.df.iterrows():
            vid = str(row["video_id"])
            lbl = float(row["label"])

            tensor_path = None
            meta_path = None
            for bdir in self.base_dirs:
                tp = bdir / self.split / "tensors" / f"{vid}.pt"
                mp = bdir / self.split / "metadata" / f"{vid}.json"
                if tp.exists() and mp.exists():
                    tensor_path = tp
                    meta_path = mp
                    break

            if tensor_path is None or meta_path is None:
                raise ValueError(
                    f"Missing preprocessed tensor for video_id '{vid}' in split '{self.split}' across search dirs: {self.base_dirs}"
                )

            # Cross-validate metadata against manifest row
            meta = self.storage.load_metadata(meta_path)
            if str(meta.get("video_id")) != vid:
                raise ValueError(
                    f"Metadata video_id mismatch for '{vid}': expected '{vid}', got '{meta.get('video_id')}'"
                )
            if "label" in meta and meta["label"] is not None and float(meta["label"]) != lbl:
                raise ValueError(
                    f"Metadata label mismatch for '{vid}': manifest has {lbl}, metadata has {meta.get('label')}"
                )
            meta_split = meta.get("split")
            if meta_split and meta_split != self.split:
                raise ValueError(
                    f"Metadata split mismatch for '{vid}': manifest has '{self.split}', metadata has '{meta_split}'"
                )
            if meta.get("status") != "success":
                raise ValueError(
                    f"Sequence for '{vid}' does not have status 'success' (got '{meta.get('status')}'). "
                    "Rejected sequences must not be indexed."
                )

            # Check recorded tensor shape if present
            if "tensor_shape" in meta:
                rec_shape = tuple(meta["tensor_shape"])
                expected_shape = (self.expected_num_frames, 3, 112, 112)
                if rec_shape != expected_shape:
                    raise ValueError(
                        f"Tensor shape mismatch in metadata for '{vid}': expected {expected_shape}, got {rec_shape}"
                    )

            manip_type = str(row.get("manipulation_type", "original"))
            family_map = {
                "original": 0,
                "pristine": 0,
                "Deepfakes": 1,
                "deepfake": 1,
                "Face2Face": 2,
                "FaceSwap": 3,
                "NeuralTextures": 4,
            }
            family_idx = family_map.get(manip_type, 0)

            self.samples.append({
                "video_id": vid,
                "label": lbl,
                "family_label": family_idx,
                "group_id": str(row.get("group_id", vid)),
                "tensor_path": tensor_path,
                "meta_path": meta_path,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        """Load and return sample at idx.

        Returns:
            Tuple according to return_family_label and return_group_id flags.
        """
        sample = self.samples[idx]
        vid = sample["video_id"]
        lbl = sample["label"]

        tensor = self.storage.load_tensor(sample["tensor_path"])

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for '{vid}', got {type(tensor)}")

        if tensor.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (T, C, H, W) for '{vid}', got {tensor.dim()}D tensor with shape {tuple(tensor.shape)}"
            )

        t, c, h, w = tensor.shape
        if t != self.expected_num_frames:
            raise ValueError(f"Expected {self.expected_num_frames} frames for '{vid}', got {t}")
        if (c, h, w) != (3, 112, 112):
            raise ValueError(f"Expected frame shape (3, 112, 112) for '{vid}', got ({c}, {h}, {w})")

        if tensor.dtype != torch.float32:
            raise ValueError(f"Expected tensor dtype torch.float32 for '{vid}', got {tensor.dtype}")

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"Tensor for '{vid}' contains NaN or Inf values")

        target = torch.tensor([lbl], dtype=torch.float32)
        if self.return_family_label and self.return_group_id:
            family_target = torch.tensor(sample["family_label"], dtype=torch.long)
            return tensor, target, vid, family_target, sample["group_id"]
        elif self.return_family_label:
            family_target = torch.tensor(sample["family_label"], dtype=torch.long)
            return tensor, target, vid, family_target
        elif self.return_group_id:
            return tensor, target, vid, sample["group_id"]

        return tensor, target, vid


def seed_worker(worker_id: int) -> None:
    """Worker seed initialization function for DataLoader determinism."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


def create_dataloader(
    dataset: ProcessedVideoDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
    drop_last: bool = False,
    sampler: Optional[Any] = None,
    batch_sampler: Optional[Any] = None,
) -> DataLoader:
    """Construct PyTorch DataLoader with deterministic seeding and device-gated pin_memory.

    Args:
        dataset: The ProcessedVideoDataset instance.
        batch_size: Mini-batch size (default: 4).
        shuffle: Whether to shuffle samples (typically True for train, False for val/test).
        num_workers: Number of subprocess workers.
        pin_memory: If True and CUDA is available, pin memory in CPU tensors.
        seed: Random seed for DataLoader generator.
        drop_last: Whether to drop the last incomplete batch.
        sampler: Optional custom Sampler instance.
        batch_sampler: Optional custom BatchSampler instance.

    Returns:
        Configured PyTorch DataLoader yielding:
            - batch_tensors: (B, T, 3, 112, 112) float32
            - batch_targets: (B, 1) float32
            - batch_video_ids: list of strings of length B
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    g = torch.Generator()
    g.manual_seed(seed)

    effective_pin_memory = pin_memory and torch.cuda.is_available()

    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
            generator=g,
            worker_init_fn=seed_worker,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        generator=g,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
    )
