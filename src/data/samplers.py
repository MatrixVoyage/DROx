"""Balanced batch sampling and temporally-consistent sequence augmentations."""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Sequence, Union
import numpy as np
import torch
from torch.utils.data import Sampler
import torchvision.transforms.functional as TF


class BalancedBatchSampler(Sampler[List[int]]):
    """Batch sampler that yields batches with an exact 1:1 ratio of Real (0) and Fake (1) samples.

    Ensures that every generated mini-batch contains exactly `batch_size // 2` real samples
    and `batch_size // 2` fake samples. This completely eliminates class prevalence bias
    during training regardless of the underlying dataset imbalance.

    Args:
        labels: Sequence of binary integer labels (0 for pristine, 1 for manipulated).
        batch_size: Even integer batch size.
        shuffle: Whether to shuffle sample indices at the start of each epoch.
        seed: Optional random seed for reproducible sampling.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ):
        if batch_size <= 0 or batch_size % 2 != 0:
            raise ValueError(f"batch_size must be a positive even integer, got {batch_size}")

        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.real_indices = np.where(self.labels == 0)[0]
        self.fake_indices = np.where(self.labels == 1)[0]

        if len(self.real_indices) == 0:
            raise ValueError("Dataset contains zero negative (real) samples; cannot balance batch.")
        if len(self.fake_indices) == 0:
            raise ValueError("Dataset contains zero positive (fake) samples; cannot balance batch.")

        # Total batches per epoch is determined by the larger class (with oversampling of minority)
        # or by the number of steps required to see the majority class.
        self.num_batches = max(
            math.ceil(len(self.real_indices) / self.half_batch),
            math.ceil(len(self.fake_indices) / self.half_batch),
        )

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic RNG seeding."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch if self.seed is not None else None)

        # Prepare index pools
        real_pool = self.real_indices.copy()
        fake_pool = self.fake_indices.copy()

        if self.shuffle:
            rng.shuffle(real_pool)
            rng.shuffle(fake_pool)

        # Oversample minority class to match required steps
        total_real_needed = self.num_batches * self.half_batch
        total_fake_needed = self.num_batches * self.half_batch

        if len(real_pool) < total_real_needed:
            repeats = math.ceil(total_real_needed / len(real_pool))
            real_seq = np.concatenate([real_pool for _ in range(repeats)])[:total_real_needed]
        else:
            real_seq = real_pool[:total_real_needed]

        if len(fake_pool) < total_fake_needed:
            repeats = math.ceil(total_fake_needed / len(fake_pool))
            fake_seq = np.concatenate([fake_pool for _ in range(repeats)])[:total_fake_needed]
        else:
            fake_seq = fake_pool[:total_fake_needed]

        for b in range(self.num_batches):
            batch_real = real_seq[b * self.half_batch : (b + 1) * self.half_batch].tolist()
            batch_fake = fake_seq[b * self.half_batch : (b + 1) * self.half_batch].tolist()

            batch = batch_real + batch_fake
            if self.shuffle:
                rng.shuffle(batch)
            yield batch


class SequenceAugmentor:
    """Temporally-consistent sequence augmentor for face crop tensors (T, C, H, W).

    CRITICAL INVARIANT:
    All frames in a given sequence must undergo IDENTICAL spatial transformations
    to prevent introducing artificial inter-frame temporal discontinuities.
    """

    def __init__(
        self,
        horizontal_flip_p: float = 0.5,
        color_jitter_p: float = 0.3,
        brightness_factor_range: tuple[float, float] = (0.85, 1.15),
        contrast_factor_range: tuple[float, float] = (0.85, 1.15),
        is_training: bool = True,
    ):
        self.horizontal_flip_p = horizontal_flip_p
        self.color_jitter_p = color_jitter_p
        self.brightness_factor_range = brightness_factor_range
        self.contrast_factor_range = contrast_factor_range
        self.is_training = is_training

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply consistent spatial transformations across the entire sequence.

        Args:
            sequence: Tensor of shape (T, C, H, W) or (B, T, C, H, W).

        Returns:
            Augmented tensor of identical shape.
        """
        if not self.is_training:
            return sequence

        has_batch = sequence.dim() == 5
        if not has_batch and sequence.dim() != 4:
            raise ValueError(f"Expected 4D (T, C, H, W) or 5D (B, T, C, H, W), got shape {tuple(sequence.shape)}")

        if has_batch:
            # Process each sequence in batch independently
            out_seqs = [self._augment_single_sequence(s) for s in sequence]
            return torch.stack(out_seqs, dim=0)
        else:
            return self._augment_single_sequence(sequence)

    def _augment_single_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply identical transform parameters to all frames in a single (T, C, H, W) sequence."""
        t, c, h, w = seq.shape

        # 1. Decide horizontal flip once for this sequence
        do_hflip = bool(torch.rand(1).item() < self.horizontal_flip_p)

        # 2. Decide color jitter once for this sequence
        do_jitter = bool(torch.rand(1).item() < self.color_jitter_p)
        if do_jitter:
            b_low, b_high = self.brightness_factor_range
            c_low, c_high = self.contrast_factor_range
            bright_factor = float(b_low + (b_high - b_low) * torch.rand(1).item())
            contrast_factor = float(c_low + (c_high - c_low) * torch.rand(1).item())
        else:
            bright_factor = 1.0
            contrast_factor = 1.0

        if not do_hflip and not do_jitter:
            return seq

        # Apply identical transform to all frames
        augmented_frames = []
        for i in range(t):
            frame = seq[i]
            if do_hflip:
                frame = TF.hflip(frame)
            if do_jitter:
                frame = TF.adjust_brightness(frame, bright_factor)
                frame = TF.adjust_contrast(frame, contrast_factor)
            augmented_frames.append(frame)

        return torch.stack(augmented_frames, dim=0)


class BalancedDatasetBatchSampler(Sampler[List[int]]):
    """Batch sampler guaranteeing 50/50 dataset balance AND real/fake balance within each half.

    For a batch of size B=16 with two datasets (e.g., FF++ and Celeb-DF):
        - 8 samples from dataset A (4 real + 4 fake)
        - 8 samples from dataset B (4 real + 4 fake)

    This operates strictly at the video/group level and never constructs a batch
    by independently sampling frames from different members of the same group.

    Args:
        labels: Sequence of binary integer labels (0=real, 1=fake).
        dataset_ids: Sequence of dataset identifiers (e.g., 'ffpp', 'celebdf').
        batch_size: Total batch size (must be divisible by 4).
        shuffle: Whether to shuffle within each pool at each epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        labels: Sequence[int],
        dataset_ids: Sequence[str],
        batch_size: int = 16,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ):
        if batch_size < 4 or batch_size % 4 != 0:
            raise ValueError(
                f"batch_size must be >= 4 and divisible by 4, got {batch_size}"
            )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        labels_arr = np.asarray(labels, dtype=np.int64)
        datasets_arr = np.asarray(dataset_ids)

        unique_datasets = sorted(set(datasets_arr))
        if len(unique_datasets) != 2:
            raise ValueError(
                f"Expected exactly 2 datasets, got {len(unique_datasets)}: {unique_datasets}"
            )

        self.dataset_a, self.dataset_b = unique_datasets
        self.quarter = batch_size // 4  # samples per (dataset, label) slot

        # Build 4 index pools: (dataset_a, real), (dataset_a, fake), (dataset_b, real), (dataset_b, fake)
        self.pools = {}
        for ds in [self.dataset_a, self.dataset_b]:
            for lbl in [0, 1]:
                key = (ds, lbl)
                mask = (datasets_arr == ds) & (labels_arr == lbl)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    raise ValueError(
                        f"Pool ({ds}, label={lbl}) is empty. Cannot construct balanced batches."
                    )
                self.pools[key] = indices

        # Number of batches per epoch: determined by largest pool (with oversampling)
        max_pool_size = max(len(v) for v in self.pools.values())
        self.num_batches = math.ceil(max_pool_size / self.quarter)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic RNG seeding."""
        self.epoch = epoch

    def _prepare_pool(
        self, pool: np.ndarray, rng: np.random.RandomState, total_needed: int
    ) -> np.ndarray:
        """Shuffle and oversample a pool to fill the required number of slots."""
        p = pool.copy()
        if self.shuffle:
            rng.shuffle(p)
        if len(p) < total_needed:
            repeats = math.ceil(total_needed / len(p))
            p = np.concatenate([p] * repeats)
        return p[:total_needed]

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(
            self.seed + self.epoch if self.seed is not None else None
        )

        total_per_pool = self.num_batches * self.quarter

        # Prepare all 4 pools
        prepared = {}
        for key, indices in self.pools.items():
            prepared[key] = self._prepare_pool(indices, rng, total_per_pool)

        for b in range(self.num_batches):
            batch = []
            start = b * self.quarter
            end = start + self.quarter

            for ds in [self.dataset_a, self.dataset_b]:
                for lbl in [0, 1]:
                    batch.extend(prepared[(ds, lbl)][start:end].tolist())

            if self.shuffle:
                rng.shuffle(batch)

            yield batch
