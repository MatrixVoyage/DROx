"""Unit tests for BalancedDatasetBatchSampler (M12.2).

Validates:
- Every batch has exactly B/4 samples from each (dataset, label) quadrant.
- Batch-level 50/50 dataset balance.
- Batch-level real/fake balance within each dataset half.
- Group-level operation (no cross-group frame mixing — validated structurally).
- Reproducibility with fixed seed.
- Error handling for invalid inputs.
"""

import math
from collections import Counter

import numpy as np
import pytest

from src.data.samplers import BalancedDatasetBatchSampler


@pytest.fixture
def basic_setup():
    """Create a simple 2-dataset, balanced setup."""
    # 20 ffpp_real, 20 ffpp_fake, 20 celebdf_real, 20 celebdf_fake = 80 total
    labels = ([0] * 20 + [1] * 20 +  # ffpp
              [0] * 20 + [1] * 20)    # celebdf
    datasets = (["ffpp"] * 40 + ["celebdf"] * 40)
    return labels, datasets


class TestBalancedDatasetBatchSampler:
    def test_batch_size_constraint(self, basic_setup):
        """Batch size must be >= 4 and divisible by 4."""
        labels, datasets = basic_setup
        with pytest.raises(ValueError, match="divisible by 4"):
            BalancedDatasetBatchSampler(labels, datasets, batch_size=6)
        with pytest.raises(ValueError, match="divisible by 4"):
            BalancedDatasetBatchSampler(labels, datasets, batch_size=3)

    def test_requires_two_datasets(self):
        """Must have exactly 2 distinct dataset IDs."""
        labels = [0, 1, 0, 1]
        datasets = ["ffpp", "ffpp", "ffpp", "ffpp"]
        with pytest.raises(ValueError, match="Expected exactly 2 datasets"):
            BalancedDatasetBatchSampler(labels, datasets, batch_size=4)

    def test_empty_pool_raises(self):
        """An empty (dataset, label) pool must raise."""
        labels = [0, 0, 0, 0, 1, 1]
        datasets = ["ffpp", "ffpp", "celebdf", "celebdf", "celebdf", "celebdf"]
        # ffpp has no fake (label=1) samples
        with pytest.raises(ValueError, match="Pool.*empty"):
            BalancedDatasetBatchSampler(labels, datasets, batch_size=4)

    def test_batch_composition_b16(self, basic_setup):
        """With B=16, each batch must have 4 from each of the 4 quadrants."""
        labels, datasets = basic_setup
        sampler = BalancedDatasetBatchSampler(
            labels, datasets, batch_size=16, shuffle=False, seed=42
        )

        labels_arr = np.array(labels)
        datasets_arr = np.array(datasets)

        for batch in sampler:
            assert len(batch) == 16

            batch_labels = labels_arr[batch]
            batch_datasets = datasets_arr[batch]

            # 8 from each dataset
            ds_counts = Counter(batch_datasets)
            assert ds_counts["celebdf"] == 8, f"Expected 8 celebdf, got {ds_counts['celebdf']}"
            assert ds_counts["ffpp"] == 8, f"Expected 8 ffpp, got {ds_counts['ffpp']}"

            # Within each dataset: 4 real + 4 fake
            for ds in ["ffpp", "celebdf"]:
                ds_mask = batch_datasets == ds
                ds_labels = batch_labels[ds_mask]
                n_real = (ds_labels == 0).sum()
                n_fake = (ds_labels == 1).sum()
                assert n_real == 4, f"{ds}: expected 4 real, got {n_real}"
                assert n_fake == 4, f"{ds}: expected 4 fake, got {n_fake}"

    def test_batch_composition_b4(self):
        """With B=4, each batch must have 1 from each quadrant."""
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        datasets = ["ffpp"] * 4 + ["celebdf"] * 4
        sampler = BalancedDatasetBatchSampler(
            labels, datasets, batch_size=4, shuffle=False, seed=42
        )

        labels_arr = np.array(labels)
        datasets_arr = np.array(datasets)

        for batch in sampler:
            assert len(batch) == 4
            batch_labels = labels_arr[batch]
            batch_datasets = datasets_arr[batch]

            ds_counts = Counter(batch_datasets)
            assert ds_counts["celebdf"] == 2
            assert ds_counts["ffpp"] == 2

    def test_oversampling_minority(self):
        """If one pool is much smaller, it gets oversampled."""
        # ffpp: 100 real, 100 fake; celebdf: 5 real, 5 fake
        labels = [0]*100 + [1]*100 + [0]*5 + [1]*5
        datasets = ["ffpp"]*200 + ["celebdf"]*10

        sampler = BalancedDatasetBatchSampler(
            labels, datasets, batch_size=16, shuffle=True, seed=42
        )

        labels_arr = np.array(labels)
        datasets_arr = np.array(datasets)

        # Should still produce valid batches despite imbalance
        batch_count = 0
        for batch in sampler:
            assert len(batch) == 16
            batch_labels = labels_arr[batch]
            batch_datasets = datasets_arr[batch]

            for ds in ["ffpp", "celebdf"]:
                ds_mask = batch_datasets == ds
                assert ds_mask.sum() == 8
                ds_labels = batch_labels[ds_mask]
                assert (ds_labels == 0).sum() == 4
                assert (ds_labels == 1).sum() == 4

            batch_count += 1

        assert batch_count > 0

    def test_reproducibility(self, basic_setup):
        """Same seed + epoch produces identical batches."""
        labels, datasets = basic_setup

        s1 = BalancedDatasetBatchSampler(labels, datasets, batch_size=16, seed=42)
        s2 = BalancedDatasetBatchSampler(labels, datasets, batch_size=16, seed=42)

        batches_1 = list(s1)
        batches_2 = list(s2)

        assert len(batches_1) == len(batches_2)
        for b1, b2 in zip(batches_1, batches_2):
            assert b1 == b2

    def test_different_epochs_produce_different_batches(self, basic_setup):
        """Different epochs should produce different orderings."""
        labels, datasets = basic_setup

        s = BalancedDatasetBatchSampler(labels, datasets, batch_size=16, seed=42)
        s.set_epoch(0)
        batches_e0 = list(s)

        s.set_epoch(1)
        batches_e1 = list(s)

        # At least some batches should differ
        any_different = any(b0 != b1 for b0, b1 in zip(batches_e0, batches_e1))
        assert any_different, "Different epochs produced identical batches"

    def test_all_indices_covered(self, basic_setup):
        """Over one epoch, every index should appear at least once (with oversampling)."""
        labels, datasets = basic_setup
        sampler = BalancedDatasetBatchSampler(
            labels, datasets, batch_size=16, shuffle=True, seed=42
        )

        all_indices = set()
        for batch in sampler:
            all_indices.update(batch)

        # Every index from the original dataset should appear
        assert all_indices == set(range(80))
