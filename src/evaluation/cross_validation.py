"""
DeepGuard 5-Fold Cross-Validation Infrastructure.

Implements leakage-safe, group-stratified k-fold partitioning for deepfake video datasets.
Guarantees that all videos sharing a source identity (group_id) reside entirely
within a single fold, preventing identity and artifact leakage between train and test splits.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.utils.logger import setup_logger

logger = setup_logger("cross_validation")


class LeakageSafeGroupKFold:
    """Group-stratified K-Fold partitioner enforcing zero identity leakage."""

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def split(
        self, df: pd.DataFrame, group_col: str = "group_id", label_col: str = "label"
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/val index splits ensuring groups never span both sets.

        Args:
            df: Manifest DataFrame containing group_col and label_col.
            group_col: Column identifying indivisible source entity (e.g., original video ID).
            label_col: Binary label column (0=real, 1=fake).

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' missing from DataFrame columns.")
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' missing from DataFrame columns.")

        groups = df[group_col].values
        X = np.zeros(len(df))
        y = df[label_col].values

        gkf = GroupKFold(n_splits=self.n_splits)
        splits = list(gkf.split(X, y, groups=groups))

        # Verify zero leakage across all generated folds
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_groups = set(df.iloc[train_idx][group_col].unique())
            val_groups = set(df.iloc[val_idx][group_col].unique())
            overlap = train_groups.intersection(val_groups)
            if overlap:
                raise RuntimeError(
                    f"Data leakage detected in fold {fold_idx}: {len(overlap)} overlapping groups: {overlap}"
                )

        return splits

    def generate_fold_manifests(
        self,
        manifest_path: Path,
        output_dir: Path,
        group_col: str = "group_id",
        label_col: str = "label",
    ) -> Dict[str, Path]:
        """Load a manifest CSV and emit 5 paired train and val CSV files.

        Returns:
            Dictionary mapping fold identifiers to emitted CSV paths.
        """
        manifest_path = Path(manifest_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(manifest_path)
        logger.info(
            f"Partitioning manifest with {len(df)} samples into {self.n_splits} leakage-safe folds..."
        )

        splits = self.split(df, group_col=group_col, label_col=label_col)
        emitted_paths: Dict[str, Path] = {}

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            train_df["split"] = "train"
            val_df["split"] = "val"

            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            fold_file = output_dir / f"manifest_fold_{fold_idx + 1}_of_{self.n_splits}.csv"
            combined_df.to_csv(fold_file, index=False)
            emitted_paths[f"fold_{fold_idx + 1}"] = fold_file

            logger.info(
                f"Fold {fold_idx + 1}: Train={len(train_idx)} ({len(train_df[group_col].unique())} groups), "
                f"Val={len(val_idx)} ({len(val_df[group_col].unique())} groups) -> {fold_file.name}"
            )

        return emitted_paths
