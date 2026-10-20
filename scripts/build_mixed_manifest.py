"""Build a mixed-domain manifest for M12 training.

This script fuses FF++ C23 (excluding the held-out Face2Face manipulation family)
and Celeb-DF v2 into a unified manifest with strict group-level leakage prevention.

The 20% pilot is sampled at the video/group level with a fixed seed, preserving
real/fake balance and manipulation proportions.

Usage:
    python scripts/build_mixed_manifest.py [--pilot-fraction 0.20] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


FFPP_MANIFEST = Path("manifests/ffpp_c23.csv")
CELEBDF_MANIFEST = Path("manifests/celebdf_v2.csv")
OUTPUT_DIR = Path("manifests")

# The held-out unseen manipulation family (for cross-manipulation evaluation)
HELD_OUT_MANIPULATION = "Face2Face"


def load_and_tag(
    manifest_path: Path,
    dataset_tag: str,
    exclude_manipulation: Optional[str] = None,
) -> pd.DataFrame:
    """Load a manifest and add a 'dataset' column.

    Args:
        manifest_path: Path to the CSV manifest.
        dataset_tag: Value for the 'dataset' column (e.g., 'ffpp', 'celebdf').
        exclude_manipulation: If set, rows with this manipulation_type are removed
                              from train/val splits but kept in test for evaluation.

    Returns:
        Tagged DataFrame.
    """
    df = pd.read_csv(manifest_path)

    # Only keep rows with an assigned split
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    df["dataset"] = dataset_tag

    # Standardize manipulation column
    if "manipulation_type" not in df.columns:
        df["manipulation_type"] = "unknown"

    if exclude_manipulation:
        # Remove from train/val only; keep in test for unseen-manipulation eval
        exclude_mask = (
            (df["manipulation_type"] == exclude_manipulation) &
            (df["split"].isin(["train", "val"]))
        )
        df = df[~exclude_mask].copy()

    return df


def validate_no_group_leakage(df: pd.DataFrame, dataset_tag: str) -> None:
    """Verify zero group leakage between train, val, and test splits."""
    for split_a in ["train", "val", "test"]:
        for split_b in ["train", "val", "test"]:
            if split_a >= split_b:
                continue
            groups_a = set(df[df["split"] == split_a]["group_id"].unique())
            groups_b = set(df[df["split"] == split_b]["group_id"].unique())
            overlap = groups_a & groups_b
            if overlap:
                raise ValueError(
                    f"[{dataset_tag}] Group leakage between {split_a} and {split_b}: "
                    f"{len(overlap)} groups overlap: {sorted(list(overlap))[:5]}..."
                )


def stratified_group_subsample(
    df: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Subsample at the group level, stratified by dataset + label + split.

    This ensures:
    - The pilot preserves real/fake balance per dataset.
    - No frames from the same group are split across pilot and non-pilot.
    - Manipulation proportions are approximately preserved.

    Args:
        df: Full mixed manifest.
        fraction: Fraction of groups to keep (e.g., 0.20).
        seed: Fixed random seed.

    Returns:
        Subsampled DataFrame.
    """
    rng = np.random.RandomState(seed)
    result_parts = []

    # Stratify by (dataset, split, label) to preserve proportions
    for (dataset, split, label), group_df in df.groupby(["dataset", "split", "label"]):
        # Get unique groups in this stratum
        unique_groups = group_df["group_id"].unique()
        n_keep = max(1, int(len(unique_groups) * fraction))

        selected_groups = rng.choice(unique_groups, size=n_keep, replace=False)
        selected_rows = group_df[group_df["group_id"].isin(selected_groups)]
        result_parts.append(selected_rows)

    if not result_parts:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(result_parts, ignore_index=True)


def build_mixed_manifest(
    pilot_fraction: Optional[float] = None,
    seed: int = 42,
) -> dict:
    """Build the mixed-domain manifest.

    Args:
        pilot_fraction: If set, also produce a pilot subset manifest.
        seed: Fixed random seed for pilot sampling.

    Returns:
        Dict with paths to created manifests.
    """
    print("=" * 70)
    print("M12 Mixed-Domain Manifest Builder")
    print("=" * 70)

    # 1. Load and tag datasets
    print(f"\nLoading FF++ C23 manifest: {FFPP_MANIFEST}")
    ffpp = load_and_tag(
        FFPP_MANIFEST,
        dataset_tag="ffpp",
        exclude_manipulation=HELD_OUT_MANIPULATION,
    )
    print(f"  FF++ rows (excluding {HELD_OUT_MANIPULATION} from train/val): {len(ffpp)}")
    print(f"  FF++ splits: {dict(ffpp['split'].value_counts())}")
    print(f"  FF++ labels: {dict(ffpp['label'].value_counts())}")
    print(f"  FF++ manipulations: {dict(ffpp['manipulation_type'].value_counts())}")

    print(f"\nLoading Celeb-DF v2 manifest: {CELEBDF_MANIFEST}")
    celebdf = load_and_tag(CELEBDF_MANIFEST, dataset_tag="celebdf")
    print(f"  Celeb-DF rows: {len(celebdf)}")
    print(f"  Celeb-DF splits: {dict(celebdf['split'].value_counts())}")
    print(f"  Celeb-DF labels: {dict(celebdf['label'].value_counts())}")

    # 2. Validate per-dataset group leakage
    print("\nValidating per-dataset group leakage...")
    validate_no_group_leakage(ffpp, "ffpp")
    validate_no_group_leakage(celebdf, "celebdf")
    print("  PASS: Zero group leakage within each dataset.")

    # 3. Prefix group_ids to prevent cross-dataset collisions
    ffpp["group_id"] = "ffpp_" + ffpp["group_id"].astype(str)
    celebdf["group_id"] = "celebdf_" + celebdf["group_id"].astype(str)

    # 4. Merge
    mixed = pd.concat([ffpp, celebdf], ignore_index=True)

    # 5. Validate combined group leakage
    print("\nValidating combined manifest group leakage...")
    validate_no_group_leakage(mixed, "mixed")
    print("  PASS: Zero group leakage in combined manifest.")

    # 6. Save full mixed manifest
    full_path = OUTPUT_DIR / "mixed_robust_full.csv"
    mixed.to_csv(full_path, index=False)
    print(f"\nFull mixed manifest saved: {full_path}")
    print(f"  Total rows: {len(mixed)}")
    print(f"  Splits: {dict(mixed['split'].value_counts())}")
    print(f"  Labels: {dict(mixed['label'].value_counts())}")
    print(f"  Datasets: {dict(mixed['dataset'].value_counts())}")

    result = {"full": str(full_path)}

    # 7. Build pilot subset if requested
    if pilot_fraction is not None and pilot_fraction > 0:
        print(f"\nBuilding {pilot_fraction*100:.0f}% stratified pilot subset (seed={seed})...")
        pilot = stratified_group_subsample(mixed, fraction=pilot_fraction, seed=seed)

        # Validate pilot leakage
        validate_no_group_leakage(pilot, "pilot")

        pilot_path = OUTPUT_DIR / f"mixed_robust_train_p{int(pilot_fraction*100)}_seed{seed}.csv"
        pilot.to_csv(pilot_path, index=False)

        print(f"  Pilot manifest saved: {pilot_path}")
        print(f"  Pilot rows: {len(pilot)}")
        print(f"  Pilot splits: {dict(pilot['split'].value_counts())}")
        print(f"  Pilot labels: {dict(pilot['label'].value_counts())}")
        print(f"  Pilot datasets: {dict(pilot['dataset'].value_counts())}")

        # Verify proportions
        for split in ["train", "val", "test"]:
            split_df = pilot[pilot["split"] == split]
            if len(split_df) > 0:
                for ds in ["ffpp", "celebdf"]:
                    ds_split = split_df[split_df["dataset"] == ds]
                    n_real = (ds_split["label"] == 0).sum()
                    n_fake = (ds_split["label"] == 1).sum()
                    print(f"    {split}/{ds}: {n_real} real, {n_fake} fake")

        result["pilot"] = str(pilot_path)

    print("\n" + "=" * 70)
    print("Manifest build complete.")
    print("=" * 70)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build M12 mixed-domain manifest")
    parser.add_argument(
        "--pilot-fraction", type=float, default=0.20,
        help="Fraction of groups to include in pilot subset (default: 0.20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for pilot subset sampling (default: 42)",
    )
    args = parser.parse_args()

    build_mixed_manifest(
        pilot_fraction=args.pilot_fraction,
        seed=args.seed,
    )
