"""Build Leave-One-Family-Out (LOFO) manifests for Milestone 13.3.

Generates 4 systematic LOFO manifests across the FF++ C23 and Celeb-DF v2 datasets:
- manifests/lofo_f2f.csv (Held-out from train/val: Face2Face)
- manifests/lofo_nt.csv  (Held-out from train/val: NeuralTextures)
- manifests/lofo_fs.csv  (Held-out from train/val: FaceSwap)
- manifests/lofo_df.csv  (Held-out from train/val: Deepfakes)

Guarantees:
1. Zero group/actor leakage between train, val, and test splits (tested & enforced).
2. Preserved canonical group assignments (mutual pair grouping for FF++, celebrity ID for Celeb-DF).
3. Test sets retain all manipulation families and pristine data for comprehensive evaluation.
4. Leaves existing frozen manifests completely unmodified.

Usage:
    .venv\\Scripts\\python.exe scripts/build_lofo_manifests.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FFPP_MANIFEST = PROJECT_ROOT / "manifests" / "ffpp_c23.csv"
CELEBDF_MANIFEST = PROJECT_ROOT / "manifests" / "celebdf_v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "manifests"

LOFO_CONFIGS = {
    "f2f": {
        "held_out_manipulation": "Face2Face",
        "description": "Leave-Face2Face-Out: Expression reenactment held out",
        "filename": "lofo_f2f.csv",
    },
    "nt": {
        "held_out_manipulation": "NeuralTextures",
        "description": "Leave-NeuralTextures-Out: Neural rendering reenactment held out",
        "filename": "lofo_nt.csv",
    },
    "fs": {
        "held_out_manipulation": "FaceSwap",
        "description": "Leave-FaceSwap-Out: Graphics/blending face swap held out",
        "filename": "lofo_fs.csv",
    },
    "df": {
        "held_out_manipulation": "Deepfakes",
        "description": "Leave-Deepfakes-Out: Autoencoder face replacement held out",
        "filename": "lofo_df.csv",
    },
}


def validate_zero_group_leakage(df: pd.DataFrame, fold_name: str) -> None:
    """Verify that train, val, and test splits share strictly zero group_ids."""
    splits = ["train", "val", "test"]
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            g1 = set(df[df["split"] == s1]["group_id"].unique())
            g2 = set(df[df["split"] == s2]["group_id"].unique())
            overlap = g1 & g2
            if overlap:
                raise ValueError(
                    f"[{fold_name}] CRITICAL GROUP LEAKAGE between {s1} and {s2}: "
                    f"{len(overlap)} groups overlap: {sorted(list(overlap))[:5]}..."
                )


def build_single_lofo_manifest(
    ffpp_raw: pd.DataFrame,
    celebdf_raw: pd.DataFrame,
    held_out_manipulation: str,
    fold_name: str,
) -> pd.DataFrame:
    """Construct a single LOFO manifest with held_out_manipulation removed from train/val."""
    # 1. FF++ processing
    ffpp = ffpp_raw.copy()
    ffpp["dataset"] = "ffpp"
    ffpp["group_id"] = "ffpp_" + ffpp["group_id"].astype(str)

    # Filter out held-out manipulation from train and val only (keep in test)
    exclude_mask = (ffpp["manipulation_type"] == held_out_manipulation) & (
        ffpp["split"].isin(["train", "val"])
    )
    ffpp_filtered = ffpp[~exclude_mask].copy()

    # 2. Celeb-DF processing
    celebdf = celebdf_raw.copy()
    celebdf["dataset"] = "celebdf"
    celebdf["group_id"] = "celebdf_" + celebdf["group_id"].astype(str)
    if "manipulation_type" not in celebdf.columns:
        celebdf["manipulation_type"] = celebdf["label"].map({0: "pristine", 1: "deepfake"})

    # 3. Concatenate
    common_cols = [
        "video_id",
        "group_id",
        "video_path",
        "label",
        "dataset_name",
        "manipulation_type",
        "split",
        "dataset",
    ]
    merged = pd.concat([ffpp_filtered[common_cols], celebdf[common_cols]], ignore_index=True)

    # 4. Strict leakage validation
    validate_zero_group_leakage(merged, fold_name)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Build 4-fold LOFO manifests for M13.3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save the generated manifests",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("Milestone 13.3: 4-Fold Leave-One-Family-Out (LOFO) Manifest Generator")
    print("=" * 75)

    assert FFPP_MANIFEST.exists(), f"Missing FF++ manifest: {FFPP_MANIFEST}"
    assert CELEBDF_MANIFEST.exists(), f"Missing Celeb-DF manifest: {CELEBDF_MANIFEST}"

    # Load canonical base manifests
    ffpp_df = pd.read_csv(FFPP_MANIFEST)
    ffpp_df = ffpp_df[ffpp_df["split"].isin(["train", "val", "test"])].copy()

    celebdf_df = pd.read_csv(CELEBDF_MANIFEST)
    celebdf_df = celebdf_df[celebdf_df["split"].isin(["train", "val", "test"])].copy()

    manifest_paths = {}

    for fold_key, cfg in LOFO_CONFIGS.items():
        held_out = cfg["held_out_manipulation"]
        desc = cfg["description"]
        fname = cfg["filename"]
        out_path = args.output_dir / fname

        print(f"\nProcessing Fold '{fold_key.upper()}': {desc}")
        print(f"  Excluding '{held_out}' from train and val splits...")

        lofo_df = build_single_lofo_manifest(
            ffpp_raw=ffpp_df,
            celebdf_raw=celebdf_df,
            held_out_manipulation=held_out,
            fold_name=fold_key,
        )

        lofo_df.to_csv(out_path, index=False)
        manifest_paths[fold_key] = out_path

        # Print breakdown
        n_train = (lofo_df["split"] == "train").sum()
        n_val = (lofo_df["split"] == "val").sum()
        n_test = (lofo_df["split"] == "test").sum()
        held_out_test_count = (
            (lofo_df["manipulation_type"] == held_out) & (lofo_df["split"] == "test")
        ).sum()

        print(f"  Saved -> {out_path.name}")
        print(f"  Total samples: {len(lofo_df)} (Train: {n_train}, Val: {n_val}, Test: {n_test})")
        print(f"  Held-out '{held_out}' test samples: {held_out_test_count}")
        print("  Leakage check: PASSED (zero overlapping group_ids).")

    print("\n" + "=" * 75)
    print("All 4 LOFO manifests generated and verified successfully:")
    for k, p in manifest_paths.items():
        print(f"  - [{k.upper()}] {p}")
    print("=" * 75)


if __name__ == "__main__":
    main()
