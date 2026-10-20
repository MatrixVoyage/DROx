"""Audit manifest group_id lineage and identity validity.

Investigates:
1. What group_id actually represents in FF++ and Celeb-DF.
2. Group size distributions and label compositions (real vs fake).
3. Whether group_id represents single biological identities or pairwise interaction components.
4. Co-occurrence of positive pairs in standard training batches.
5. Implications for Identity/Manipulation Disentanglement (M13.2c).
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"

def main():
    print("=" * 80)
    print("MANIFEST & GROUP-ID SCIENTIFIC LINEAGE AUDIT")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Total rows in manifest: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")

    # 1. Dataset Breakdown
    for ds in ["ffpp", "celebdf"]:
        ds_df = df[df["dataset"] == ds]
        print("\n" + "-" * 60)
        print(f"DATASET AUDIT: {ds.upper()} ({len(ds_df)} samples)")
        print("-" * 60)

        groups = ds_df["group_id"].unique()
        print(f"Total unique group_ids: {len(groups)}")

        # Group size distribution
        group_sizes = ds_df["group_id"].value_counts()
        print(f"Group sizes - Min: {group_sizes.min()}, Max: {group_sizes.max()}, "
              f"Mean: {group_sizes.mean():.2f}, Median: {group_sizes.median():.1f}")
        print(f"Group size quantiles: 25%={group_sizes.quantile(0.25):.1f}, "
              f"50%={group_sizes.quantile(0.5):.1f}, 75%={group_sizes.quantile(0.75):.1f}, "
              f"90%={group_sizes.quantile(0.9):.1f}")

        # Check composition within groups: Real vs Fake
        groups_with_real = set(ds_df[ds_df["label"] == 0]["group_id"].unique())
        groups_with_fake = set(ds_df[ds_df["label"] == 1]["group_id"].unique())
        both = groups_with_real & groups_with_fake
        only_real = groups_with_real - groups_with_fake
        only_fake = groups_with_fake - groups_with_real

        print(f"Groups with BOTH real and fake videos: {len(both)} ({len(both)/len(groups)*100:.1f}%)")
        print(f"Groups with ONLY real videos: {len(only_real)} ({len(only_real)/len(groups)*100:.1f}%)")
        print(f"Groups with ONLY fake videos: {len(only_fake)} ({len(only_fake)/len(groups)*100:.1f}%)")

        # Specific analysis for FF++
        if ds == "ffpp":
            print("\nFF++ Detailed Lineage Analysis:")
            # Sample a few groups
            for sample_grp in list(groups)[:5]:
                grp_rows = ds_df[ds_df["group_id"] == sample_grp]
                vids = grp_rows["video_id"].tolist()
                manip_types = grp_rows["manipulation_type"].tolist()
                labels = grp_rows["label"].tolist()
                print(f"  Group '{sample_grp}' (N={len(grp_rows)}):")
                for vid, m_type, lbl in zip(vids[:6], manip_types[:6], labels[:6]):
                    print(f"    - {vid} (type={m_type}, label={lbl})")
                if len(vids) > 6:
                    print(f"    ... and {len(vids) - 6} more")

            # Check: does a group contain multiple pristine videos?
            orig_per_group = ds_df[ds_df["manipulation_type"] == "original"].groupby("group_id")["video_id"].count()
            multi_orig_groups = orig_per_group[orig_per_group > 1]
            single_orig_groups = orig_per_group[orig_per_group == 1]
            zero_orig_groups = len(groups) - len(orig_per_group)

            print(f"\nPristine video count per group in FF++:")
            print(f"  - Groups with 0 pristine videos: {zero_orig_groups}")
            print(f"  - Groups with exactly 1 pristine video: {len(single_orig_groups)}")
            print(f"  - Groups with >= 2 pristine videos: {len(multi_orig_groups)} (Max: {orig_per_group.max()})")
            if len(multi_orig_groups) > 0:
                print("  Sample groups with >= 2 pristine videos:")
                for grp in multi_orig_groups.index[:3]:
                    orig_vids = ds_df[(ds_df["group_id"] == grp) & (ds_df["manipulation_type"] == "original")]["video_id"].tolist()
                    print(f"    Group '{grp}': Pristine videos = {orig_vids}")

        # Specific analysis for Celeb-DF
        if ds == "celebdf":
            print("\nCeleb-DF Detailed Lineage Analysis:")
            for sample_grp in list(groups)[:5]:
                grp_rows = ds_df[ds_df["group_id"] == sample_grp]
                vids = grp_rows["video_id"].tolist()
                manip_types = grp_rows["manipulation_type"].tolist()
                labels = grp_rows["label"].tolist()
                print(f"  Group '{sample_grp}' (N={len(grp_rows)}):")
                for vid, m_type, lbl in zip(vids[:6], manip_types[:6], labels[:6]):
                    print(f"    - {vid} (type={m_type}, label={lbl})")
                if len(vids) > 6:
                    print(f"    ... and {len(vids) - 6} more")

            # YouTube-real vs Celeb-real
            yt_groups = [g for g in groups if "yt_" in g]
            celeb_groups = [g for g in groups if "celebdf_id" in g]
            print(f"\nCeleb-DF Group Types:")
            print(f"  - Celebrity identity groups ('celebdf_idX'): {len(celeb_groups)}")
            print(f"  - YouTube-real video groups ('yt_X'): {len(yt_groups)}")

    # 2. Split Leakage Audit
    print("\n" + "=" * 60)
    print("SPLIT ISOLATION AUDIT (LEAKAGE CHECK)")
    print("=" * 60)
    for ds in ["ffpp", "celebdf"]:
        ds_df = df[df["dataset"] == ds]
        train_grps = set(ds_df[ds_df["split"] == "train"]["group_id"].unique())
        val_grps = set(ds_df[ds_df["split"] == "val"]["group_id"].unique())
        test_grps = set(ds_df[ds_df["split"] == "test"]["group_id"].unique())

        tv_overlap = train_grps & val_grps
        tt_overlap = train_grps & test_grps
        vt_overlap = val_grps & test_grps

        print(f"Dataset {ds.upper()}:")
        print(f"  Train groups: {len(train_grps)}, Val groups: {len(val_grps)}, Test groups: {len(test_grps)}")
        print(f"  Train & Val overlap: {len(tv_overlap)}")
        print(f"  Train & Test overlap: {len(tt_overlap)}")
        print(f"  Val & Test overlap: {len(vt_overlap)}")
        assert len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0, "CRITICAL: Group leakage detected!"

    # 3. Training Batch Co-Occurrence Simulation
    print("\n" + "=" * 60)
    print("TRAINING BATCH CO-OCCURRENCE SIMULATION")
    print("=" * 60)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"Total training samples: {len(train_df)}")

    # Simulate random batches of size 16 and 32
    for b_size in [16, 32]:
        rng = np.random.default_rng(42)
        n_trials = 1000
        co_occur_counts = []
        for _ in range(n_trials):
            batch_indices = rng.choice(len(train_df), size=b_size, replace=False)
            batch_groups = train_df.iloc[batch_indices]["group_id"].values
            # Number of samples in the batch that have at least one positive pair (same group_id)
            unique_grps, counts = np.unique(batch_groups, return_counts=True)
            n_with_pair = np.sum(counts[counts > 1])
            co_occur_counts.append(n_with_pair)

        pct_batches_with_pairs = np.mean(np.array(co_occur_counts) > 0) * 100
        avg_samples_with_pair = np.mean(co_occur_counts)
        print(f"Batch size {b_size}:")
        print(f"  - Batches with at least one identity pair: {pct_batches_with_pairs:.1f}%")
        print(f"  - Average samples per batch that share an identity pair: {avg_samples_with_pair:.2f} / {b_size}")

if __name__ == "__main__":
    main()
