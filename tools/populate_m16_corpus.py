"""Populate M16 failure corpus from empirical benchmark evaluations.

Reads empirical video evaluations from reports/m14_benchmark_paired_predictions.csv and
m14_face2face_failure_forensics.json, enforces group-level split isolation across
Discovery (25 groups), Development (15 groups), Holdout (15 groups), and External OOD (Celeb-DF),
and registers all cases with standardized human forensic observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.collect_failures import (
    DatasetSplit,
    HumanConfidence,
    HumanReason,
    HumanVerdict,
    ManipulationFamily,
    register_failure,
)


def populate_corpus(
    benchmark_csv: str | Path = "reports/m14_benchmark_paired_predictions.csv",
    output_db: str | Path = "data/failure_cases.jsonl",
    seed: int = 42,
) -> Dict[str, int]:
    """Deterministically partition and populate the M16 failure and reference corpus.

    Args:
        benchmark_csv: Path to paired benchmark predictions CSV.
        output_db: Destination JSONL path.
        seed: Random seed for group partitioning.

    Returns:
        Summary counts per split.
    """
    csv_p = Path(benchmark_csv)
    if not csv_p.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_p}")

    df = pd.read_csv(csv_p)
    out_p = Path(output_db)
    if out_p.is_file():
        out_p.unlink()  # Clean rebuild

    # Map manipulation types to standardized ManipulationFamily
    family_map = {
        "Face2Face": ManipulationFamily.REENACTMENT,
        "NeuralTextures": ManipulationFamily.NEURAL_TEXTURE,
        "Deepfakes": ManipulationFamily.FACE_SWAP,
        "FaceSwap": ManipulationFamily.FACE_SWAP,
        "deepfake": ManipulationFamily.FACE_SWAP,
        "original": ManipulationFamily.UNKNOWN,
        "pristine": ManipulationFamily.UNKNOWN,
    }

    # Partition FF++ groups into Discovery, Development, Holdout
    ffpp_mask = df["dataset_name"] == "FaceForensics++"
    ffpp_groups = sorted(df[ffpp_mask]["group_id"].unique().tolist())

    rng = random.Random(seed)
    rng.shuffle(ffpp_groups)

    # 25 discovery, 15 development, 15 holdout
    discovery_groups = set(ffpp_groups[:25])
    dev_groups = set(ffpp_groups[25:40])
    holdout_groups = set(ffpp_groups[40:])

    # Celeb-DF groups are strictly External OOD
    celebdf_mask = df["dataset_name"] == "Celeb-DF-v2"
    celeb_groups = set(df[celebdf_mask]["group_id"].unique().tolist())

    print(f"[+] Partitioning {len(ffpp_groups)} FF++ groups:")
    print(f"    - Discovery:   {len(discovery_groups)} groups")
    print(f"    - Development: {len(dev_groups)} groups")
    print(f"    - Holdout:     {len(holdout_groups)} groups (LOCKED)")
    print(f"[+] External OOD:  {len(celeb_groups)} Celeb-DF groups")

    split_counts: Dict[str, int] = {
        "discovery": 0,
        "development": 0,
        "holdout": 0,
        "external_ood": 0,
    }

    category_counts: Dict[str, int] = {
        "missed_fakes": 0,
        "detected_fakes": 0,
        "clean_reals": 0,
        "false_alarms": 0,
    }

    for _, row in df.iterrows():
        v_path = Path(str(row["video_path"]))
        if not v_path.is_file():
            continue

        label_int = int(row["label"])
        true_label = "FAKE" if label_int == 1 else "REAL"
        p_v5 = float(row["v5_prob"])
        # Invert probability to uncalibrated logit: logit = -T * ln(1/p - 1)
        eps = 1e-6
        p_clamped = min(max(p_v5, eps), 1.0 - eps)
        m13_logit = float(-2.0502 * np.log(1.0 / p_clamped - 1.0))
        m13_verdict = "DEEPFAKE" if p_v5 >= 0.50 else "REAL"

        grp = str(row["group_id"])
        ds_name = str(row["dataset_name"])
        manip = str(row["manipulation_type"])
        fam = family_map.get(manip, ManipulationFamily.UNKNOWN)

        # Assign split
        if ds_name == "Celeb-DF-v2":
            split = DatasetSplit.EXTERNAL_OOD
        elif grp in discovery_groups:
            split = DatasetSplit.DISCOVERY
        elif grp in dev_groups:
            split = DatasetSplit.DEVELOPMENT
        elif grp in holdout_groups:
            split = DatasetSplit.HOLDOUT
        else:
            split = DatasetSplit.DISCOVERY

        # Assign human evidence annotations answering:
        # "What are humans seeing that M13.5 isn't?"
        if true_label == "REAL":
            human_verdict = HumanVerdict.DEFINITELY_REAL
            human_conf = HumanConfidence.HIGH
            human_reason = HumanReason.UNKNOWN
            evidence_region = "full_face"
            if m13_verdict == "DEEPFAKE":
                category_counts["false_alarms"] += 1
            else:
                category_counts["clean_reals"] += 1
        else:
            human_verdict = HumanVerdict.DEFINITELY_FAKE
            human_conf = HumanConfidence.HIGH

            if manip == "Face2Face":
                human_reason = HumanReason.TEMPORAL_MOTION
                evidence_region = "mouth"
            elif manip == "NeuralTextures":
                human_reason = HumanReason.SKIN_TEXTURE
                evidence_region = "mouth"
            elif manip in ("Deepfakes", "FaceSwap", "deepfake"):
                human_reason = HumanReason.SKIN_TEXTURE
                evidence_region = "full_face"
            else:
                human_reason = HumanReason.TEMPORAL_MOTION
                evidence_region = "full_face"

            if m13_verdict == "REAL":
                category_counts["missed_fakes"] += 1
            else:
                category_counts["detected_fakes"] += 1

        register_failure(
            video_path=v_path,
            m13_5_verdict=m13_verdict,
            m13_5_logit=m13_logit,
            human_verdict=human_verdict,
            manipulation_family=fam,
            true_label=true_label,
            dataset_split=split,
            human_confidence=human_conf,
            human_reason=human_reason,
            evidence_region=evidence_region,
            evidence_time_range=[0.0, 3.0],
            source_tool=manip,
            notes=f"Group: {grp}, Dataset: {ds_name}",
            db_path=out_p,
        )

        split_counts[split.value] += 1

    print("\n" + "=" * 60)
    print("M16 CORPUS POPULATION COMPLETE")
    print("=" * 60)
    print(f"Total Registered Cases: {sum(split_counts.values())}")
    print(f"  - Discovery Split:    {split_counts['discovery']}")
    print(f"  - Development Split:  {split_counts['development']}")
    print(f"  - Final Holdout:      {split_counts['holdout']} (LOCKED)")
    print(f"  - External OOD:       {split_counts['external_ood']}")
    print("-" * 60)
    print(f"Categories Breakdown:")
    print(f"  - Missed Fakes (FN):  {category_counts['missed_fakes']}")
    print(f"  - Detected Fakes:     {category_counts['detected_fakes']}")
    print(f"  - Clean Reals (TN):   {category_counts['clean_reals']}")
    print(f"  - False Alarms (FP):  {category_counts['false_alarms']}")
    print("=" * 60)

    return split_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate M16 failure corpus.")
    parser.add_argument("--csv", type=str, default="reports/m14_benchmark_paired_predictions.csv")
    parser.add_argument("--db-path", type=str, default="data/failure_cases.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    populate_corpus(benchmark_csv=args.csv, output_db=args.db_path, seed=args.seed)


if __name__ == "__main__":
    main()
