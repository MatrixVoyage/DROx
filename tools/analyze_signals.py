"""Step 4: Signal Discriminability Analysis.

Computes AUROC, Cohen's d effect size, and Mann-Whitney U test p-values for all extracted
forensic features across the failure corpus. Only features with AUROC >= 0.65 are marked
as candidate signals for Step 5 Evidence-Gated Fusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def extract_signals_for_video(
    video_path: str | Path,
    face_crops: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run all 10 forensic signal extractors on a single video file."""
    from src.forensics.signals import (
        AudioSignal,
        CompressionSignal,
        FrequencySignal,
        IdentitySignal,
        LandmarkSignal,
        LipSyncSignal,
        MotionSignal,
        ProvenanceSignal,
        SpatialSignal,
        TemporalSignal,
    )

    path = Path(video_path)
    results: Dict[str, Any] = {}

    # Read face crops or extract from video if none provided
    crops = face_crops or []
    if not crops and path.is_file():
        import cv2

        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            indices = set(np.linspace(0, count - 1, min(16, count), dtype=int))
            frame_idx = 0
            while cap.isOpened() and len(crops) < 16:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if frame_idx in indices:
                    h, w = frame.shape[:2]
                    sz = min(h, w) // 2
                    cy, cx = h // 2, w // 2
                    crop = frame[cy - sz // 2 : cy + sz // 2, cx - sz // 2 : cx + sz // 2]
                    if crop.size > 0:
                        crops.append(cv2.resize(crop, (112, 112)))
                frame_idx += 1
        cap.release()

    # 1. Spatial
    if crops:
        s_res = SpatialSignal().extract(crops[0])
        results.update({k: v for k, v in s_res.items() if isinstance(v, (int, float))})

    # 2. Temporal
    if len(crops) >= 2:
        t_res = TemporalSignal().extract(crops)
        results.update({k: v for k, v in t_res.items() if isinstance(v, (int, float))})

    # 3. Motion
    if len(crops) >= 2:
        m_res = MotionSignal().extract(crops)
        results.update({k: v for k, v in m_res.items() if isinstance(v, (int, float))})

    # 4. Frequency
    if crops:
        f_res = FrequencySignal().extract(crops[0])
        results.update({k: v for k, v in f_res.items() if isinstance(v, (int, float))})

    # 5. Landmarks
    if len(crops) >= 2:
        lm_res = LandmarkSignal().extract(crops)
        results.update({k: v for k, v in lm_res.items() if isinstance(v, (int, float))})

    # 6. Audio
    a_res = AudioSignal().extract(path)
    results.update({k: v for k, v in a_res.items() if isinstance(v, (int, float))})

    # 7. Lip Sync
    ls_res = LipSyncSignal().extract(path, crops, a_res)
    results.update({k: v for k, v in ls_res.items() if isinstance(v, (int, float))})

    # 8. Identity
    if len(crops) >= 2:
        id_res = IdentitySignal().extract(crops)
        results.update({k: v for k, v in id_res.items() if isinstance(v, (int, float))})

    # 9. Compression
    c_res = CompressionSignal().extract(path, crops)
    results.update({k: v for k, v in c_res.items() if isinstance(v, (int, float))})

    # 10. Provenance
    p_res = ProvenanceSignal().extract(path)
    results.update(
        {
            "tool_is_ai_generator": 1.0 if p_res.get("tool_is_ai_generator") else 0.0,
            "has_timestamp_anomaly": 1.0 if p_res.get("has_timestamp_anomaly") else 0.0,
        }
    )

    return results


def _bootstrap_auroc_ci(
    y_true: np.ndarray,
    x_scores: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval for direction-invariant AUROC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc = roc_auc_score(y_b, x_scores[idx])
            auc = max(auc, 1.0 - auc)
            aucs.append(auc)
        except Exception:
            continue

    if len(aucs) < 50:
        return 0.5000, 0.5000
    ci_lower = float(np.percentile(aucs, 2.5))
    ci_upper = float(np.percentile(aucs, 97.5))
    return round(ci_lower, 4), round(ci_upper, 4)


def _benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Compute False Discovery Rate (FDR) adjusted q-values using Benjamini-Hochberg procedure."""
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with their original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    q_values = [1.0] * m

    min_q = 1.0
    # Walk backwards from highest p-value
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p_val = sorted_pairs[rank_idx]
        rank = rank_idx + 1  # 1-indexed rank
        q_val = min(1.0, (p_val * m) / rank)
        min_q = min(min_q, q_val)
        q_values[orig_idx] = round(float(min_q), 6)

    return q_values


def batch_extract_signals(
    failure_db_path: str | Path = "data/failure_cases.jsonl",
    output_signal_path: str | Path = "data/signal_extractions.jsonl",
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> int:
    """Run batch forensic extraction across registered failure cases with resume capability.

    Args:
        failure_db_path: Path to failure_cases.jsonl.
        output_signal_path: Destination for signal_extractions.jsonl.
        split: Optional split filter (e.g. 'discovery').
        limit: Optional maximum number of videos to extract.

    Returns:
        Number of newly extracted cases.
    """
    f_path = Path(failure_db_path)
    if not f_path.is_file():
        raise FileNotFoundError(f"Failure database not found at {f_path}")

    out_path = Path(output_signal_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if out_path.is_file():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line).get("video_id"))
                    except Exception:
                        pass

    cases = []
    with open(f_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if split:
        cases = [c for c in cases if c.get("dataset_split", "").lower() == split.lower()]

    if limit:
        cases = cases[:limit]

    print(f"[+] Found {len(cases)} cases to process ({len(existing_ids)} already extracted).")
    extracted_count = 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, case in enumerate(cases):
            vid_id = case.get("video_id")
            if vid_id in existing_ids:
                continue

            vid_path = case.get("video_path")
            if not vid_path or not Path(vid_path).is_file():
                continue

            try:
                signals = extract_signals_for_video(vid_path)
                record = dict(case)
                record.update(signals)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                extracted_count += 1
                if extracted_count % 10 == 0 or i == len(cases) - 1:
                    print(f"    [{i+1}/{len(cases)}] Extracted: {vid_id} ({case.get('dataset_split')})")
            except Exception as e:
                print(f"    [!] Error on {vid_id}: {e}")

    print(f"[+] Batch extraction finished. Newly extracted: {extracted_count}")
    return extracted_count


def analyze_discriminability(
    signal_db_path: str | Path = "data/signal_extractions.jsonl",
    output_path: str | Path = "data/signal_analysis.json",
    split: Optional[str] = "discovery",
    fdr_alpha: float = 0.05,
    n_bootstraps: int = 500,
) -> Dict[str, Any]:
    """Rank all extracted forensic features by their empirical ability to separate missed fakes from clean reals.

    Uses Benjamini-Hochberg False Discovery Rate (FDR) correction and bootstrap confidence intervals.
    AUROC is treated strictly as a diagnostic statistic, not a blind promotion or architecture filter.

    Args:
        signal_db_path: JSONL file containing extracted signals and labels.
        output_path: Output JSON file path.
        split: Dataset split to analyze (default 'discovery'; if None, analyzes all).
        fdr_alpha: FDR significance threshold (default: 0.05).
        n_bootstraps: Number of bootstrap iterations for AUROC 95% CI.

    Returns:
        Dictionary of ranked feature statistics.
    """
    records = []
    sig_p = Path(signal_db_path)
    if not sig_p.is_file():
        print(f"[-] Signal database not found at {sig_p}.")
        return {}

    with open(sig_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("[-] No records in signal database.")
        return {}

    df = pd.DataFrame(records)
    if "true_label" not in df.columns:
        raise ValueError("signal_db must contain 'true_label' column (FAKE/REAL).")

    # Enforce dataset split isolation
    if split and "dataset_split" in df.columns:
        df_split = df[df["dataset_split"].str.lower() == split.lower()]
        if len(df_split) > 0:
            df = df_split
            print(f"[+] Filtering strictly to dataset_split='{split}' ({len(df)} records)")
        else:
            print(f"[-] No records found for split='{split}', using full corpus ({len(df)} records)")

    y = df["true_label"].map({"FAKE": 1, "REAL": 0, 1: 1, 0: 0}).values
    ignore_cols = {
        "video_id",
        "video_path",
        "dataset_split",
        "true_label",
        "m13_5_verdict",
        "human_verdict",
        "human_confidence",
        "human_reason",
        "evidence_region",
        "evidence_time_range",
        "manipulation_family",
        "source_tool",
        "source_url",
        "notes",
        "collected_at",
        "is_learned_feature",
        "has_c2pa",
        "c2pa_status",
        "creation_tool",
        "tool_is_ai_generator",
        "has_timestamp_anomaly",
    }

    feature_cols = [c for c in df.columns if c not in ignore_cols]
    raw_results = []
    feature_names = []
    raw_p_values = []

    for col in feature_cols:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        valid_mask = ~series.isna()
        if valid_mask.sum() < 4:
            continue

        x = series[valid_mask].values
        y_valid = y[valid_mask]

        fake_vals = x[y_valid == 1]
        real_vals = x[y_valid == 0]

        if len(fake_vals) < 2 or len(real_vals) < 2:
            continue

        # Direction-invariant AUROC
        try:
            auc = roc_auc_score(y_valid, x)
            auc = max(auc, 1.0 - auc)
        except Exception:
            auc = 0.5000

        # Mann-Whitney U test
        try:
            _, pval = stats.mannwhitneyu(fake_vals, real_vals, alternative="two-sided")
        except Exception:
            pval = 1.0

        # Cohen's d effect size
        pooled_std = np.sqrt((fake_vals.std() ** 2 + real_vals.std() ** 2) / 2.0)
        cohen_d = abs(fake_vals.mean() - real_vals.mean()) / (pooled_std + 1e-9)

        # Bootstrap 95% Confidence Interval for AUROC
        ci_lower, ci_upper = _bootstrap_auroc_ci(y_valid, x, n_bootstraps=n_bootstraps)

        feature_names.append(col)
        raw_p_values.append(float(pval))
        raw_results.append(
            {
                "col": col,
                "auroc": round(float(auc), 4),
                "auroc_ci_95": [ci_lower, ci_upper],
                "cohen_d": round(float(cohen_d), 4),
                "p_value_raw": round(float(pval), 6),
                "n_fake": len(fake_vals),
                "n_real": len(real_vals),
                "fake_mean": round(float(fake_vals.mean()), 4),
                "real_mean": round(float(real_vals.mean()), 4),
            }
        )

    # Benjamini-Hochberg Multiple-Testing Correction
    q_values = _benjamini_hochberg(raw_p_values)

    results = {}
    for res, q_val in zip(raw_results, q_values):
        col = res["col"]
        res["q_value_fdr"] = q_val
        res["fdr_significant"] = bool(q_val < fdr_alpha)
        # Diagnostic categorisation
        if q_val < fdr_alpha and res["auroc"] >= 0.70:
            res["diagnostic_category"] = "candidate_informative"
        elif q_val < fdr_alpha:
            res["diagnostic_category"] = "statistically_significant_weak_effect"
        elif res["auroc"] >= 0.65:
            res["diagnostic_category"] = "exploratory_underpowered"
        else:
            res["diagnostic_category"] = "uninformative"

        results[col] = res

    # Rank by AUROC descending
    ranked = dict(sorted(results.items(), key=lambda item: item[1]["auroc"], reverse=True))

    print("\n" + "=" * 95)
    print(f"M16 FORENSIC SIGNAL DISCRIMINABILITY (Step 4 — Split: {split or 'ALL'})")
    print(f"Multiple Testing Correction: Benjamini-Hochberg FDR (alpha={fdr_alpha})")
    print("=" * 95)
    print(
        f"{'Feature':<35} {'AUROC':>7} {'95% CI':>16} {'Cohen_d':>9} {'p-raw':>10} {'q-FDR':>10} {'Category':>16}"
    )
    print("-" * 95)
    for feat, st in list(ranked.items())[:25]:
        ci_str = f"[{st['auroc_ci_95'][0]:.2f}, {st['auroc_ci_95'][1]:.2f}]"
        print(
            f"{feat:<35} {st['auroc']:>7.4f} {ci_str:>16} {st['cohen_d']:>9.4f} {st['p_value_raw']:>10.4f} {st['q_value_fdr']:>10.4f} {st['diagnostic_category']:>16}"
        )
    print("=" * 95)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    print(f"[+] Saved discriminability analysis to {out_p}")
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze forensic signal discriminability with Benjamini-Hochberg FDR.")
    parser.add_argument("--db-path", type=str, default="data/signal_extractions.jsonl", help="Signal database path")
    parser.add_argument("--output-path", type=str, default="data/signal_analysis.json", help="Analysis output path")
    parser.add_argument("--extract-from", type=str, default=None, help="Extract signals from failure_cases.jsonl before analysis")
    parser.add_argument("--split", type=str, default="discovery", help="Dataset split (discovery/development/holdout)")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR significance alpha")
    parser.add_argument("--bootstraps", type=int, default=500, help="AUROC bootstrap iterations")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to extract")
    args = parser.parse_args()

    if args.extract_from:
        batch_extract_signals(
            failure_db_path=args.extract_from,
            output_signal_path=args.db_path,
            split=args.split,
            limit=args.limit,
        )

    analyze_discriminability(
        signal_db_path=args.db_path,
        output_path=args.output_path,
        split=args.split,
        fdr_alpha=args.alpha,
        n_bootstraps=args.bootstraps,
    )


if __name__ == "__main__":
    main()

