"""Step 6: Final Locked Holdout & Unseen Generator Benchmark.

Evaluates M16 Evidence-Gated Fusion against the M13.5 baseline on:
1. The Final Locked Holdout split (66 videos: Face2Face, NeuralTextures, FaceSwap, Deepfakes, Reals).
   NOTE: This split was NEVER touched or looked at during signal discovery or development.
2. The External OOD split (114 videos: Celeb-DF-v2).

Statistically supported promotion gates enforced:
- Paired Bootstrap Delta AUC (95% CI)
- Paired McNemar Decision Test (net recoveries c >= b)
- Specificity Non-Inferiority Margin (Spec_M16 >= Spec_M13.5 - 0.02)
- Expected Calibration Error (ECE)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.evaluate_ood import evaluate_ood


def run_holdout_benchmark(
    signal_db_path: str | Path = "data/signal_extractions.jsonl",
    output_path: str | Path = "reports/m16_step6_holdout_benchmark.json",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    sig_p = Path(signal_db_path)
    if not sig_p.is_file():
        raise FileNotFoundError(f"Signal extractions DB not found: {sig_p}")

    records = []
    with open(sig_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"[+] Loaded {len(df)} total signal extractions.")

    # Discovery statistics for normalization (strictly computed on Discovery split)
    df_disc = df[df["dataset_split"].str.lower() == "discovery"]
    mu_mot = float(df_disc["motion_divergence"].mean())
    std_mot = float(df_disc["motion_divergence"].std())
    mu_fore = float(df_disc["forehead_diff"].mean())
    std_fore = float(df_disc["forehead_diff"].std())

    # We evaluate on Holdout AND External OOD splits
    eval_splits = ["holdout", "external_ood"]
    df_eval = df[df["dataset_split"].str.lower().isin(eval_splits)].copy()

    print(f"[+] Evaluation Split counts: {df_eval['dataset_split'].value_counts().to_dict()}")

    m13_results = []
    m16_results = []

    for _, row in df_eval.iterrows():
        v_id = str(row["video_id"])
        true_lbl = str(row["true_label"]).upper()
        ds_split = str(row["dataset_split"]).lower()
        tool_name = str(row.get("source_tool", "unknown"))

        # Group label for OOD breakdown (must contain both reals and fakes for valid binary AUROC)
        if ds_split == "external_ood":
            gen_group = "CelebDF_v2_OOD"
        else:
            gen_group = "FF++_Holdout_Overall"

        base_logit = float(row["m13_5_logit"])
        base_prob = float(row["m13_5_prob"])

        # M16 Evidence-Gated prediction:
        # Winning modalities from Step 5: motion_divergence (sign=-1, alpha=0.5), forehead_diff (sign=-1, alpha=0.4)
        mot_val = float(row.get("motion_divergence", mu_mot))
        fore_val = float(row.get("forehead_diff", mu_fore))

        z_mot = (mot_val - mu_mot) / (std_mot + 1e-6)
        z_fore = (fore_val - mu_fore) / (std_fore + 1e-6)

        # Evidence anomaly (lower divergence & lower forehead diff -> more fake)
        comb_evidence = (-0.5 * z_mot) + (-0.4 * z_fore)
        m16_logit = base_logit + comb_evidence
        m16_prob = float(1.0 / (1.0 + np.exp(-m16_logit / 2.0502)))

        m13_results.append({
            "video_id": v_id,
            "generator": gen_group,
            "true_label": true_lbl,
            "prob": round(base_prob, 4),
            "logit": round(base_logit, 4),
        })

        m16_results.append({
            "video_id": v_id,
            "generator": gen_group,
            "true_label": true_lbl,
            "prob": round(m16_prob, 4),
            "logit": round(m16_logit, 4),
        })

    # Execute Step 6 statistical evaluation
    eval_report, all_pass = evaluate_ood(
        m13_5_results=m13_results,
        m16_results=m16_results,
        output_path=output_path,
        threshold=threshold,
        n_bootstraps=500,
    )

    # Generate Markdown Summary
    md_path = Path(output_path).with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Milestone 16 — Step 6 Holdout & OOD Benchmark Report\n\n")
        f.write(f"**Final Promotion Verdict**: `{eval_report.get('promotion_verdict', 'N/A')}`\n\n")
        f.write("| Dataset / Family | Model | AUROC (95% CI) | Detection Rate | Specificity | ECE | McNemar (b / c / p) | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")

        for gen, r in eval_report.items():
            if not isinstance(r, dict) or "m13_5" not in r:
                continue
            m13 = r["m13_5"]
            m16 = r["m16"]
            ci = r["delta_auc_95_ci"]
            ci_str = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
            mcn_str = f"b={r['mcnemar_regressions_b']}, c={r['mcnemar_recoveries_c']}, p={r['mcnemar_p_value']:.3f}"
            status = r["status"]
            f.write(
                f"| `{gen}` | M13.5 | {m13['auroc']:.4f} | {m13['detection_rate']:.1%} | {m13['specificity']:.1%} | {m13['ece']:.3f} | — | — |\n"
                f"| | **M16** | **{m16['auroc']:.4f}** ({r['delta_auc']:+.4f} {ci_str}) | **{m16['detection_rate']:.1%}** | **{m16['specificity']:.1%}** | **{m16['ece']:.3f}** | {mcn_str} | {status} |\n"
            )

    print(f"[+] Saved Step 6 Holdout Benchmark reports to {output_path} and {md_path}")
    return eval_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 6 Holdout and OOD benchmark.")
    parser.add_argument("--db-path", type=str, default="data/signal_extractions.jsonl")
    parser.add_argument("--output", type=str, default="reports/m16_step6_holdout_benchmark.json")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    run_holdout_benchmark(
        signal_db_path=args.db_path,
        output_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
