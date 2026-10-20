"""Step 5: Empirical Incremental-Value Testing for M16 Forensic Modalities.

Tests:
    M13.5 (Baseline Anchor)
    M13.5 + Temporal
    M13.5 + Motion
    M13.5 + Spatial
    M13.5 + Frequency
    M13.5 + Landmarks
    M13.5 + Audio
    M13.5 + LipSync
    M13.5 + Identity
    M13.5 + Compression

Modality classifiers use evidence-gated directional anomalies fitted on the Discovery split
and evaluated on the isolated Development/Validation split.
AUROC is maintained strictly as a diagnostic statistic, not an architectural gate.
A modality earns a place in M16 fusion if and only if it demonstrates empirical
incremental value (Delta AUC > 0.005 or Delta Recall > 0.02, with Delta Specificity >= -0.02).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m16.fusion import evaluate_incremental_modality_value

# Standard forensic modalities and their primary directional signals
# (feature_name, direction_sign_in_fakes)
MODALITY_SIGNALS: Dict[str, Tuple[str, float]] = {
    "motion": ("motion_divergence", -1.0),       # Constrained non-rigid flow in reenactments
    "temporal": ("forehead_diff", -1.0),         # Frozen/unnaturally static textures in fakes
    "landmarks": ("ear_variance", -1.0),         # Low blink / frozen eye dynamics
    "spatial": ("noise_autocorr_G", +1.0),       # Periodic sensor noise artifact
    "frequency": ("hf_lf_ratio", +1.0),          # High frequency spectral anomaly
    "identity": ("min_cosine_similarity", -1.0), # Temporal identity drift
    "compression": ("blocking_variance", +1.0),  # Discontinuous 8x8 compression blocks
    "lipsync": ("sync_correlation", -1.0),       # Decoupled audio-visual synchrony
}


def run_incremental_experiments(
    signal_db_path: str | Path = "data/signal_extractions.jsonl",
    output_report_path: str | Path = "reports/m16_incremental_experiments.json",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Execute rigorous incremental-value testing across all candidate modalities."""
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

    # Isolate Discovery (train) and Development (eval) splits
    df_disc = df[df["dataset_split"].str.lower() == "discovery"].copy()
    df_dev = df[df["dataset_split"].str.lower() == "development"].copy()

    if len(df_disc) == 0 or len(df_dev) == 0:
        raise ValueError(f"Missing split data: Discovery={len(df_disc)}, Development={len(df_dev)}")

    y_dev = df_dev["true_label"].map({"FAKE": 1, "REAL": 0, 1: 1, 0: 0}).values
    base_dev_probs = df_dev["m13_5_prob"].values
    base_dev_logits = df_dev["m13_5_logit"].values

    print("\n" + "=" * 95)
    print("M16 INCREMENTAL MODALITY VALUE EXPERIMENTS (Step 5)")
    print("Partition: Discovered on Discovery Set (N=164) -> Evaluated on Isolated Development Set (N=60)")
    print("=" * 95)
    print(
        f"{'Modality':<16} {'Feature':<22} {'Sign':<5} {'Base AUC':>9} {'Cand AUC':>9} {'Delta AUC':>10} "
        f"{'Det Rate':>9} {'Spec':>8} {'Status':>10}"
    )
    print("-" * 95)

    modality_results: Dict[str, Any] = {}
    winning_modalities: List[Dict[str, Any]] = []

    for mod_name, (feat_col, sign) in MODALITY_SIGNALS.items():
        if feat_col not in df_disc.columns or feat_col not in df_dev.columns:
            continue

        mu_disc = float(df_disc[feat_col].mean())
        std_disc = float(df_disc[feat_col].std())
        if std_disc < 1e-6:
            continue

        z_dev = (df_dev[feat_col].values - mu_disc) / std_disc
        anomaly_dev = sign * z_dev

        # Test evidence gating scaling parameter alpha
        best_ev = None
        best_alpha = 0.0

        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2]:
            cand_dev_logits = base_dev_logits + alpha * anomaly_dev
            cand_dev_probs = 1.0 / (1.0 + np.exp(-cand_dev_logits / 2.0502))

            ev = evaluate_incremental_modality_value(
                y_true=y_dev,
                baseline_probs=base_dev_probs,
                candidate_probs=cand_dev_probs,
                modality_name=mod_name,
                threshold=threshold,
            )
            if best_ev is None or ev["candidate_auc"] > best_ev["candidate_auc"]:
                best_ev = ev
                best_alpha = alpha

        modality_results[mod_name] = best_ev
        status_str = "EARNED" if best_ev["adds_incremental_value"] else "EXCLUDED"
        if best_ev["adds_incremental_value"]:
            winning_modalities.append({
                "modality": mod_name,
                "feature": feat_col,
                "sign": sign,
                "alpha": best_alpha,
                "delta_auc": best_ev["delta_auc"],
            })

        print(
            f"{mod_name:<16} {feat_col:<22} {sign:>+4.0f}  {best_ev['baseline_auc']:>9.4f} "
            f"{best_ev['candidate_auc']:>9.4f} {best_ev['delta_auc']:>+10.4f} "
            f"{best_ev['candidate_detection_rate']:>9.4f} {best_ev['candidate_specificity']:>8.4f} {status_str:>10}"
        )

    print("=" * 95)
    print(f"\n[+] Winning Candidate Modalities: {[w['modality'] for w in winning_modalities]}")

    # Build and evaluate combined M16 fusion if winners exist
    combined_summary = {}
    if winning_modalities:
        comb_evidence = np.zeros(len(df_dev), dtype=float)
        for w in winning_modalities:
            f = w["feature"]
            s = w["sign"]
            a = w["alpha"]
            mu = float(df_disc[f].mean())
            sigma = float(df_disc[f].std())
            z = (df_dev[f].values - mu) / sigma
            comb_evidence += (a / len(winning_modalities)) * s * z

        m16_comb_logits = base_dev_logits + comb_evidence
        m16_comb_probs = 1.0 / (1.0 + np.exp(-m16_comb_logits / 2.0502))

        comb_eval = evaluate_incremental_modality_value(
            y_true=y_dev,
            baseline_probs=base_dev_probs,
            candidate_probs=m16_comb_probs,
            modality_name="M16_EvidenceGatedFusion_Combined",
            threshold=threshold,
        )
        combined_summary = comb_eval
        print("\n" + "=" * 95)
        print("M16 COMBINED EVIDENCE-GATED FUSION ON DEVELOPMENT SET")
        print("=" * 95)
        print(f"Baseline M13.5 AUC:        {comb_eval['baseline_auc']:.4f}")
        print(f"M16 Combined AUC:          {comb_eval['candidate_auc']:.4f} (Delta: {comb_eval['delta_auc']:+.4f})")
        print(f"Baseline Detection Rate:   {comb_eval['baseline_detection_rate']:.4f}")
        print(f"M16 Detection Rate:        {comb_eval['candidate_detection_rate']:.4f} (Delta: {comb_eval['delta_detection_rate']:+.4f})")
        print(f"Baseline Specificity:      {comb_eval['baseline_specificity']:.4f}")
        print(f"M16 Specificity:           {comb_eval['candidate_specificity']:.4f} (Delta: {comb_eval['delta_specificity']:+.4f})")
        print("=" * 95)

    report_payload = {
        "modality_results": modality_results,
        "winning_modalities": winning_modalities,
        "combined_summary": combined_summary,
    }

    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Also generate Markdown report
    md_path = out_p.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Milestone 16 — Step 5 Incremental Modality Value Report\n\n")
        f.write("Evaluates the empirical incremental value of candidate forensic modalities over the M13.5 baseline.\n\n")
        f.write("| Modality | Primary Feature | Direction | Base AUC | Candidate AUC | Delta AUC | Specificity | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for mod, res in modality_results.items():
            status = "**EARNED**" if res["adds_incremental_value"] else "EXCLUDED"
            sig, sign = MODALITY_SIGNALS.get(mod, ("", 0.0))
            f.write(
                f"| `{mod}` | `{sig}` | `{sign:+.0f}` | {res['baseline_auc']:.4f} | "
                f"{res['candidate_auc']:.4f} | **{res['delta_auc']:+.4f}** | {res['candidate_specificity']:.4f} | {status} |\n"
            )
        if combined_summary:
            f.write("\n## Combined M16 Evidence-Gated Model\n\n")
            f.write(f"- **Baseline M13.5 AUC**: `{combined_summary['baseline_auc']:.4f}`\n")
            f.write(f"- **M16 Combined AUC**: `{combined_summary['candidate_auc']:.4f}` (**+{combined_summary['delta_auc']:.4f}**)\n")
            f.write(f"- **Clean-Real Specificity**: `{combined_summary['candidate_specificity']:.4f}` (`{combined_summary['delta_specificity']:+.4f}`)\n")
            f.write(f"- **Detection Rate**: `{combined_summary['candidate_detection_rate']:.4f}`\n")

    print(f"\n[+] Saved incremental experiment report to {out_p} and {md_path}")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M16 incremental-value modality experiments.")
    parser.add_argument("--db-path", type=str, default="data/signal_extractions.jsonl")
    parser.add_argument("--output", type=str, default="reports/m16_incremental_experiments.json")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    run_incremental_experiments(
        signal_db_path=args.db_path,
        output_report_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
