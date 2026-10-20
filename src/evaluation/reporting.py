"""Comparative benchmark reporting module.

Generates research-grade Markdown and JSON reports comparing project-generated
experimental results against reference baselines from Lal et al. (2025).

Research Integrity Rule (Rule 10):
Reference paper values are strictly labeled as [Literature Reference].
Project results are populated exclusively from executed experiment artifacts.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


# Reference baseline values reported in Lal et al. (2025) - Scientific Reports
LAL_2025_REFERENCES = {
    "ffpp_c23_models": [
        {"model": "Lal et al. (2025) Proposed [Ref]", "ac": "0.9288 ± 0.012", "pr": "0.940 ± 0.010", "rc": "0.943 ± 0.009", "f1": "0.944 ± 0.008", "auc": "0.962 ± 0.008", "eer": "4.2 ± 0.5%"},
        {"model": "ResNeXt-50 alone [Ref]", "ac": "0.8900 ± 0.015", "pr": "0.880 ± 0.020", "rc": "0.870 ± 0.014", "f1": "0.875 ± 0.012", "auc": "0.912 ± 0.010", "eer": "8.5 ± 1.0%"},
        {"model": "LSTM temporal only [Ref]", "ac": "0.8920 ± 0.014", "pr": "0.890 ± 0.015", "rc": "0.882 ± 0.013", "f1": "0.885 ± 0.011", "auc": "0.920 ± 0.009", "eer": "7.8 ± 0.9%"},
        {"model": "EfficientNet-B4 [Ref]", "ac": "0.9240 ± 0.013", "pr": "0.940 ± 0.010", "rc": "0.908 ± 0.011", "f1": "0.924 ± 0.009", "auc": "0.950 ± 0.007", "eer": "5.1 ± 0.6%"},
        {"model": "Vision Transformer (ViT) [Ref]", "ac": "0.9150 ± 0.014", "pr": "0.920 ± 0.012", "rc": "0.910 ± 0.010", "f1": "0.915 ± 0.009", "auc": "0.945 ± 0.008", "eer": "5.5 ± 0.7%"},
    ],
    "ablations": [
        {"variant": "Full Model [Ref]", "ac": "0.9288 ± 0.012", "auc": "0.962 ± 0.008", "eer": "4.2 ± 0.5%"},
        {"variant": "w/o Dual Attention [Ref]", "ac": "0.8900 ± 0.015", "auc": "0.917 ± 0.010", "eer": "8.7 ± 1.1%"},
        {"variant": "w/o BiLSTM (CNN only) [Ref]", "ac": "0.9050 ± 0.014", "auc": "0.935 ± 0.009", "eer": "6.5 ± 0.8%"},
        {"variant": "w/o Self-Attention Gate [Ref]", "ac": "0.9150 ± 0.013", "auc": "0.945 ± 0.008", "eer": "5.3 ± 0.7%"},
    ],
    "cross_dataset": [
        {"target": "Celeb-DF v2 [Ref]", "reference_auc": "0.950 ± 0.009", "baseline_vit_auc": "0.760 ± 0.012"},
        {"target": "DFDC [Ref]", "reference_auc": "0.920 ± 0.010", "baseline_vit_auc": "0.680 ± 0.015"},
    ]
}


class BenchmarkReporter:
    """Consolidates experiment and cross-dataset evaluation outputs into structured comparative reports."""

    def __init__(self, title: str = "Deepfake Detection Benchmark Report") -> None:
        self.title = title
        self.indomain_runs: List[Dict[str, Any]] = []
        self.ablation_runs: List[Dict[str, Any]] = []
        self.cross_dataset_results: Optional[Dict[str, Any]] = None

    def add_indomain_experiment(self, experiment_summary: Dict[str, Any], display_name: Optional[str] = None) -> None:
        """Add an in-domain experiment run summary."""
        entry = {
            "name": display_name or experiment_summary.get("experiment_name", "Unknown"),
            "summary": experiment_summary,
        }
        self.indomain_runs.append(entry)

    def add_ablation_experiment(self, experiment_summary: Dict[str, Any], variant_label: str) -> None:
        """Add an ablation experiment run summary."""
        self.ablation_runs.append({
            "variant": variant_label,
            "summary": experiment_summary,
        })

    def set_cross_dataset_results(self, cross_dataset_payload: Dict[str, Any]) -> None:
        """Set cross-dataset evaluation results payload."""
        self.cross_dataset_results = cross_dataset_payload

    def _fmt(self, val: Optional[float], is_pct: bool = False) -> str:
        """Format numerical float values safely."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if is_pct:
            return f"{val * 100:.2f}%"
        return f"{val:.4f}"

    def generate_markdown(self) -> str:
        """Render complete Markdown benchmark report."""
        lines = [
            f"# {self.title}",
            "",
            "> [!NOTE]",
            "> All entries tagged with **[Ref]** correspond to external published values reported in *Lal et al. (2025)* (Scientific Reports).",
            "> All project-generated values correspond strictly to executed code experiments with zero fabrication.",
            "",
            "## 1. In-Domain Benchmark: FaceForensics++ (FF++) C23",
            "",
            "| Model / Source | Accuracy | Precision | Recall | F1-Score | ROC-AUC | EER |",
            "|---|---|---|---|---|---|---|",
        ]

        # Reference rows
        for r in LAL_2025_REFERENCES["ffpp_c23_models"]:
            lines.append(f"| {r['model']} | {r['ac']} | {r['pr']} | {r['rc']} | {r['f1']} | {r['auc']} | {r['eer']} |")

        # Project executed runs
        for run in self.indomain_runs:
            name = run["name"]
            metrics = run["summary"].get("test_metrics") or {}
            ac = self._fmt(metrics.get("accuracy"))
            pr = self._fmt(metrics.get("precision"))
            rc = self._fmt(metrics.get("recall"))
            f1 = self._fmt(metrics.get("f1"))
            auc = self._fmt(metrics.get("roc_auc"))
            eer = self._fmt(metrics.get("eer"), is_pct=True)
            lines.append(f"| **{name} (Project Run)** | **{ac}** | **{pr}** | **{rc}** | **{f1}** | **{auc}** | **{eer}** |")

        lines.extend([
            "",
            "## 2. Systematic Ablation Study (FF++ C23)",
            "",
            "| Architecture Variant | Accuracy | ROC-AUC | EER |",
            "|---|---|---|---|",
        ])

        # Reference ablation rows
        for a in LAL_2025_REFERENCES["ablations"]:
            lines.append(f"| {a['variant']} | {a['ac']} | {a['auc']} | {a['eer']} |")

        # Project executed ablations
        for ab in self.ablation_runs:
            lbl = ab["variant"]
            metrics = ab["summary"].get("test_metrics") or {}
            ac = self._fmt(metrics.get("accuracy"))
            auc = self._fmt(metrics.get("roc_auc"))
            eer = self._fmt(metrics.get("eer"), is_pct=True)
            lines.append(f"| **{lbl} (Project Run)** | **{ac}** | **{auc}** | **{eer}** |")

        lines.extend([
            "",
            "## 3. Cross-Dataset Generalization (Trained on FF++ C23)",
            "",
            "| Target Benchmark | Test Samples | Accuracy | ROC-AUC | EER | $\\Delta$ AUC vs In-Domain | Paper Ref AUC |",
            "|---|---|---|---|---|---|---|",
        ])

        if self.cross_dataset_results and "benchmarks" in self.cross_dataset_results:
            benches = self.cross_dataset_results["benchmarks"]
            for target_name, data in benches.items():
                m = data.get("metrics", {})
                num_s = data.get("num_samples", 0)
                ac = self._fmt(m.get("accuracy"))
                auc = self._fmt(m.get("roc_auc"))
                eer = self._fmt(m.get("eer"), is_pct=True)
                drop = self._fmt(data.get("auc_drop_vs_indomain"))

                # Match paper ref
                ref_auc = "N/A"
                if "celeb" in target_name.lower():
                    ref_auc = "0.950 ± 0.009"
                elif "dfdc" in target_name.lower():
                    ref_auc = "0.920 ± 0.010"

                lines.append(f"| **{target_name}** | {num_s} | {ac} | {auc} | {eer} | -{drop} | {ref_auc} |")
        else:
            lines.append("| *No cross-dataset evaluation results added yet* | - | - | - | - | - | - |")

        return "\n".join(lines)

    def save_reports(self, output_dir: Union[str, Path], filename_stem: str = "benchmark_report") -> Path:
        """Save both Markdown and JSON report files to destination directory.

        Args:
            output_dir: Target output directory.
            filename_stem: Base name for report files.

        Returns:
            Path to the saved Markdown report.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        md_content = self.generate_markdown()
        md_file = out_p / f"{filename_stem}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        json_file = out_p / f"{filename_stem}.json"
        json_payload = {
            "title": self.title,
            "indomain_runs": self.indomain_runs,
            "ablation_runs": self.ablation_runs,
            "cross_dataset_results": self.cross_dataset_results,
            "literature_references": LAL_2025_REFERENCES,
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)

        return md_file
