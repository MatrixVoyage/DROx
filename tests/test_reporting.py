"""Unit tests for BenchmarkReporter and reference labeling."""

import json
from pathlib import Path

from src.evaluation.reporting import BenchmarkReporter, LAL_2025_REFERENCES


def test_benchmark_reporter_markdown_and_json(tmp_path: Path):
    """Verify BenchmarkReporter formats tables correctly and preserves reference labels."""
    reporter = BenchmarkReporter(title="Test Benchmark Run")

    # Add mock in-domain run
    mock_indomain = {
        "experiment_name": "resnext50_soft_lstm",
        "test_metrics": {
            "accuracy": 0.931,
            "precision": 0.942,
            "recall": 0.945,
            "f1": 0.943,
            "roc_auc": 0.965,
            "eer": 0.039,
        }
    }
    reporter.add_indomain_experiment(mock_indomain, display_name="Baseline (Soft + LSTM)")

    # Add mock ablation run
    mock_ablation = {
        "experiment_name": "resnext50_none_lstm",
        "test_metrics": {
            "accuracy": 0.895,
            "roc_auc": 0.920,
            "eer": 0.081,
        }
    }
    reporter.add_ablation_experiment(mock_ablation, variant_label="w/o Attention")

    # Add mock cross-dataset results
    mock_cross = {
        "benchmarks": {
            "Celeb-DF v2": {
                "num_samples": 518,
                "metrics": {"accuracy": 0.912, "roc_auc": 0.948, "eer": 0.052},
                "auc_drop_vs_indomain": 0.017,
            }
        }
    }
    reporter.set_cross_dataset_results(mock_cross)

    md = reporter.generate_markdown()

    # Invariants: Must contain reference labels
    assert "[Ref]" in md
    assert "Lal et al. (2025)" in md
    assert "Baseline (Soft + LSTM) (Project Run)" in md
    assert "0.9310" in md
    assert "w/o Attention (Project Run)" in md
    assert "Celeb-DF v2" in md

    # Verify report persistence
    report_file = reporter.save_reports(tmp_path, filename_stem="test_report")
    assert report_file.exists()
    json_file = tmp_path / "test_report.json"
    assert json_file.exists()

    with open(json_file, "r") as f:
        data = json.load(f)
    assert data["title"] == "Test Benchmark Run"
    assert len(data["indomain_runs"]) == 1
    assert len(data["ablation_runs"]) == 1
