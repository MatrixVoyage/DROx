"""Evaluation module for Deepfake Video Detection."""

from src.evaluation.attribution import (
    FrameAttentionStats,
    SequenceAttentionSummary,
    compute_sequence_attention_stats,
    create_attention_overlay,
    create_attribution_panel,
    extract_attention_weights,
    interpolate_attention_heatmap,
)
from src.evaluation.calibration import (
    CalibrationMetrics,
    TemperatureScaler,
    compute_calibration_metrics,
    find_optimal_thresholds,
)
from src.evaluation.cross_dataset import (
    CrossDatasetEvaluator,
    DatasetBenchmarkResult,
)
from src.evaluation.error_analysis import (
    CorrelationFinding,
    ErrorSampleMetadata,
    analyze_detection_correlations,
    categorize_errors,
    generate_error_analysis_summary,
)
from src.evaluation.metrics import EvaluationMetrics, compute_eer, compute_metrics
from src.evaluation.robustness import (
    PerturbationConfig,
    apply_additive_noise,
    apply_contrast_reduction,
    apply_gaussian_blur,
    apply_jpeg_compression_simulation,
    apply_spatial_downsampling,
)
from src.evaluation.baseline_reproduction import (
    validate_cache_provenance,
    reproduce_baseline_metrics,
    verify_physical_subset,
)
from src.evaluation.score_analysis import (
    compute_group_statistics,
    compute_cliffs_delta,
    compute_cohens_d,
    compute_distribution_distance,
    analyze_all_score_groups,
    plot_score_distributions,
    plot_roc_curves,
    plot_pr_curves,
)
from src.evaluation.preprocessing_ablation import run_preprocessing_ablation_suite
from src.evaluation.temporal_ablation import (
    run_temporal_ablation_suite,
    plot_temporal_comparison,
)
from src.evaluation.attention_ablation import (
    run_gate5a_observational_attribution,
    run_gate5b_interventional_ablation,
    plot_attention_comparison,
)
from src.evaluation.error_diagnosis import (
    build_error_matrix_for_dataset,
    analyze_multithreshold_errors,
    analyze_family_error_breakdown,
    generate_error_gallery,
)
from src.evaluation.statistics import (
    bootstrap_video_level_metrics,
    compute_paired_mcnemar_test,
)

__all__ = [
    "EvaluationMetrics",
    "compute_eer",
    "compute_metrics",
    "CrossDatasetEvaluator",
    "DatasetBenchmarkResult",
    "CalibrationMetrics",
    "TemperatureScaler",
    "compute_calibration_metrics",
    "find_optimal_thresholds",
    "PerturbationConfig",
    "apply_spatial_downsampling",
    "apply_gaussian_blur",
    "apply_additive_noise",
    "apply_contrast_reduction",
    "apply_jpeg_compression_simulation",
    "FrameAttentionStats",
    "SequenceAttentionSummary",
    "extract_attention_weights",
    "compute_sequence_attention_stats",
    "interpolate_attention_heatmap",
    "create_attention_overlay",
    "create_attribution_panel",
    "ErrorSampleMetadata",
    "CorrelationFinding",
    "categorize_errors",
    "analyze_detection_correlations",
    "generate_error_analysis_summary",
    "validate_cache_provenance",
    "reproduce_baseline_metrics",
    "verify_physical_subset",
    "compute_group_statistics",
    "compute_cliffs_delta",
    "compute_cohens_d",
    "compute_distribution_distance",
    "analyze_all_score_groups",
    "plot_score_distributions",
    "plot_roc_curves",
    "plot_pr_curves",
    "run_preprocessing_ablation_suite",
    "run_temporal_ablation_suite",
    "plot_temporal_comparison",
    "run_gate5a_observational_attribution",
    "run_gate5b_interventional_ablation",
    "plot_attention_comparison",
    "build_error_matrix_for_dataset",
    "analyze_multithreshold_errors",
    "analyze_family_error_breakdown",
    "generate_error_gallery",
    "bootstrap_video_level_metrics",
    "compute_paired_mcnemar_test",
]
