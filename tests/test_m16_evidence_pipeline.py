"""Unit and integration tests for Milestone 16 (M16) evidence-driven pipeline.

Validates:
- Step 1: Failure case collection infrastructure, human observations, and dataset splits
- Step 2: Full pipeline instrumentation with strict M13.5_CONTROL vs M16_EXPERIMENTAL isolation
- Step 3: All 10 forensic signal extractors (including separate provenance channel)
- Step 4: Discriminability ranking with Benjamini-Hochberg FDR correction and bootstrap CIs
- Step 5: EvidenceGatedFusion initialization, forward shapes, and incremental-value evaluation
- Step 6: OOD unseen generator promotion gate with paired bootstrap and McNemar tests
"""

from __future__ import annotations

import cv2
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from tools.collect_failures import (
    DatasetSplit,
    FailureCase,
    HumanConfidence,
    HumanReason,
    HumanVerdict,
    ManipulationFamily,
    load_failure_cases,
    register_failure,
)
from src.pipeline_instrumented import FailureStage, FrameTrace, PipelineMode, PipelineTrace, InstrumentedPipeline
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
from tools.analyze_signals import analyze_discriminability
from src.m16.fusion import EvidenceGatedFusion, evaluate_incremental_modality_value
from tools.evaluate_ood import compute_metrics, evaluate_ood


# ─────────────────────────────────────────────
# STEP 1 TESTS: FAILURE REGISTRATION
# ─────────────────────────────────────────────
def test_step1_failure_registration(tmp_path: Path):
    dummy_video = tmp_path / "test_fake.mp4"
    dummy_video.write_bytes(b"dummy video byte content for sha256 hashing")

    db_file = tmp_path / "failures.jsonl"

    case = register_failure(
        video_path=dummy_video,
        m13_5_verdict="REAL",
        m13_5_logit=-2.5,
        human_verdict=HumanVerdict.DEFINITELY_FAKE,
        manipulation_family=ManipulationFamily.REENACTMENT,
        true_label="FAKE",
        dataset_split=DatasetSplit.DISCOVERY,
        human_confidence=HumanConfidence.HIGH,
        human_reason=HumanReason.MOUTH_GEOMETRY,
        evidence_region="mouth",
        evidence_time_range=[1.2, 3.5],
        source_tool="Face2Face",
        notes="Mouth boundary jitter observed",
        db_path=db_file,
    )

    assert isinstance(case, FailureCase)
    assert len(case.sha256) == 64
    assert case.video_id == case.sha256[:12]
    assert case.m13_5_logit == -2.5
    assert 0.0 < case.m13_5_prob < 0.50  # Scaled by T=2.0502
    assert case.manipulation_family == "reenactment"
    assert case.dataset_split == "discovery"
    assert case.human_reason == "mouth_geometry"
    assert case.evidence_region == "mouth"
    assert case.evidence_time_range == [1.2, 3.5]

    loaded = load_failure_cases(db_file)
    assert len(loaded) == 1
    assert loaded[0].video_id == case.video_id
    assert loaded[0].human_reason == "mouth_geometry"


# ─────────────────────────────────────────────
# STEP 2 TESTS: PIPELINE TRACES & CONTROL INVARIANT
# ─────────────────────────────────────────────
def test_step2_trace_dataclasses():
    ft = FrameTrace(
        frame_idx=0,
        timestamp_ms=0.0,
        face_detected=True,
        face_bbox=[10, 10, 100, 100],
        face_confidence=0.99,
        face_area_ratio=0.15,
        head_pose_deg=[2.0, -1.0, 0.0],
        blur_score=125.4,
        compression_score=0.85,
    )
    d_ft = ft.to_dict()
    assert d_ft["face_detected"] is True
    assert d_ft["blur_score"] == 125.4

    pt = PipelineTrace(
        video_id="sample_01",
        video_path="sample.mp4",
        pipeline_mode=PipelineMode.M13_5_CONTROL.value,
        total_frames=100,
        sampled_frames=16,
        faces_detected=16,
        detection_rate=1.0,
        frame_traces=[ft],
        raw_logit=2.5,
        calibrated_prob=0.7718,
        verdict_at_050="DEEPFAKE",
        verdict_at_019="DEEPFAKE",
        verdict_at_013="DEEPFAKE",
    )
    d_pt = pt.to_dict()
    assert d_pt["pipeline_mode"] == "M13_5_CONTROL"
    assert d_pt["detection_rate"] == 1.0
    assert d_pt["verdict_at_050"] == "DEEPFAKE"


def test_step2_m13_5_control_invariants():
    """Verify that M13_5_CONTROL enforces strictly 112x112 resolution."""
    # Mock predictor for unit test
    class MockPredictor:
        def __init__(self):
            self.model = torch.nn.Identity()
            self.device = torch.device("cpu")
            self.pipeline = None

    mock_pred = MockPredictor()
    pipe_control = InstrumentedPipeline(mock_pred, mode=PipelineMode.M13_5_CONTROL)
    assert pipe_control.target_resolution == 112
    assert pipe_control.mode == PipelineMode.M13_5_CONTROL

    pipe_exp = InstrumentedPipeline(mock_pred, mode=PipelineMode.M16_EXPERIMENTAL, target_resolution=224)
    assert pipe_exp.target_resolution == 224
    assert pipe_exp.mode == PipelineMode.M16_EXPERIMENTAL


# ─────────────────────────────────────────────
# STEP 3 TESTS: TEN FORENSIC EXTRACTORS
# ─────────────────────────────────────────────
def test_step3_spatial_signal():
    img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    res = SpatialSignal().extract(img)
    assert res["spatial_available"] is True
    for c in ["B", "G", "R"]:
        assert f"noise_std_{c}" in res
        assert f"noise_kurtosis_{c}" in res
        assert f"noise_autocorr_{c}" in res


def test_step3_temporal_signal():
    seq = [np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8) for _ in range(4)]
    res = TemporalSignal().extract(seq)
    assert res["temporal_available"] is True
    assert "mean_frame_diff" in res
    assert "diff_variance" in res
    assert "frozen_pixel_ratio" in res
    assert "eye_to_forehead_ratio" in res


def test_step3_motion_signal():
    seq = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(3)]
    for i, frame in enumerate(seq):
        cv2.circle(frame, (50 + i * 4, 50), 15, (255, 255, 255), -1)
    res = MotionSignal().extract(seq)
    assert res["motion_available"] is True
    assert "mean_motion_mag" in res
    assert "motion_smoothness" in res
    assert "motion_divergence" in res


def test_step3_frequency_signal():
    img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    res = FrequencySignal().extract(img)
    assert res["frequency_available"] is True
    assert "low_freq_ratio" in res
    assert "mid_freq_ratio" in res
    assert "high_freq_ratio" in res
    assert "nyquist_peak" in res
    assert "spectral_isotropy" in res


def test_step3_landmark_signal():
    seq = [np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8) for _ in range(4)]
    res = LandmarkSignal().extract(seq)
    assert res["landmark_available"] is True
    assert "landmark_jitter" in res
    assert "symmetry_score" in res
    assert "ear_mean" in res


def test_step3_audio_signal(tmp_path: Path):
    non_existent = tmp_path / "no_audio.mp4"
    res = AudioSignal().extract(non_existent)
    assert res["audio_available"] is False


def test_step3_lipsync_signal():
    seq = [np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8) for _ in range(5)]
    res = LipSyncSignal().extract("dummy.mp4", seq)
    assert res["lipsync_available"] is True
    assert "sync_correlation" in res


def test_step3_identity_signal():
    seq = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(3)]
    res = IdentitySignal().extract(seq)
    assert res["identity_available"] is True
    assert "mean_cosine_similarity" in res
    assert "identity_drift" in res


def test_step3_compression_signal(tmp_path: Path):
    seq = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(2)]
    res = CompressionSignal().extract(tmp_path / "dummy.mp4", seq)
    assert res["compression_available"] is True
    assert "blocking_mean" in res
    assert "dct_hist_peakiness" in res


def test_step3_provenance_signal(tmp_path: Path):
    vid = tmp_path / "test.mp4"
    vid.write_bytes(b"dummy")
    res = ProvenanceSignal().extract(vid)
    assert res["provenance_available"] is True
    assert res["is_learned_feature"] is False  # Separate non-learned evidence channel


# ─────────────────────────────────────────────
# STEP 4 TESTS: FDR CORRECTION & BOOTSTRAP CIs
# ─────────────────────────────────────────────
def test_step4_discriminability_analysis(tmp_path: Path):
    sig_file = tmp_path / "signals.jsonl"
    out_file = tmp_path / "ranked.json"

    # Create synthetic dataset with discovery split
    records = []
    for i in range(30):
        is_fake = i < 15
        records.append(
            {
                "video_id": f"vid_{i}",
                "dataset_split": "discovery",
                "true_label": "FAKE" if is_fake else "REAL",
                # Highly discriminative signal
                "signal_strong": float(np.random.normal(0.85 if is_fake else 0.15, 0.05)),
                # Weak noisy signal
                "signal_noisy": float(np.random.normal(0.50, 0.20)),
            }
        )

    with open(sig_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ranked = analyze_discriminability(
        signal_db_path=sig_file,
        output_path=out_file,
        split="discovery",
        fdr_alpha=0.05,
        n_bootstraps=100,
    )

    assert "signal_strong" in ranked
    st = ranked["signal_strong"]
    assert st["auroc"] >= 0.90
    assert "auroc_ci_95" in st
    assert len(st["auroc_ci_95"]) == 2
    assert "q_value_fdr" in st
    assert st["fdr_significant"] is True
    assert st["diagnostic_category"] == "candidate_informative"
    assert out_file.is_file()


# ─────────────────────────────────────────────
# STEP 5 TESTS: INCREMENTAL VALUE FUSION
# ─────────────────────────────────────────────
def test_step5_evidence_gated_fusion():
    modalities = ["temporal", "motion"]
    priors = {"temporal": 0.78, "motion": 0.65}

    fusion = EvidenceGatedFusion(
        candidate_modalities=modalities,
        signal_priors=priors,
        anchor_dim=128,
    )

    assert "temporal" in fusion.active_branches
    assert "motion" in fusion.active_branches

    # Forward pass test
    b_size = 4
    m13_feats = torch.randn(b_size, 128)
    sig_feats = {
        "temporal": torch.randn(b_size, 1),
        "motion": torch.randn(b_size, 1),
    }

    logits = fusion(m13_feats, sig_feats)
    assert logits.shape == (b_size, 1)

    report = fusion.gate_report()
    assert "temporal" in report
    assert "motion" in report


def test_step5_incremental_value_evaluation():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    # Realistic baseline with ranking confusion on missed fakes (AUC ~ 0.56, detection rate 0%)
    base_probs = [0.42, 0.38, 0.22, 0.18, 0.40, 0.35, 0.15, 0.10]
    cand_probs = [0.85, 0.88, 0.90, 0.92, 0.11, 0.14, 0.18, 0.22]

    res = evaluate_incremental_modality_value(
        y_true=y_true,
        baseline_probs=base_probs,
        candidate_probs=cand_probs,
        modality_name="temporal_flow",
    )

    assert res["modality"] == "temporal_flow"
    assert res["delta_auc"] > 0
    assert res["delta_detection_rate"] > 0
    assert res["adds_incremental_value"] is True


# ─────────────────────────────────────────────
# STEP 6 TESTS: STATISTICAL OOD PROMOTION GATE
# ─────────────────────────────────────────────
def test_step6_unseen_generator_promotion(tmp_path: Path):
    m13_results = [
        {"generator": "Sora-2024", "true_label": "FAKE", "prob": 0.35},
        {"generator": "Sora-2024", "true_label": "FAKE", "prob": 0.40},
        {"generator": "Sora-2024", "true_label": "REAL", "prob": 0.05},
        {"generator": "Sora-2024", "true_label": "REAL", "prob": 0.08},
    ]
    # M16 statistically improves detection on Sora while maintaining clean specificity
    m16_passing = [
        {"generator": "Sora-2024", "true_label": "FAKE", "prob": 0.85},
        {"generator": "Sora-2024", "true_label": "FAKE", "prob": 0.90},
        {"generator": "Sora-2024", "true_label": "REAL", "prob": 0.04},
        {"generator": "Sora-2024", "true_label": "REAL", "prob": 0.07},
    ]

    out_file = tmp_path / "ood_report.json"
    res, passed = evaluate_ood(m13_results, m16_passing, output_path=out_file, n_bootstraps=100)
    assert passed is True
    assert res["Sora-2024"]["passes"] is True
    assert "delta_auc" in res["Sora-2024"]
    assert "mcnemar_recoveries_c" in res["Sora-2024"]
    assert res["promotion_verdict"] == "PROMOTE M16"
