"""
Test suite for Groq AI forensic analyst module, canonical evidence schema,
numerical consistency validation, provenance tracing, causal guardrails,
telemetry grounding, and human-friendly investigator reporting.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.forensics.groq_analyst import (
    get_groq_api_key,
    is_groq_available,
    build_canonical_evidence,
    validate_numerical_consistency,
    validate_groq_report,
    get_report_validation_details,
    generate_safe_fallback_report,
    generate_human_friendly_forensic_analysis,
    extract_canonical_numbers,
    CanonicalForensicEvidence,
)


@pytest.fixture
def sample_audit_data():
    return {
        "video_filename": "id0_id20_0005.mp4",
        "video_fake_probability": 0.853,
        "confidence_interval": [0.625, 0.998],
        "verdict": "NO MANIPULATION DETECTED",
        "verdict_confidence": "Moderate Confidence",
        "kinematic_residual_energy": 15.31,
        "temporal_consistency_score": 0.842,
        "structured_telemetry": {
            "stream": {
                "source_file_name": "id0_id20_0005.mp4",
                "file_sha256": "cd70b725d60c6d7a",
                "width": 944,
                "height": 500,
                "container_fps": 30.0,
                "duration_sec": 15.3,
                "codec_name": "mpeg4",
                "decode_status": "SUCCESS",
            },
            "spatial": {
                "outer_boundary_energy": 0.1607,
                "inner_landmark_energy": 0.138,
                "boundary_to_landmark_ratio": 1.16,
                "spatial_frequency_anomaly_flag": False,
            },
            "temporal": {
                "total_windows_evaluated": 4,
                "total_frames_analyzed": 64,
                "frame_differential_rate": 15.31,
                "differential_energy_unit": "2D Euclidean pixel differential per second",
                "window_score_consistency": 0.842,
                "peak_window": {
                    "window_id": 3,
                    "calibrated_score": 0.998,
                    "start_sec": 6.0,
                    "end_sec": 8.0,
                    "face_confidence": 1.0,
                    "selection_policy": "Highest calibrated manipulation probability among evaluated temporal windows",
                },
            },
            "baseline": {
                "is_eligible": True,
                "reference_dataset": "FaceForensics++ C23 Validation",
                "sample_video_count": 1490,
                "kinematic_z_score": 2.93,
                "boundary_ratio_z_score": 0.52,
            },
            "quality": {
                "sufficient_coverage": True,
                "mean_face_confidence": 1.0,
                "confound_codes": [],
                "quality_warnings": [],
                "face_coverage_ratio": 0.99994,
            },
            "decision": {
                "decision_state": "SUPPORTING_MANIPULATION",
                "active_threshold": 0.92,
                "calibrated_probability": 0.853,
                "model_name": "M20PyramidalDualStream",
                "model_version": "2.0.0",
                "calibration_method": "Platt Scaling (Validation fit)",
            },
        },
    }


def test_get_groq_api_key_from_env():
    """Verify that GROQAPI or GROQ_API_KEY is extracted properly."""
    key = get_groq_api_key()
    assert key is not None
    assert key.startswith("gsk_")


def test_is_groq_available():
    """Check that is_groq_available returns boolean status."""
    assert is_groq_available() is True


def test_build_canonical_evidence_single_run_provenance(sample_audit_data):
    """Verify single-run provenance: model_score and dispersion must come from the same analysis run."""
    evidence = build_canonical_evidence(
        audit_data=sample_audit_data,
        pred_data={"probability": 0.5125, "decision": "FAKE", "engine": "v5-family-invariant"},
        threshold=0.50,
    )
    assert isinstance(evidence, CanonicalForensicEvidence)
    assert evidence.video_filename == "id0_id20_0005.mp4"
    assert evidence.decision_context["model_score"] == 0.853
    disp = evidence.decision_context["score_dispersion_range"]
    assert disp[0] <= evidence.decision_context["model_score"] <= disp[1]

    # Comparator engine is tracked separately without contaminating the primary score
    comp = evidence.decision_context["comparator_engine"]
    assert comp is not None
    assert comp["engine_name"] == "v5-family-invariant"
    assert comp["model_score"] == 0.5125
    assert evidence.decision_context["data_consistency_status"] == "CROSS_MODEL_DIVERGENCE"


def test_validate_numerical_consistency_catches_out_of_interval(sample_audit_data):
    """Verify that a point estimate outside stated dispersion fails numerical consistency validation."""
    evidence = build_canonical_evidence(sample_audit_data, threshold=0.50)
    evidence.decision_context["model_score"] = 0.5125
    evidence.decision_context["score_dispersion_range"] = [0.2074, 0.2474]
    
    is_valid, issues = validate_numerical_consistency(evidence)
    assert is_valid is False
    assert any("lies outside stated dispersion range" in issue for issue in issues)


def test_validate_groq_report_catches_unsupported_causal_claims(sample_audit_data):
    """Verify that validator catches claims of blending seams, unnatural velocity, or fake credentials."""
    evidence = build_canonical_evidence(sample_audit_data, threshold=0.92)
    
    flawed_report = """
    ## AI Video Analysis
    id0_id20_0005.mp4
    No manipulation detected

    ### What does this result mean?
    The score of 85.3% is below 92.0%. High face coverage proves tracking accuracy.

    ### What stood out in the analysis?
    A 1.16x boundary ratio indicates a blending seam.
    Kinematic deviation indicates unnatural velocity jumps.
    Window #3 indicates localized manipulation.

    ### What should be reviewed next?
    Review 6.0s to 8.0s.

    ### Important limitations
    Prepared by: Certified Audio-Visual Fraud Investigator.
    """
    is_valid, issues = validate_groq_report(flawed_report, evidence)
    assert is_valid is False
    assert any("human credential" in issue.lower() for issue in issues)
    assert any("blending seam" in issue.lower() for issue in issues)
    assert any("unnatural velocity jumps" in issue.lower() for issue in issues)
    assert any("localized manipulation" in issue.lower() for issue in issues)


def test_validate_groq_report_catches_developer_field_narration(sample_audit_data):
    """Verify that validator rejects developer field name narration."""
    evidence = build_canonical_evidence(sample_audit_data, threshold=0.92)
    narrated_report = """
    ## AI Video Analysis
    id0_id20_0005.mp4
    No manipulation detected

    ### What does this result mean?
    We evaluated field model_score at 85.3% against field active_threshold at 92.0%.
    """
    is_valid, issues = validate_groq_report(narrated_report, evidence)
    assert is_valid is False
    assert any("field name narration" in issue.lower() for issue in issues)


def test_validate_groq_report_catches_ungrounded_numerical_claims(sample_audit_data):
    """Verify that validator catches fabricated or ungrounded percentages."""
    evidence = build_canonical_evidence(sample_audit_data, threshold=0.92)
    ungrounded_report = """
    ## AI Video Analysis
    id0_id20_0005.mp4
    No manipulation detected

    ### What does this result mean?
    The system detected a 43.7% anomaly rating on the video.
    """
    is_valid, issues = validate_groq_report(ungrounded_report, evidence)
    assert is_valid is False
    assert any("ungrounded percentage claim" in issue.lower() for issue in issues)


def test_generate_safe_fallback_report_investigator_format(sample_audit_data):
    """Verify that fallback report follows the 4-section investigator briefing format."""
    evidence = build_canonical_evidence(
        sample_audit_data,
        pred_data={"probability": 0.5125, "decision": "FAKE", "engine": "v5-family-invariant"},
        threshold=0.50,
    )
    fallback = generate_safe_fallback_report(evidence)
    
    # 4 core sections
    assert "## AI Video Analysis" in fallback
    assert "### What does this result mean?" in fallback
    assert "### What stood out in the analysis?" in fallback
    assert "### What should be reviewed next?" in fallback
    assert "### Important limitations" in fallback
    
    # Divergence advisory
    assert "Cross-Engine Divergence Advisory" in fallback
    assert "v5-family-invariant" in fallback
    assert "51.25%" in fallback
    assert "M20PyramidalDualStream" in fallback
    assert "85.30%" in fallback
    
    # Evidence-grounded explanations
    assert "Changes between frames" in fallback
    assert "Consistency across time windows" in fallback
    assert "Boundary frequency contrast" in fallback
    assert "Technical measurements available in Structured Telemetry" in fallback
    
    # Does not contain raw developer field names
    assert "field model_score" not in fallback
    assert "field active_threshold" not in fallback


def test_groq_analyst_synthesis_api_endpoint(sample_audit_data):
    """Test the POST /api/v1/groq-analyst-synthesis endpoint returns validated 4-section briefing."""
    client = TestClient(app)

    sample_request = {
        "audit_data": sample_audit_data,
        "prediction_data": {
            "probability": 0.853,
            "decision": "REAL",
            "engine": "m22-ensemble",
        },
        "threshold": 0.92,
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [
            {
                "message": {
                    "content": (
                        "## AI Video Analysis\n"
                        "id0_id20_0005.mp4\n"
                        "No manipulation detected\n\n"
                        "### What does this result mean?\n"
                        "The detection system did not flag this video because its model score was 85.30%, "
                        "below the active decision threshold of 92.00%. Under the configured rule, this produces "
                        "a no-manipulation-detected verdict. This is an automated result and not standalone proof of authenticity.\n\n"
                        "### What stood out in the analysis?\n"
                        "• Changes between frames\n"
                        "The system measured an average frame-differential energy of 15.31. On its own, it cannot tell us whether that movement is natural or manipulated.\n\n"
                        "• Consistency across time windows\n"
                        "The dashboard reports 84.2% window-score consistency across 4 windows, with a peak response of 99.8% in Window #3 between 6.0s and 8.0s.\n\n"
                        "### What should be reviewed next?\n"
                        "Review individual temporal-window scores and frames from 6.0s to 8.0s.\n\n"
                        "### Important limitations\n"
                        "This report summarizes automated analysis. It does not establish the specific method of manipulation.\n\n"
                        "*AI-generated analytical assistance · Technical measurements available in Structured Telemetry*"
                    )
                }
            }
        ]
    }).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        resp = client.post("/api/v1/groq-analyst-synthesis", json=sample_request)
        assert resp.status_code == 200
        body = resp.json()
        assert "groq_report" in body
        assert "## AI Video Analysis" in body["groq_report"]
        assert "What does this result mean?" in body["groq_report"]
        assert body["validated"] is True
        assert body["validation_level"] == "VERIFIED_CANONICAL"
        assert body["canonical_evidence"] is not None
        assert body["canonical_evidence"]["video_filename"] == "id0_id20_0005.mp4"
