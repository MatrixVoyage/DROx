"""Unit tests verifying technical dashboard assets and endpoints."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_dashboard_index_endpoint(client):
    """Verify GET / serves index.html with expected UI elements."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "Deepfake Video Detection" in html
    assert "Research & Operational Limitations" in html
    assert "dropzone" in html
    assert "thresholdSlider" in html
    assert "verdictBox" in html
    assert "rejectAlertBanner" in html
    assert "diagnosticContainer" in html
    assert "v3-robust" in html
    assert "v3-robust (M12)" in html
    assert "prepLatencyBar" in html
    assert "provArch" in html
    assert "forensicAuditBtn" in html
    assert "forensicAuditContainer" in html
    assert "forensicSaliencyImg" in html


def test_dashboard_static_css(client):
    """Verify static CSS stylesheet is served."""
    resp = client.get("/static/dashboard.css")
    assert resp.status_code == 200
    assert "--bg-base" in resp.text
    assert "--state-warning" in resp.text
    assert ".verdict-box.reject" in resp.text
    assert ".diagnostic-container" in resp.text
    assert ".btn-forensic" in resp.text
    assert ".forensic-saliency-box" in resp.text
    assert ".forensic-window-card" in resp.text


def test_dashboard_static_js(client):
    """Verify static JavaScript bundle is served."""
    resp = client.get("/static/dashboard.js")
    assert resp.status_code == 200
    assert "api/v1/predict" in resp.text
    assert "renderPrediction" in resp.text
    assert "REJECT ≠ FAKE" in resp.text
    assert "diagOodBadge" in resp.text
    assert "forensicAuditBtn" in resp.text
    assert "api/v1/forensic-audit" in resp.text
    assert "renderForensicAudit" in resp.text

