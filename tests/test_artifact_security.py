"""Tests for forensic artifact path containment and security guards."""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.api.app import app, resolve_artifact, ARTIFACT_ROOT


@pytest.fixture
def client():
    return TestClient(app)


def test_resolve_artifact_traversal_prevention():
    """Verify that path traversal attempts using .. or path separators are rejected."""
    # Simple traversal attempts
    with pytest.raises(HTTPException) as exc1:
        resolve_artifact("../secret.txt")
    assert exc1.value.status_code == 404

    with pytest.raises(HTTPException) as exc2:
        resolve_artifact("..\\windows\\system32\\cmd.exe")
    assert exc2.value.status_code == 404

    with pytest.raises(HTTPException) as exc3:
        resolve_artifact("/etc/passwd")
    assert exc3.value.status_code == 404

    # Empty filename
    with pytest.raises(HTTPException) as exc4:
        resolve_artifact("")
    assert exc4.value.status_code == 404


def test_resolve_artifact_valid_file():
    """Verify that a legitimate artifact file inside ARTIFACT_ROOT is resolved correctly."""
    test_file = ARTIFACT_ROOT / "test_valid_artifact.png"
    test_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    try:
        resolved = resolve_artifact("test_valid_artifact.png")
        assert resolved == test_file.resolve()
        assert resolved.is_file()
    finally:
        if test_file.exists():
            test_file.unlink()


def test_artifact_endpoint_http(client):
    """Verify HTTP GET /artifacts/forensics/{filename} responds properly."""
    # Non-existent artifact returns 404
    resp = client.get("/artifacts/forensics/nonexistent_artifact_xyz.png")
    assert resp.status_code == 404

    # Valid artifact returns 200
    test_file = ARTIFACT_ROOT / "http_test_artifact.png"
    test_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        resp = client.get("/artifacts/forensics/http_test_artifact.png")
        assert resp.status_code == 200
        assert resp.content == b"\x89PNG\r\n\x1a\n"
    finally:
        if test_file.exists():
            test_file.unlink()
