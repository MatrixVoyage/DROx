"""Integrity test verifying that frozen M10 and M12 baseline artifacts remain byte-identical."""

import hashlib
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREEZE_PATH = PROJECT_ROOT / "reports" / "m12_baseline_freeze.json"


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    assert filepath.is_file(), f"File does not exist: {filepath}"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_m12_frozen_baseline_artifacts_integrity():
    """Verify that all frozen M12 model, calibration, and reference artifacts are bitwise identical."""
    assert FREEZE_PATH.is_file(), f"Freeze manifest not found: {FREEZE_PATH}"
    with open(FREEZE_PATH, "r") as f:
        freeze_data = json.load(f)

    fingerprints = freeze_data["fingerprints"]

    # 1. M10 Checkpoint
    m10_path = PROJECT_ROOT / fingerprints["m10_checkpoint"]["path"]
    assert sha256_file(m10_path) == fingerprints["m10_checkpoint"]["sha256"], (
        f"M10 checkpoint {m10_path} has been modified!"
    )

    # 2. M12 Checkpoint
    m12_path = PROJECT_ROOT / fingerprints["m12_checkpoint"]["path"]
    assert sha256_file(m12_path) == fingerprints["m12_checkpoint"]["sha256"], (
        f"M12 checkpoint {m12_path} has been modified!"
    )

    # 3. M12 OOD Reference
    ood_path = PROJECT_ROOT / fingerprints["ood_reference"]["path"]
    assert sha256_file(ood_path) == fingerprints["ood_reference"]["sha256"], (
        f"M12 OOD reference {ood_path} has been modified!"
    )

    # 4. M12 Calibration Config
    calib_path = PROJECT_ROOT / fingerprints["calibration_config"]["path"]
    assert sha256_file(calib_path) == fingerprints["calibration_config"]["sha256"], (
        f"M12 calibration config {calib_path} has been modified!"
    )
