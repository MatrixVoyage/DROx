"""Milestone 14 Phase A: Model Immutability & Data Lineage Audit.

Enforces the M14 Model Immutability Contract:
1. Frozen Checkpoint bitwise SHA-256 hashes (M12, M13.2b, v5 champion)
2. Calibration and Operating Threshold immutability (T=2.0502, tau*=0.1260 in temperature_scaled_probability)
3. Zero group leakage between Celeb-DF v2 test benchmark (518 videos) and M13 train/val splits
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from src.data.manifest import validate_split_integrity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
M12_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
M13_2B_PATH = PROJECT_ROOT / "checkpoints" / "m13_2b_stage5" / "best.pth"
M13_4_PATH = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
CALIB_PATH = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"
CELEBDF_MANIFEST = PROJECT_ROOT / "manifests" / "celebdf_v2.csv"
LOFO_F2F_MANIFEST = PROJECT_ROOT / "manifests" / "lofo_f2f.csv"

M12_EXPECTED_HASH = "c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56"
M13_2B_EXPECTED_HASH = "d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611"
V5_CHAMPION_EXPECTED_HASH = "4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_m14_frozen_checkpoint_hashes():
    """Verify bitwise immutability of all baseline and champion model checkpoints."""
    assert M12_PATH.is_file(), f"M12 checkpoint missing at {M12_PATH}"
    assert sha256_file(M12_PATH) == M12_EXPECTED_HASH, "M12 baseline hash altered"

    assert M13_2B_PATH.is_file(), f"M13.2b checkpoint missing at {M13_2B_PATH}"
    assert sha256_file(M13_2B_PATH) == M13_2B_EXPECTED_HASH, "M13.2b baseline hash altered"

    assert M13_4_PATH.is_file(), f"v5 champion checkpoint missing at {M13_4_PATH}"
    assert sha256_file(M13_4_PATH) == V5_CHAMPION_EXPECTED_HASH, "v5 champion hash altered"


def test_m14_frozen_calibration_contract():
    """Verify calibration parameters and decision thresholds remain frozen."""
    assert CALIB_PATH.is_file(), f"calibration_config.json missing at {CALIB_PATH}"
    with open(CALIB_PATH, "r") as f:
        calib_data = json.load(f)

    assert calib_data["score_space"] == "temperature_scaled_probability"
    assert calib_data["temperature"] == 2.0502
    assert calib_data["threshold_standard"] == 0.5000
    assert calib_data["threshold_high_sensitivity"] == 0.1260
    assert calib_data["threshold_selection_source"] == "validation_only"

    # Verify diagnostic OOD
    assert calib_data["uncertainty_safety"]["ood_status"] == "diagnostic"
    assert calib_data["uncertainty_safety"]["reject_on_ood"] is False


def test_m14_celebdf_lineage_and_zero_leakage():
    """Verify official Celeb-DF v2 test partition has 518 videos and zero leakage into M13 train/val."""
    assert CELEBDF_MANIFEST.is_file(), f"Missing Celeb-DF manifest at {CELEBDF_MANIFEST}"
    df_celeb = pd.read_csv(CELEBDF_MANIFEST)

    test_df = df_celeb[df_celeb["split"] == "test"]
    assert len(test_df) == 518, f"Expected 518 test videos, found {len(test_df)}"
    assert (test_df["label"] == 1).sum() == 340, "Expected 340 fake videos in test set"
    assert (test_df["label"] == 0).sum() == 178, "Expected 178 real videos in test set"

    # Lineage check against M13 LOFO training manifest
    assert LOFO_F2F_MANIFEST.is_file(), f"Missing LOFO-F2F manifest at {LOFO_F2F_MANIFEST}"
    df_lofo = pd.read_csv(LOFO_F2F_MANIFEST)

    lofo_train_vids = set(df_lofo[df_lofo["split"] == "train"]["video_id"])
    lofo_val_vids = set(df_lofo[df_lofo["split"] == "val"]["video_id"])
    celeb_test_vids = set(test_df["video_id"])

    # Video ID intersection
    assert lofo_train_vids.isdisjoint(celeb_test_vids), "Test videos leaked into LOFO training split!"
    assert lofo_val_vids.isdisjoint(celeb_test_vids), "Test videos leaked into LOFO validation split!"

    # Group ID intersection
    lofo_train_groups = set(df_lofo[df_lofo["split"] == "train"]["group_id"])
    lofo_val_groups = set(df_lofo[df_lofo["split"] == "val"]["group_id"])
    celeb_test_groups = set(test_df["group_id"])

    assert lofo_train_groups.isdisjoint(celeb_test_groups), "Test celebrity identities leaked into training split!"
    assert lofo_val_groups.isdisjoint(celeb_test_groups), "Test celebrity identities leaked into validation split!"
