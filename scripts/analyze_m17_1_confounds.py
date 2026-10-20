"""Phase 1 Confound Analysis for Milestone 17.1.

Audits candidate confounds (resolution, bitrate, fps, codec, duration, file size)
between Real and Fake videos within Celeb-DF and each FF++ manipulation family.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("m17_1_confounds")

MANIFEST_CELEBDF = PROJECT_ROOT / "manifests" / "celebdf_v2.csv"
MANIFEST_MIXED = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "m17_1_confounds_audit.json"


def extract_stream_metadata(video_path: str) -> Dict[str, Any]:
    """Extract stream properties from raw video file."""
    p = Path(video_path)
    if not p.is_file():
        return {"error": "file_not_found"}

    file_size_mb = p.stat().st_size / (1024 * 1024)

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return {"error": "cannot_open"}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()

    duration_sec = count / max(1.0, fps)
    bitrate_kbps = (file_size_mb * 1024 * 8) / max(0.1, duration_sec)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / max(1, h), 4),
        "fps": round(fps, 2),
        "frame_count": count,
        "duration_sec": round(duration_sec, 2),
        "bitrate_kbps": round(bitrate_kbps, 1),
        "file_size_mb": round(file_size_mb, 3),
        "codec": fourcc_str,
    }


def compare_distributions(reals: List[float], fakes: List[float]) -> Dict[str, Any]:
    """Compute summary statistics and statistical tests between real and fake distributions."""
    r_arr = np.array(reals, dtype=np.float64)
    f_arr = np.array(fakes, dtype=np.float64)

    ks_stat, ks_pval = ks_2samp(r_arr, f_arr)
    try:
        mwu_stat, mwu_pval = mannwhitneyu(r_arr, f_arr, alternative="two-sided")
    except Exception:
        mwu_stat, mwu_pval = 0.0, 1.0

    return {
        "real_mean": float(np.mean(r_arr)),
        "real_std": float(np.std(r_arr)),
        "real_median": float(np.median(r_arr)),
        "fake_mean": float(np.mean(f_arr)),
        "fake_std": float(np.std(f_arr)),
        "fake_median": float(np.median(f_arr)),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "mwu_stat": float(mwu_stat),
        "mwu_pval": float(mwu_pval),
        "divergent": bool(ks_pval < 0.05),
    }


def main():
    logger.info("Starting M17.1 Real vs Fake Candidate Confound Analysis...")

    results: Dict[str, Any] = {}

    # 1. Celeb-DF v2 Test Split
    celeb_df = pd.read_csv(MANIFEST_CELEBDF)
    celeb_test = celeb_df[celeb_df["split"] == "test"].copy()
    logger.info(f"Auditing Celeb-DF-v2 test set ({len(celeb_test)} videos)...")

    celeb_records = []
    for _, row in celeb_test.iterrows():
        meta = extract_stream_metadata(row["video_path"])
        if "error" not in meta:
            meta["label"] = int(row["label"])
            meta["video_id"] = str(row["video_id"])
            celeb_records.append(meta)

    cdf = pd.DataFrame(celeb_records)
    c_reals = cdf[cdf["label"] == 0]
    c_fakes = cdf[cdf["label"] == 1]

    celeb_family_res = {
        "num_reals": len(c_reals),
        "num_fakes": len(c_fakes),
        "metrics": {},
        "codecs_real": c_reals["codec"].value_counts().to_dict(),
        "codecs_fake": c_fakes["codec"].value_counts().to_dict(),
    }
    for col in ["width", "height", "aspect_ratio", "fps", "bitrate_kbps", "duration_sec", "file_size_mb"]:
        celeb_family_res["metrics"][col] = compare_distributions(c_reals[col].tolist(), c_fakes[col].tolist())
    results["Celeb-DF-v2"] = celeb_family_res

    # 2. FF++ Test Split across 4 manipulation families
    mixed_df = pd.read_csv(MANIFEST_MIXED)
    mixed_test = mixed_df[mixed_df["split"] == "test"].copy()
    logger.info(f"Auditing FF++ test set ({len(mixed_test)} videos)...")

    ffpp_records = []
    for _, row in mixed_test.iterrows():
        meta = extract_stream_metadata(row["video_path"])
        if "error" not in meta:
            meta["label"] = int(row["label"])
            meta["manipulation_type"] = str(row["manipulation_type"])
            meta["video_id"] = str(row["video_id"])
            ffpp_records.append(meta)

    fdf = pd.DataFrame(ffpp_records)
    ffpp_clean = fdf[fdf["manipulation_type"] == "original"]

    for fam in ["Face2Face", "FaceSwap", "Deepfakes", "NeuralTextures"]:
        ffpp_fam = fdf[fdf["manipulation_type"] == fam]
        fam_res = {
            "num_reals": len(ffpp_clean),
            "num_fakes": len(ffpp_fam),
            "metrics": {},
            "codecs_real": ffpp_clean["codec"].value_counts().to_dict(),
            "codecs_fake": ffpp_fam["codec"].value_counts().to_dict(),
        }
        for col in ["width", "height", "aspect_ratio", "fps", "bitrate_kbps", "duration_sec", "file_size_mb"]:
            fam_res["metrics"][col] = compare_distributions(ffpp_clean[col].tolist(), ffpp_fam[col].tolist())
        results[f"FF++_{fam}"] = fam_res

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved confound analysis JSON to {OUTPUT_JSON}")

    # Log summary of key findings
    print("\n" + "=" * 80)
    print("M17.1 REAL vs. FAKE METADATA CONFOUND AUDIT SUMMARY")
    print("=" * 80)
    for domain, data in results.items():
        print(f"\n--- {domain} (Reals: {data['num_reals']}, Fakes: {data['num_fakes']}) ---")
        print(f"  Codecs: Reals={data['codecs_real']} | Fakes={data['codecs_fake']}")
        for col in ["width", "height", "bitrate_kbps", "fps", "duration_sec"]:
            m = data["metrics"][col]
            div_flag = "⚠️ DIVERGENT (p < 0.05)" if m["divergent"] else "✓ MATCHED"
            print(f"  {col:14s}: Real={m['real_mean']:.1f} (std={m['real_std']:.1f}) | Fake={m['fake_mean']:.1f} (std={m['fake_std']:.1f}) | KS-p={m['ks_pval']:.4f} {div_flag}")


if __name__ == "__main__":
    main()
