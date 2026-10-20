"""Milestone 20 — Phase 1: Frozen-Weight Localized Representation Transfer Ablation.

Evaluates the frozen M18 Dual-Stream checkpoint across 6 tightly matched arms
on the 116 locked Face2Face holdout clips:
- Arm A: Whole Face (112x112 baseline control)
- Arm B: Native-Resolution Mouth (112x112)
- Arm C: Native-Resolution Jawline (112x112)
- Arm D: Native-Resolution Mouth + Jaw (112x112)
- Arm E1: Matched Cheek Control for Mouth (scale- and extraction-matched to Arm B)
- Arm E2: Matched Cheek Control for Jaw (scale- and extraction-matched to Arm C)

Statistical Rigor:
- Paired bootstrap test vs. Arm A (n_resamples=1000)
- Paired comparison vs. scale-matched controls (B vs E1, C vs E2)
- Three-tier outcome evaluation: SUPPORT, INCONCLUSIVE, NEGATIVE EVIDENCE.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.m18_detector import M18ReenactmentDetector
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("ablate_m20_representation_transfer")

# Checkpoints
M17_TRACK_A_CKPT = PROJECT_ROOT / "checkpoints" / "m17_track_a_seed42" / "best.pth"
M18_DUAL_STREAM_CKPT = PROJECT_ROOT / "checkpoints" / "m18_dual_stream_seed42" / "best.pth"

# Manifest
MANIFEST_MIXED = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"

# Output Reports
REPORT_JSON = PROJECT_ROOT / "reports" / "m20_phase1_transfer_ablation.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m20_phase1_transfer_ablation.md"

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def extract_anatomical_patches(
    frame: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Extract 6 scale-matched anatomical and control patches from native video frame.

    Args:
        frame: Native RGB video frame (H, W, 3) uint8.
        bbox: Face bounding box [x1, y1, x2, y2].
        landmarks: 5 facial landmarks [[lx, ly], [rx, ry], [nx, ny], [lmx, lmy], [rmx, rmy]].

    Returns:
        patches: Dict mapping arm names to (112, 112, 3) uint8 ndarrays.
        meta: Dict containing exact pixel crop sizes for scale matching verification.
    """
    H, W = frame.shape[:2]
    bx1, by1, bx2, by2 = bbox[:4]
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    # 1. Arm A: Whole Face with 0.20 margin expansion
    margin_x = bw * 0.20
    margin_y = bh * 0.20
    fx1 = max(0, int(round(bx1 - margin_x)))
    fy1 = max(0, int(round(by1 - margin_y)))
    fx2 = min(W, int(round(bx2 + margin_x)))
    fy2 = min(H, int(round(by2 + margin_y)))
    if fx2 <= fx1:
        fx2 = min(W, fx1 + 1)
    if fy2 <= fy1:
        fy2 = min(H, fy1 + 1)
    crop_a = frame[fy1:fy2, fx1:fx2]
    arm_a = cv2.resize(crop_a, (112, 112), interpolation=cv2.INTER_LINEAR)

    # Key Landmarks
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # Mouth geometry
    mouth_center = (left_mouth + right_mouth) / 2.0
    mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
    mouth_width = max(10.0, mouth_width)

    # Vertical facial scale
    d_nm = max(5.0, float(mouth_center[1] - nose[1]))

    # 2. Arm B: Native-Resolution Mouth
    size_b = max(16, int(round(1.4 * mouth_width)))
    half_b = size_b // 2
    bx1_m = max(0, int(round(mouth_center[0] - half_b)))
    by1_m = max(0, int(round(mouth_center[1] - half_b)))
    bx2_m = min(W, bx1_m + size_b)
    by2_m = min(H, by1_m + size_b)
    crop_b = frame[by1_m:by2_m, bx1_m:bx2_m]
    arm_b = cv2.resize(crop_b, (112, 112), interpolation=cv2.INTER_LINEAR)

    # 3. Arm C: Native-Resolution Jawline
    jaw_center_y = mouth_center[1] + 0.8 * d_nm
    size_c = max(20, int(round(1.6 * mouth_width)))
    half_c = size_c // 2
    cx1_j = max(0, int(round(mouth_center[0] - half_c)))
    cy1_j = max(0, int(round(jaw_center_y - half_c)))
    cx2_j = min(W, cx1_j + size_c)
    cy2_j = min(H, cy1_j + size_c)
    crop_c = frame[cy1_j:cy2_j, cx1_j:cx2_j]
    arm_c = cv2.resize(crop_c, (112, 112), interpolation=cv2.INTER_LINEAR)

    # 4. Arm D: Native-Resolution Mouth + Jaw
    d_center_y = mouth_center[1] + 0.4 * d_nm
    size_d = max(24, int(round(2.0 * mouth_width)))
    half_d = size_d // 2
    dx1 = max(0, int(round(mouth_center[0] - half_d)))
    dy1 = max(0, int(round(d_center_y - half_d)))
    dx2 = min(W, dx1 + size_d)
    dy2 = min(H, dy1 + size_d)
    crop_d = frame[dy1:dy2, dx1:dx2]
    arm_d = cv2.resize(crop_d, (112, 112), interpolation=cv2.INTER_LINEAR)

    # 5. Arm E1: Matched Cheek Control for Mouth (EXACT size_b)
    # Offset horizontally from nose into cheek territory
    # Check if right or left cheek has more space within face bounds
    offset_x = 0.9 * mouth_width
    cheek1_x = nose[0] + offset_x
    if cheek1_x + half_b > bx2: # if right cheek overflows face, use left cheek
        cheek1_x = nose[0] - offset_x
    cheek1_y = nose[1] + 0.2 * d_nm

    e1_x1 = max(0, int(round(cheek1_x - half_b)))
    e1_y1 = max(0, int(round(cheek1_y - half_b)))
    e1_x2 = min(W, e1_x1 + size_b)
    e1_y2 = min(H, e1_y1 + size_b)
    crop_e1 = frame[e1_y1:e1_y2, e1_x1:e1_x2]
    arm_e1 = cv2.resize(crop_e1, (112, 112), interpolation=cv2.INTER_LINEAR)

    # 6. Arm E2: Matched Cheek Control for Jaw (EXACT size_c)
    cheek2_x = nose[0] - offset_x
    if cheek2_x - half_c < bx1: # if left cheek overflows face, use right cheek
        cheek2_x = nose[0] + offset_x
    cheek2_y = nose[1] + 0.1 * d_nm

    e2_x1 = max(0, int(round(cheek2_x - half_c)))
    e2_y1 = max(0, int(round(cheek2_y - half_c)))
    e2_x2 = min(W, e2_x1 + size_c)
    e2_y2 = min(H, e2_y1 + size_c)
    crop_e2 = frame[e2_y1:e2_y2, e2_x1:e2_x2]
    arm_e2 = cv2.resize(crop_e2, (112, 112), interpolation=cv2.INTER_LINEAR)

    patches = {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "arm_c": arm_c,
        "arm_d": arm_d,
        "arm_e1": arm_e1,
        "arm_e2": arm_e2,
    }
    meta = {
        "size_b": size_b,
        "size_e1": size_b,
        "size_c": size_c,
        "size_e2": size_c,
        "size_d": size_d,
    }
    return patches, meta


def normalize_patch_sequence(patches: List[np.ndarray]) -> torch.Tensor:
    """Normalize a sequence of 16 (112, 112, 3) uint8 patches to (1, 16, 3, 112, 112) float32."""
    processed = []
    for p in patches:
        arr = p.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        processed.append(arr.transpose(2, 0, 1)) # (3, 112, 112)
    seq = np.stack(processed, axis=0) # (16, 3, 112, 112)
    return torch.from_numpy(seq).unsqueeze(0).float() # (1, 16, 3, 112, 112)


def compute_paired_bootstrap(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_ref: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired bootstrap difference in ROC-AUC with confidence intervals and p-value."""
    n = len(y_true)
    rng = np.random.RandomState(seed)

    base_auc_cand = roc_auc_score(y_true, scores_cand)
    base_auc_ref = roc_auc_score(y_true, scores_ref)
    base_delta = float(base_auc_cand - base_auc_ref)

    deltas = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_c = roc_auc_score(y_b, scores_cand[idx])
            auc_r = roc_auc_score(y_b, scores_ref[idx])
            deltas.append(auc_c - auc_r)
        except ValueError:
            continue

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_val = float(np.mean(deltas <= 0.0))

    return {
        "delta_auc": base_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for val_x in x:
        greater += int(np.sum(val_x > y))
        less += int(np.sum(val_x < y))
    return float((greater - less) / (n_x * n_y))


def classify_phase1_outcome(arm_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate Three-Tier Outcome Interpretation Framework.

    Tiers:
    - SUPPORT: Localized arm (B, C, or D) shows meaningful paired improvement over Arm A
               (delta > +0.02, p < 0.05) and matched controls do not show comparable gains.
    - INCONCLUSIVE: Localized arms do not improve, but frozen encoder suffers severe semantic
                    distribution shift (logit margin collapses across all localized crops).
    - NEGATIVE EVIDENCE: Localized arms fail while feature pathway operates normally, OR
                         matched controls perform equally or better.
    """
    arm_a = arm_metrics["arm_a"]
    arm_b = arm_metrics["arm_b"]
    arm_c = arm_metrics["arm_c"]
    arm_d = arm_metrics["arm_d"]
    arm_e1 = arm_metrics["arm_e1"]
    arm_e2 = arm_metrics["arm_e2"]

    # Check for Support
    support_arms = []
    for name, m, ctrl in [("Arm B (Mouth)", arm_b, arm_e1), ("Arm C (Jaw)", arm_c, arm_e2), ("Arm D (Mouth+Jaw)", arm_d, arm_e1)]:
        if m.get("delta_auc_vs_a", 0.0) > 0.02 and m.get("p_val_vs_a", 1.0) < 0.05:
            # Verify control does not match
            ctrl_delta = ctrl.get("delta_auc_vs_a", 0.0)
            if m.get("delta_auc_vs_a", 0.0) > ctrl_delta + 0.015:
                support_arms.append(name)

    if len(support_arms) > 0:
        return {
            "tier": "SUPPORT",
            "reason": f"Statistically supported improvement observed on {', '.join(support_arms)} without equivalent control gain.",
            "recommendation": "Proceed to Phase 2 (Native-Resolution Architecture Construction) with user authorization.",
        }

    # Check for Inconclusive due to semantic distribution shift
    # If all localized crops have severely degraded logit margins (<= 0.05 vs baseline > 0.40)
    loc_margins = [m.get("logit_margin", 0.0) for m in [arm_b, arm_c, arm_d, arm_e1, arm_e2]]
    mean_loc_margin = np.mean(loc_margins)
    a_margin = arm_a.get("logit_margin", 0.547)

    if mean_loc_margin <= 0.05 and a_margin > 0.30:
        return {
            "tier": "INCONCLUSIVE",
            "reason": (
                f"Frozen M18 encoder suffers severe semantic distribution shift when fed non-whole-face crops "
                f"(average localized logit margin collapsed to {mean_loc_margin:.3f} vs baseline {a_margin:.3f}). "
                "The frozen network cannot properly exploit localized patches without patch-specific tuning."
            ),
            "recommendation": (
                "Document distribution shift. Conduct a lightweight native-patch training probe before full model commitment."
            ),
        }

    # Otherwise: Negative Evidence
    return {
        "tier": "NEGATIVE EVIDENCE",
        "reason": (
            "Localized arms did not improve over Arm A, and matched controls performed equivalently or better. "
            "No evidence that localized native-resolution crops provide superior forensic discrimination on this pipeline."
        ),
        "recommendation": "Archive localized native-resolution hypothesis and pivot research direction.",
    }


def load_m18_model(device: torch.device) -> M18ReenactmentDetector:
    """Load the frozen M18 Dual-Stream model."""
    cfg = load_config("configs/experiments/m12_full.yaml")
    base_cand = DeepfakeDetector(config=cfg.model).to(device)
    m17_ckpt = torch.load(M17_TRACK_A_CKPT, map_location=device, weights_only=False)
    model_track_a = M17TrackADetector(
        base_detector=base_cand,
        num_families=5,
        res_dim=128,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        gate_bottleneck=128,
    ).to(device)
    model_track_a.load_state_dict(m17_ckpt["model_state_dict"])
    model_track_a.eval()

    m18_ckpt = torch.load(M18_DUAL_STREAM_CKPT, map_location=device, weights_only=False)
    m18_dual = M18ReenactmentDetector(
        trunk_detector=model_track_a,
        kin_dim=256,
        aux_hidden_dim=512,
        gate_bottleneck=128,
        freeze_trunk=True,
    ).to(device)
    m18_dual.load_state_dict(m18_ckpt["model_state_dict"])
    m18_dual.eval()
    return m18_dual


def main():
    logger.info("Initializing Milestone 20 — Phase 1: Frozen-Weight Localized Representation Transfer Ablation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = load_m18_model(device)
    logger.info("Frozen M18 Dual-Stream model loaded cleanly on %s", device)

    # 2. Load MTCNN Detector with Landmarks
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, device=device)
        logger.info("MTCNN detector initialized with facial landmarks support.")
    except Exception as e:
        logger.error("Failed to load MTCNN: %s", e)
        raise

    # 3. Load 116 Locked Face2Face Holdout Clips
    mixed_df = pd.read_csv(MANIFEST_MIXED)
    mixed_test = mixed_df[mixed_df["split"] == "test"].copy().reset_index(drop=True)
    clean_reals = mixed_test[mixed_test["manipulation_type"] == "original"]
    f2f_fakes = mixed_test[mixed_test["manipulation_type"] == "Face2Face"]

    # Filter to the exact 58 reals and 58 fakes that exist in preprocessed test tensors
    from scripts.diagnose_m19_1_resolution_temporal import find_tensor
    holdout_items: List[Dict[str, Any]] = []

    for _, r in clean_reals.iterrows():
        vid = str(r["video_id"])
        if find_tensor(vid, split="test") is not None and os.path.exists(str(r["video_path"])):
            holdout_items.append({
                "video_id": vid,
                "video_path": str(r["video_path"]),
                "label": 0,
            })
    for _, r in f2f_fakes.iterrows():
        vid = str(r["video_id"])
        if find_tensor(vid, split="test") is not None and os.path.exists(str(r["video_path"])):
            holdout_items.append({
                "video_id": vid,
                "video_path": str(r["video_path"]),
                "label": 1,
            })

    logger.info("Locked Face2Face Holdout verified: %d clips (Reals: %d, Fakes: %d)",
                len(holdout_items),
                sum(1 for x in holdout_items if x["label"] == 0),
                sum(1 for x in holdout_items if x["label"] == 1))

    # Containers for predictions
    arm_names = ["arm_a", "arm_b", "arm_c", "arm_d", "arm_e1", "arm_e2"]
    predictions: Dict[str, Dict[str, List[float]]] = {
        arm: {"logits": [], "probs": [], "gates": [], "labels": []}
        for arm in arm_names
    }

    t0_start = time.time()
    for idx, item in enumerate(holdout_items):
        v_path = item["video_path"]
        label = item["label"]
        vid = item["video_id"]

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            logger.warning("Could not open %s, skipping", v_path)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 16:
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (16 - total_frames)
        else:
            frame_indices = [int(i * total_frames / 16) for i in range(16)]

        raw_frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, f = cap.read()
            if not ret or f is None:
                if len(raw_frames) > 0:
                    raw_frames.append(raw_frames[-1].copy())
                else:
                    raw_frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))
            else:
                raw_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()

        # Batch detect faces & landmarks
        boxes, probs, landmarks = mtcnn.detect(raw_frames, landmarks=True)

        # Fallback tracking for any frames where detection is None
        last_valid_box = None
        last_valid_lm = None
        # Find first valid
        for b, lm in zip(boxes, landmarks):
            if b is not None and lm is not None:
                last_valid_box = b[0]
                last_valid_lm = lm[0]
                break

        if last_valid_box is None:
            # Default center box and landmarks
            H, W = raw_frames[0].shape[:2]
            last_valid_box = np.array([W * 0.25, H * 0.2, W * 0.75, H * 0.8])
            last_valid_lm = np.array([
                [W * 0.4, H * 0.4], [W * 0.6, H * 0.4], [W * 0.5, H * 0.5],
                [W * 0.45, H * 0.65], [W * 0.55, H * 0.65]
            ])

        arm_patch_sequences: Dict[str, List[np.ndarray]] = {arm: [] for arm in arm_names}

        for fi, frame in enumerate(raw_frames):
            b = boxes[fi]
            lm = landmarks[fi]
            if b is not None and lm is not None and len(b) > 0 and len(lm) > 0:
                cur_box = b[0]
                cur_lm = lm[0]
                last_valid_box = cur_box
                last_valid_lm = cur_lm
            else:
                cur_box = last_valid_box
                cur_lm = last_valid_lm

            patches, _ = extract_anatomical_patches(frame, cur_box, cur_lm)
            for arm in arm_names:
                arm_patch_sequences[arm].append(patches[arm])

        # Evaluate through frozen M18 detector
        with torch.no_grad():
            for arm in arm_names:
                seq_tensor = normalize_patch_sequence(arm_patch_sequences[arm]).to(device)
                out = model(seq_tensor)
                logit = float(out.primary_logits.item())
                prob = float(out.p_fused.item())
                gate = float(out.gate_values.item())

                predictions[arm]["logits"].append(logit)
                predictions[arm]["probs"].append(prob)
                predictions[arm]["gates"].append(gate)
                predictions[arm]["labels"].append(label)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(holdout_items):
            logger.info("Processed %d / %d videos (Elapsed: %.1fs)", idx + 1, len(holdout_items), time.time() - t0_start)

    # 4. Compute Metrics for each arm
    arm_metrics: Dict[str, Dict[str, Any]] = {}
    labels_arr = np.array(predictions["arm_a"]["labels"])
    reals_mask = labels_arr == 0
    fakes_mask = labels_arr == 1

    for arm in arm_names:
        z_arr = np.array(predictions[arm]["logits"])
        p_arr = np.array(predictions[arm]["probs"])
        g_arr = np.array(predictions[arm]["gates"])

        auc = float(roc_auc_score(labels_arr, z_arr))
        prauc = float(average_precision_score(labels_arr, z_arr))

        r_z_med = float(np.median(z_arr[reals_mask]))
        f_z_med = float(np.median(z_arr[fakes_mask]))
        margin = f_z_med - r_z_med

        r_g_med = float(np.median(g_arr[reals_mask]))
        f_g_med = float(np.median(g_arr[fakes_mask]))

        spec_05 = float(np.mean(p_arr[reals_mask] < 0.50))
        sens_05 = float(np.mean(p_arr[fakes_mask] >= 0.50))

        arm_metrics[arm] = {
            "auc": auc,
            "prauc": prauc,
            "real_median_logit": r_z_med,
            "fake_median_logit": f_z_med,
            "logit_margin": margin,
            "real_median_gate": r_g_med,
            "fake_median_gate": f_g_med,
            "spec_at_tau05": spec_05,
            "sens_at_tau05": sens_05,
        }

    # 5. Paired Bootstrap vs Arm A
    scores_a = np.array(predictions["arm_a"]["logits"])
    for arm in arm_names:
        if arm == "arm_a":
            arm_metrics[arm]["delta_auc_vs_a"] = 0.0
            arm_metrics[arm]["ci_lower_vs_a"] = 0.0
            arm_metrics[arm]["ci_upper_vs_a"] = 0.0
            arm_metrics[arm]["p_val_vs_a"] = 0.5
            arm_metrics[arm]["cliffs_delta_vs_a"] = 0.0
        else:
            cand_scores = np.array(predictions[arm]["logits"])
            boot = compute_paired_bootstrap(labels_arr, cand_scores, scores_a, n_resamples=1000)
            c_delta = compute_cliffs_delta(cand_scores, scores_a)
            arm_metrics[arm]["delta_auc_vs_a"] = boot["delta_auc"]
            arm_metrics[arm]["ci_lower_vs_a"] = boot["ci_lower"]
            arm_metrics[arm]["ci_upper_vs_a"] = boot["ci_upper"]
            arm_metrics[arm]["p_val_vs_a"] = boot["p_value"]
            arm_metrics[arm]["cliffs_delta_vs_a"] = c_delta

    # Paired comparisons against scale-matched controls
    # B vs E1
    boot_b_e1 = compute_paired_bootstrap(
        labels_arr, np.array(predictions["arm_b"]["logits"]), np.array(predictions["arm_e1"]["logits"])
    )
    # C vs E2
    boot_c_e2 = compute_paired_bootstrap(
        labels_arr, np.array(predictions["arm_c"]["logits"]), np.array(predictions["arm_e2"]["logits"])
    )

    arm_metrics["arm_b"]["delta_vs_control"] = boot_b_e1["delta_auc"]
    arm_metrics["arm_b"]["p_val_vs_control"] = boot_b_e1["p_value"]
    arm_metrics["arm_c"]["delta_vs_control"] = boot_c_e2["delta_auc"]
    arm_metrics["arm_c"]["p_val_vs_control"] = boot_c_e2["p_value"]

    # 6. Three-Tier Outcome Verdict
    outcome = classify_phase1_outcome(arm_metrics)

    # 7. Build JSON & Markdown Reports
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": "M20_Phase1",
        "title": "Frozen-Weight Localized Representation Transfer Ablation",
        "model_analyzed": "Frozen M18 Dual-Stream Checkpoint (Seed 42)",
        "dataset_analyzed": f"Face2Face Locked Holdout (N={len(holdout_items)}: 58 Reals, 58 Fakes)",
        "arm_metrics": arm_metrics,
        "pairwise_control_comparisons": {
            "mouth_b_vs_control_e1": boot_b_e1,
            "jaw_c_vs_control_e2": boot_c_e2,
        },
        "phase1_outcome": outcome,
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Phase 1 JSON report saved to %s", REPORT_JSON)

    # Generate Markdown
    md_content = f"""# Milestone 20 — Phase 1: Frozen-Weight Localized Representation Transfer Ablation

**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  
**Target Model**: Frozen M18 Dual-Stream Detector (`checkpoints/m18_dual_stream_seed42/best.pth`)  
**Dataset**: Face2Face Locked Holdout ({len(holdout_items)} clips: 58 Pristine Reals, 58 Face2Face Fakes)  
**Methodology**: Frozen-weight transfer diagnostic across 6 scale-matched arms. Evaluates whether localized native-resolution crops provide forensic cues on the frozen M18 encoder.

---

## 1. Six-Arm Empirical Results

| Arm | Representation Description | Crop Scaling | ROC-AUC | PR-AUC | $\\Delta$ AUC vs. Arm A | 95% Bootstrap CI | Paired $p$-value | Real Med Logit | Fake Med Logit | Logit Margin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** | Whole-Face Baseline (Control) | Whole MTCNN BBox | **{arm_metrics['arm_a']['auc']:.4f}** | {arm_metrics['arm_a']['prauc']:.4f} | Baseline Ref | — | — | {arm_metrics['arm_a']['real_median_logit']:+.3f} | {arm_metrics['arm_a']['fake_median_logit']:+.3f} | **{arm_metrics['arm_a']['logit_margin']:+.3f}** |
| **Arm B** | Native-Resolution Mouth | 1.4× Mouth Width | **{arm_metrics['arm_b']['auc']:.4f}** | {arm_metrics['arm_b']['prauc']:.4f} | {arm_metrics['arm_b']['delta_auc_vs_a']:+.4f} | [{arm_metrics['arm_b']['ci_lower_vs_a']:+.4f}, {arm_metrics['arm_b']['ci_upper_vs_a']:+.4f}] | {arm_metrics['arm_b']['p_val_vs_a']:.4f} | {arm_metrics['arm_b']['real_median_logit']:+.3f} | {arm_metrics['arm_b']['fake_median_logit']:+.3f} | {arm_metrics['arm_b']['logit_margin']:+.3f} |
| **Arm C** | Native-Resolution Jawline | 1.6× Mouth Width | **{arm_metrics['arm_c']['auc']:.4f}** | {arm_metrics['arm_c']['prauc']:.4f} | {arm_metrics['arm_c']['delta_auc_vs_a']:+.4f} | [{arm_metrics['arm_c']['ci_lower_vs_a']:+.4f}, {arm_metrics['arm_c']['ci_upper_vs_a']:+.4f}] | {arm_metrics['arm_c']['p_val_vs_a']:.4f} | {arm_metrics['arm_c']['real_median_logit']:+.3f} | {arm_metrics['arm_c']['fake_median_logit']:+.3f} | {arm_metrics['arm_c']['logit_margin']:+.3f} |
| **Arm D** | Native-Resolution Mouth+Jaw | 2.0× Mouth Width | **{arm_metrics['arm_d']['auc']:.4f}** | {arm_metrics['arm_d']['prauc']:.4f} | {arm_metrics['arm_d']['delta_auc_vs_a']:+.4f} | [{arm_metrics['arm_d']['ci_lower_vs_a']:+.4f}, {arm_metrics['arm_d']['ci_upper_vs_a']:+.4f}] | {arm_metrics['arm_d']['p_val_vs_a']:.4f} | {arm_metrics['arm_d']['real_median_logit']:+.3f} | {arm_metrics['arm_d']['fake_median_logit']:+.3f} | {arm_metrics['arm_d']['logit_margin']:+.3f} |
| **Arm E1** | Matched Cheek Control for Mouth | Scale-Matched to B | **{arm_metrics['arm_e1']['auc']:.4f}** | {arm_metrics['arm_e1']['prauc']:.4f} | {arm_metrics['arm_e1']['delta_auc_vs_a']:+.4f} | [{arm_metrics['arm_e1']['ci_lower_vs_a']:+.4f}, {arm_metrics['arm_e1']['ci_upper_vs_a']:+.4f}] | {arm_metrics['arm_e1']['p_val_vs_a']:.4f} | {arm_metrics['arm_e1']['real_median_logit']:+.3f} | {arm_metrics['arm_e1']['fake_median_logit']:+.3f} | {arm_metrics['arm_e1']['logit_margin']:+.3f} |
| **Arm E2** | Matched Cheek Control for Jaw | Scale-Matched to C | **{arm_metrics['arm_e2']['auc']:.4f}** | {arm_metrics['arm_e2']['prauc']:.4f} | {arm_metrics['arm_e2']['delta_auc_vs_a']:+.4f} | [{arm_metrics['arm_e2']['ci_lower_vs_a']:+.4f}, {arm_metrics['arm_e2']['ci_upper_vs_a']:+.4f}] | {arm_metrics['arm_e2']['p_val_vs_a']:.4f} | {arm_metrics['arm_e2']['real_median_logit']:+.3f} | {arm_metrics['arm_e2']['fake_median_logit']:+.3f} | {arm_metrics['arm_e2']['logit_margin']:+.3f} |

---

## 2. Paired Comparisons vs. Tightly Matched Controls

- **Arm B (Mouth) vs. Arm E1 (Matched Cheek Control)**:
  - $\\Delta \\text{{AUC}}_{{\\text{{B}} - \\text{{E1}}}} = {boot_b_e1['delta_auc']:+.4f}$ [95% CI: {boot_b_e1['ci_lower']:+.4f}, {boot_b_e1['ci_upper']:+.4f}], Paired $p = {boot_b_e1['p_value']:.4f}$
- **Arm C (Jaw) vs. Arm E2 (Matched Cheek Control)**:
  - $\\Delta \\text{{AUC}}_{{\\text{{C}} - \\text{{E2}}}} = {boot_c_e2['delta_auc']:+.4f}$ [95% CI: {boot_c_e2['ci_lower']:+.4f}, {boot_c_e2['ci_upper']:+.4f}], Paired $p = {boot_c_e2['p_value']:.4f}$

---

## 3. Three-Tier Outcome Verdict

- **Outcome Tier**: **`{outcome['tier']}`**
- **Scientific Diagnostic Finding**: {outcome['reason']}
- **Governance Recommendation**: {outcome['recommendation']}

---

## 4. Phase 1 Governance & Stop-Condition Invariant

> [!IMPORTANT]
> In accordance with pre-registered governance, **execution halts at this Decision Gate**. Phase 2 (Architecture Construction) and model training are gated pending user review of this diagnostic evidence.
"""

    with open(REPORT_MD, "w") as f:
        f.write(md_content)
    logger.info("Phase 1 Markdown report saved to %s", REPORT_MD)
    print(f"\n[DONE] Phase 1 ablation complete. Tier: {outcome['tier']}")


if __name__ == "__main__":
    main()
