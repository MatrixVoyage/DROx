# Milestone 18 (M18) — Targeted Reenactment-Specific Representation Report

**Date**: 2026-10-05  
**Final Governance Verdict**: **`RETAIN_M13_5_CHAMPION`**  
**Candidate**: M18 Dual-Stream Reenactment Detector (`checkpoints/m18_dual_stream_seed42/best.pth`)  
**Frozen Baseline Controls**: M17 Track A (S1+S2 Trunk) & M13.5 Champion Control (`v5`)  

---

## 1. Five Strict Promotion Gates Summary

| Gate | Evaluated Benchmark | Target Standard | Candidate Result | Gate Status |
| :--- | :--- | :---: | :---: | :---: |
| **Gate 1: Reenactment Recovery** | Face2Face Locked Holdout | AUC $\ge 0.8000$, $p < 0.05$ | **0.7727** ($p=0.1980$) | **[FAIL]** |
| **Gate 2: Cross-Domain Retention** | Celeb-DF-v2 Holdout | Non-inferior ($\Delta \ge -0.015$) | **0.8352** ($\Delta = -0.0042$) | **[PASS]** |
| **Gate 3: Clean Specificity** | FF++ & Celeb-DF Reals | $\ge 95.0\%$ on both | FF++: 1.7%, Celeb: 2.3% | **[FAIL]** |
| **Gate 4: Compression Robustness** | JPEG Q=50 & H.264 CRF 23 | $\ge 0.650$ / $\ge 0.600$ | Q50: **0.7572**, H.264: **0.5947** | **[FAIL]** |
| **Gate 5: Production Envelope** | Latency & VRAM (RTX 3050) | P50 $\le 35$ ms, $\le 5.5$ GB | P50: 19.7 ms, VRAM: 0.82 GB | **[PASS]** |

---

## 2. 3-Way Paired Benchmark Matrix

| Benchmark | M18 Dual-Stream | M18 Kinematic-Only | M17 Track A (S1+S2) | M13.5 Champion (v5) | $\Delta$ Dual vs Track A | $\Delta$ Dual vs v5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gate1_Face2Face** | **0.7727** | 0.5110 | 0.7426 | 0.7946 | +0.0295 ($p=0.1980$) | -0.0211 |
| **Gate2_CelebDF** | **0.8352** | 0.6016 | 0.8384 | 0.7806 | -0.0042 ($p=0.6350$) | +0.0529 |
| **Gate4_JPEG_Q50** | **0.7572** | 0.5447 | 0.7320 | 0.7428 | +0.0255 ($p=0.0010$) | +0.0142 |
| **Gate4_H264_CRF23** | **0.5947** | 0.6065 | 0.5949 | 0.5984 | -0.0027 ($p=0.5340$) | -0.0058 |
| **FFpp_Mixed** | **0.8617** | 0.5325 | 0.8510 | 0.8344 | +0.0106 ($p=0.1480$) | +0.0271 |

---

## 3. Governance Audit & Final Decision

- **Promotion Verdict**: `RETAIN_M13_5_CHAMPION`
- **Active Production Champion**: Retained M13.5 Champion (`checkpoints/m13_3_f2f_seed42/best.pth`)
- **Research Status**: M18 retained as a valuable research milestone; production invariants preserved.
