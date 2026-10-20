# Milestone 13.2c: Identity/Manipulation Disentanglement — Authoritative Scientific Report

**M12 Baseline SHA-256**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (Bitwise Frozen Immutable Baseline)  
**Experimental Baseline**: `M13.2b-Stage5` (Stage-5 Spatial Head + Penultimate Multi-Family Head)  
**Strict Invariants**: Zero test/Face2Face data leakage. Preprocessing and evaluation protocol identical across all models.  

---

## 1. Primary Research Question & Verdict

> [!IMPORTANT]
> **Primary Scientific Question**: Does explicitly separating identity-related and manipulation-related representations produce genuine ranking-level improvements for held-out Face2Face beyond M13.2b-Stage5, or does it merely shift classification thresholds / probability calibration?

### Formal Hypothesis Evaluation
- **Verdict**: **DISENTANGLEMENT HYPOTHESIS REJECTED / INSUFFICIENT EVIDENCE**. Face2Face ROC-AUC delta: -0.0066 (95% CI: [-0.0601, 0.0417], p-val=0.79). Disentanglement does not produce a statistically significant ranking improvement over M13.2b-Stage5.

---

## 2. Full Comparative Matrix (Fixed 560-Query Benchmark)

| Benchmark Metric | Frozen M12 Baseline | M13.2a-Joint | M13.2b-Stage5 (Baseline) | M13.2c-TrueGroup | M13.2c-ShuffledGroup (Neg Ctrl) | M13.2c-OrthOnly (Ablation) |
|---|---|---|---|---|---|---|
| **Held-Out Face2Face ROC-AUC** | 0.7780 | 0.7920 | 0.7789 | 0.7723 | 0.7797 | 0.7674 |
| **Held-Out Face2Face PR-AUC** | 0.6845 | 0.7063 | 0.6821 | 0.7088 | 0.7124 | 0.7128 |
| **Face2Face Detection Rate (τ=0.5)** | 15.52% | 41.38% | 50.00% | 53.45% | 53.45% | 44.83% |
| **Face2Face Optimal F1 (Youden J)** | 0.6606 | 0.6972 | 0.6667 | 0.6724 | 0.6903 | 0.6842 |
| **Clean Real Specificity (τ=0.5)** | 97.96% | 92.86% | 90.82% | 84.69% | 89.80% | 90.82% |
| **Clean Real False Alarm Rate (FRR)** | 2.04% | 7.14% | 9.18% | 15.31% | 10.20% | 9.18% |
| **In-Domain Seen Fake ROC-AUC** | 0.9519 | 0.9583 | 0.9493 | 0.9523 | 0.9483 | 0.9500 |
| **In-Domain Seen Fake F1** | 0.7904 | 0.9129 | 0.9298 | 0.9352 | 0.9425 | 0.9298 |
| **Celeb-DF Cross-Dataset AUC** | 0.7290 | 0.7140 | 0.7599 | 0.6731 | 0.7005 | 0.6878 |
| **Celeb-DF Cross-Dataset F1** | 0.3830 | 0.5138 | 0.5045 | 0.4248 | 0.4727 | 0.4340 |
| **Degraded (JPEG Q=40) AUC** | 0.5044 | 0.5573 | 0.6360 | 0.6618 | 0.6310 | 0.6526 |
| **Expected Calibration Error (ECE)** | 0.3314 | 0.2107 | 0.1825 | 0.1147 | 0.1185 | 0.1261 |
| **Brier Score (Calibration)** | 0.3233 | 0.2059 | 0.1753 | 0.1468 | 0.1418 | 0.1457 |

---

## 3. Paired Bootstrap Hypothesis Testing (Against M13.2b-Stage5)

| Candidate Configuration | Δ ROC-AUC [95% Bootstrap CI] | p-value | Δ PR-AUC [95% Bootstrap CI] | p-value | Statistical Significance |
|---|---|---|---|---|---|
| **M13.2c-TrueGroup** | -0.0074 [-0.0601, +0.0417] | 0.7900 | +0.0218 [-0.0443, +0.0939] | 0.5920 | NO (p >= 0.05) |
| **M13.2c-ShuffledGroup** | -0.0000 [-0.0514, +0.0503] | 0.9820 | +0.0254 [-0.0386, +0.0943] | 0.5020 | NO (p >= 0.05) |
| **M13.2c-OrthOnly** | -0.0124 [-0.0708, +0.0416] | 0.6920 | +0.0248 [-0.0366, +0.0885] | 0.4700 | NO (p >= 0.05) |
| **M13.2a-Joint** | +0.0134 [-0.0101, +0.0398] | 0.2580 | +0.0217 [-0.0026, +0.0528] | 0.1060 | NO (p >= 0.05) |

---

## 4. Disentanglement Subspace Probing Analysis

| Disentangled Model | z_id Same-Group Sim | z_id Diff-Group Sim | z_id ΔSim (Identity Signal) | z_manip Same-Group Sim | z_manip Diff-Group Sim | z_manip ΔSim (Residual Identity) |
|---|---|---|---|---|---|---|
| **M13.2c-TrueGroup** | 0.7447 | 0.5627 | **+0.1821** | 0.5663 | 0.0891 | +0.4772 |
| **M13.2c-ShuffledGroup** | 0.9619 | 0.9580 | **+0.0039** | 0.5406 | 0.0789 | +0.4617 |
| **M13.2c-OrthOnly** | 0.7205 | 0.5006 | **+0.2199** | 0.4702 | 0.0587 | +0.4115 |

---

## 5. Computational and Resource Overhead

| Model | Total Parameters | Overhead vs M12 | P50 Latency (RTX 3050) | P95 Latency | Peak VRAM |
|---|---|---|---|---|---|
| **Frozen M12 Baseline** | 28,289,150 | +0.00% | 23.55 ms | 41.24 ms | 486.79 MB |
| **M13.2a-Joint** | 28,948,611 | +2.33% | 18.9 ms | 20.46 ms | 933.82 MB |
| **M13.2b-Stage5** | 28,963,460 | +2.38% | 20.06 ms | 21.51 ms | 1156.55 MB |
| **M13.2c-TrueGroup** | 29,749,893 | +5.16% | 21.36 ms | 23.39 ms | 1386.55 MB |
| **M13.2c-ShuffledGroup** | 29,749,893 | +5.16% | 21.38 ms | 23.29 ms | 1616.54 MB |
| **M13.2c-OrthOnly** | 29,749,893 | +5.16% | 21.4 ms | 23.13 ms | 1846.54 MB |
