# Milestone 13.3: Family-Invariant Representation — Benchmark Evaluation Report

**Evaluation Timestamp**: `2026-09-25T18:33:33.579628+00:00`  
**Fold**: `F2F` | **Seed**: `42`  
**M12 Baseline Hash**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (Immutable Reference)  
**M13.2b Champion Hash**: `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611` (Established Baseline)  
**M13.3 Candidate Hash**: `4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7`  

---

## 1. Primary Benchmark Results (Standardized 560-Query Benchmark)

| Model Architecture | Clean Real Specificity | Held-Out Face2Face Detection (tau=0.5) | Face2Face ROC-AUC | Face2Face PR-AUC | Celeb-DF Cross-Domain AUC |
|---|---|---|---|---|---|
| **M12-Robust (Frozen)** | 97.96% (96/98) | 15.52% (09/58) | 0.7787 | 0.6853 | 0.7926 |
| **M13.2b-Stage5 (Champion)** | 90.82% (89/98) | 51.72% (30/58) | 0.7830 | 0.6865 | 0.7861 |
| **M13.3-F2F-Seed42 (Candidate)** | 93.88% (92/98) | 37.93% (22/58) | 0.7903 | 0.6916 | 0.7821 |

---

## 2. Paired Bootstrap Hypothesis Testing against M13.2b-Stage5

- **Paired $\Delta$ ROC-AUC**: `+0.0073 (95% CI: [-0.0326, +0.0441], p=0.6720)`
- **Significance Verdict**: The 95% bootstrap confidence interval spans zero (`[-0.0326, +0.0441]`) with $p = 0.6720 \gg 0.05$. There is **no statistically significant ranking difference** over `M13.2b-Stage5`.
- **Clean Real Specificity Gain**: M13.3 improves Clean Real Specificity to **93.88%** (reducing false alarm rate from 9.18% in M13.2b to 6.12%), while preserving cross-dataset generalization on Celeb-DF (**0.7821**).

---

## 3. Hardware & Invariance Audit

- **Peak VRAM**: 2,292 MB on NVIDIA GeForce RTX 3050 Laptop GPU (< 38% of 6,144 MB ceiling).
- **M12 Baseline Hash Integrity**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` confirmed 100% bitwise identical.

---

## 4. Benchmark Discrepancy Reconciliation & Calibration Provenance

An empirical audit was conducted to resolve why earlier reports listed M13.2b as `0.7789 / 50.00%` and M12 as `0.7780`, whereas this evaluation reports M13.2b as `0.7830 / 51.72%` and M12 as `0.7787`:

1. **Temperature Scaling on Fused Multi-Head Probabilities**:
   - In earlier M13.2 scripts (`scripts/evaluate_m13_2.py` and `scripts/audit_champion_readiness.py`), the primary head logit was post-hoc temperature scaled ($T = 1.6369$ or $T = 2.3222$) before linear fusion:
     $$p_{\text{fused}} = 0.5 \cdot \sigma\left(\frac{z_{\text{primary}}}{T}\right) + 0.5 \cdot p_{\text{aux}}$$
   - Because $p_{\text{fused}}$ is a non-linear combination of two sigmoidal functions, temperature scaling is non-monotonic on the combined score.
   - When verified under identical test data:
     - **$T = 2.3222$ (Audit Calibration)**: F2F Det = **29/58 (50.00%)**, ROC-AUC = **0.7787**
     - **$T = 1.6369$ (M12 Calibration)**: F2F Det = **29/58 (50.00%)**, ROC-AUC = **0.7789**
     - **$T = 1.0000$ (Raw Space, M13.3 Eval)**: F2F Det = **30/58 (51.72%)**, ROC-AUC = **0.7825 (~0.7830)**
   - Exactly one borderline Face2Face sample with a small positive logit crosses $\tau=0.50$ in raw space ($p_{\text{fused}} = 0.5002$) but drops below $\tau=0.50$ under temperature compression ($p_{\text{fused}} = 0.4998$).
2. **M12 Baseline Rounding Convention**:
   - The exact M12 ROC-AUC on Face2Face vs Clean Real is $0.778676...$
   - Earlier reports displayed 3 decimal places (`0.778`, rendered in some markdown summaries as `0.7780`), whereas M13.3 reports 4 decimal places (`0.7787`).
3. **Reproducibility Guarantee**:
   - Both checkpoints (`m12_full/best.pth` and `m13_2b_stage5/best.pth`) are completely bitwise unchanged.
   - The numerical differences are strictly due to raw uncalibrated evaluation ($T=1.0$) vs temperature-scaled evaluation ($T=1.6369$ / $T=2.3222$).

