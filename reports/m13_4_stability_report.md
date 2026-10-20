# Milestone 13.4: Calibration, Threshold & Multi-Seed Stability Report

**Evaluation Timestamp**: `2026-09-25T19:07:41.917788+00:00`  
**M12 Baseline Hash**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (Immutable Reference)  
**M13.2b Champion Hash**: `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611` (Principal Comparison Baseline)  

---

## 1. Phase A: Multi-Seed Stability (LOFO-F2F Benchmark)

| Model / Seed | Clean Specificity | Face2Face Det (tau=0.5) | Face2Face ROC-AUC | Face2Face PR-AUC | Celeb-DF Cross-Domain AUC |
|---|---|---|---|---|---|
| **M12-Robust (Frozen)** | 97.96% (96/98) | 15.52% (09/58) | 0.7787 | 0.6853 | 0.7926 |
| **M13.2b-Stage5 (Champion)** | 90.82% (89/98) | 51.72% (30/58) | 0.7830 | 0.6865 | 0.7861 |
| **M13.3-F2F-Seed42** | 93.88% (92/98) | 37.93% (22/58) | 0.7903 | 0.6916 | 0.7821 |
| **M13.3-F2F-Seed1337** | 91.84% (90/98) | 43.10% (25/58) | 0.7947 | 0.7168 | 0.8054 |
| **M13.3-F2F-Seed2026** | 89.80% (88/98) | 48.28% (28/58) | 0.7903 | 0.6419 | 0.8019 |

### Cross-Seed Summary (N=3 Seeds):
- **Clean Specificity**: `91.84% ± 2.04%` (SE: `1.18%`)
- **Face2Face Det Rate**: `43.10% ± 5.17%` (SE: `2.99%`)
- **Face2Face ROC-AUC**: `0.7918 ± 0.0025` (SE: `0.0015`)
- **Face2Face PR-AUC**: `0.6834 ± 0.0381` (SE: `0.0220`)
- **Celeb-DF AUC**: `0.7965 ± 0.0126` (SE: `0.0073`)

---

## 2. Phase B: Validation-Constrained Threshold Policy

| Threshold Policy | Operating Threshold (tau) | Test Clean Specificity | Test Face2Face Detection |
|---|---|---|---|
| **Standard (tau=0.50)** | `tau = 0.5000` | 93.88% (False Alarms: 06) | 37.93% (22/58) |
| **Spec-Constrained >= 90%** | `tau = 0.0200` | 93.88% (False Alarms: 06) | 43.10% (25/58) |
| **Spec-Constrained >= 95%** | `tau = 0.8400` | 93.88% (False Alarms: 06) | 31.03% (18/58) |
| **Validation Youden J** | `tau = 0.0487` | 93.88% (False Alarms: 06) | 41.38% (24/58) |

---

## 3. Phase C: Calibration Diagnostics & Probability Distributions

- **Fitted Validation Temperature**: `T = 2.0502`
- **Raw Test ECE / Brier**: `0.2729` / `0.2683`
- **Temperature Scaled ECE / Brier**: `0.0866` / `0.1848`
- **Isotonic Calibrated ECE / Brier**: `0.0619` / `0.1777`

### Score Distributions (Percentiles: p10, p50 median, p90):
- **Clean Real**: Median=`0.0003`, Mean=`0.0613`, IQR=`[0.0003, 0.0004]` (p10=`0.0003`, p90=`0.0010`)
- **Seen Fake**: Median=`0.9995`, Mean=`0.8389`, IQR=`[0.9962, 0.9997]` (p10=`0.0038`, p90=`0.9997`)
- **Held Out F2F**: Median=`0.0039`, Mean=`0.3588`, IQR=`[0.0004, 0.9669]` (p10=`0.0003`, p90=`0.9994`)

---

## 4. Phase D: Systematic 7-Way Branch Ablation

| Branch Permutation | Test Clean Specificity | Face2Face Det (tau=0.5) | Face2Face ROC-AUC | Face2Face PR-AUC |
|---|---|---|---|---|
| **1. Temporal Only** | 93.88% | 37.93% (22/58) | 0.7903 | 0.6916 |
| **2. Frequency Only** | 71.43% | 58.62% (34/58) | 0.7057 | 0.5200 |
| **3. Texture Only** | 53.06% | 60.34% (35/58) | 0.5873 | 0.4134 |
| **4. Temporal + Frequency** | 93.88% | 37.93% (22/58) | 0.7718 | 0.6806 |
| **5. Temporal + Texture** | 93.88% | 37.93% (22/58) | 0.6935 | 0.6107 |
| **6. Frequency + Texture** | 65.31% | 58.62% (34/58) | 0.6810 | 0.5086 |
| **7. Full Multi-Domain Fusion** | 93.88% | 37.93% (22/58) | 0.7903 | 0.6916 |
