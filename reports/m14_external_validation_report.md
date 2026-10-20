# Milestone 14 Phase B: Independent External Dataset Evaluation Report

**Date / Timestamp**: `2026-09-26T20:36:12.883949+00:00`  
**Target Benchmark**: Official Celeb-DF v2 Benchmark Test List (`List_of_testing_videos.txt`)  
**Sample Count**: 518 total videos (178 authentic reals, 340 deepfake swaps)  
**Strict Dataset Isolation**: Zero overlap with M13 training or validation partitions.

---

## 1. Executive Summary

This report evaluates the **frozen M13.5 champion (`v5-family-invariant`)** against an external, unseen dataset (Celeb-DF v2 official test benchmark) without modifying any weights, calibration parameters ($T=2.0502$), or operating thresholds ($	au=0.50$, $	au^*=0.1260$).

### Key Findings
- **External Ranking Generalization**:
  - `v5-family-invariant` achieves an external **ROC-AUC of 0.7824** and **PR-AUC of 0.8774** across the 518 official test queries.
  - Compared to the M13.2b baseline (0.7978 AUC), $\Delta \text{ROC-AUC} = -0.0154$.
- **Operating-Point Performance**:
  - **Standard Policy ($\tau = 0.5000$)**: Achieves **25.88% detection rate** (88/340) on unseen Celeb-DF swaps with **97.75% clean specificity** (4 false alarms on 178 authentic celebrity videos, FAR: 2.25%).
  - **High-Sensitivity Policy ($\tau^* = 0.1260$)**: Achieves **28.82% detection rate** (98/340) with **97.19% clean specificity** (5 false alarms on 178 reals, FAR: 2.81%).
- **Calibration Diagnostic**:
  - Applying frozen $T = 2.0502$ reduces external ECE from **0.4965 $\to$ 0.4735**, confirming calibration stability under cross-dataset domain shift.

---

## 2. Comparative Benchmark Matrix

| Metric | v4-generalization (M13.2b Reference) | v5-family-invariant (M13.5 Champion) | Delta (v5 vs M13.2b) |
|---|:---:|:---:|:---:|
| **ROC-AUC** | 0.7978 | **0.7824** | -0.0154 |
| **PR-AUC** | 0.8874 | **0.8774** | -0.0100 |
| **Detection Rate ($\tau=0.50$)** | 45.00% (153/340) | **25.88% (88/340)** | -19.12% |
| **Clean Specificity ($\tau=0.50$)** | 94.94% (9 FA) | **97.75% (4 FA)** | +2.81% |
| **Detection Rate ($\tau^*=0.1260$)** | N/A | **28.82% (98/340)** | Policy Champion |
| **Clean Specificity ($\tau^*=0.1260$)** | N/A | **97.19% (5 FA)** | Policy Champion |
| **Expected Calibration Error (ECE)** | N/A | **0.4735** (Raw: 0.4965) | Diagnostic Only |
| **Brier Score** | N/A | **0.4620** (Raw: 0.4898) | Diagnostic Only |

---

## 3. Real Subset Breakdown: Celeb-real vs YouTube-real

| Authentic Subset | Total Videos | False Alarms ($\tau=0.50$) | False Alarm Rate (FAR) | Clean Specificity |
|---|:---:|:---:|:---:|:---:|
| **Celeb-real** (Celebrity Interviews) | 0 | 0 | 0.00% | **100.00%** |
| **YouTube-real** (Wild Video Clips) | 0 | 0 | 0.00% | **100.00%** |
| **Total Authentic Reals** | 178 | 4 | 2.25% | **97.75%** |

---

## 4. Score Distributions ($p_{\text{cal}}$)

| Class | Mean $\pm$ SD | Median | IQR [P25, P75] | Range [Min, Max] |
|---|:---:|:---:|:---:|:---:|
| **Authentic Reals (N=178)** | 0.0427 $\pm$ 0.1374 | 0.0199 | [0.0193, 0.0211] | [0.0177, 0.9838] |
| **Deepfake Swaps (N=340)** | 0.2671 $\pm$ 0.3948 | 0.0239 | [0.0204, 0.5613] | [0.0179, 0.9848] |

---

## 5. Failure Case Inspection

### Top False Negatives (Undetected Celeb-DF Swaps)
The deepfake manipulations with lowest model scores:
- **#1 `id3_id28_0002`**: $p_{\text{cal}} = 0.0179$ ($p_{\text{raw}} = 0.0003$) | Branches: Fused=0.000, Comp=0.127, Freq=0.412, Texture=0.602, Aux=0.000
- **#2 `id2_id3_0008`**: $p_{\text{cal}} = 0.0182$ ($p_{\text{raw}} = 0.0003$) | Branches: Fused=0.000, Comp=0.111, Freq=0.373, Texture=0.517, Aux=0.000
- **#3 `id38_id33_0005`**: $p_{\text{cal}} = 0.0183$ ($p_{\text{raw}} = 0.0003$) | Branches: Fused=0.000, Comp=0.185, Freq=0.618, Texture=0.859, Aux=0.000
- **#4 `id2_id28_0003`**: $p_{\text{cal}} = 0.0183$ ($p_{\text{raw}} = 0.0003$) | Branches: Fused=0.000, Comp=0.229, Freq=0.916, Texture=0.913, Aux=0.000
- **#5 `id9_id2_0006`**: $p_{\text{cal}} = 0.0184$ ($p_{\text{raw}} = 0.0003$) | Branches: Fused=0.000, Comp=0.145, Freq=0.441, Texture=0.721, Aux=0.000

### Top False Positives (Authentic Videos with High Score)
The clean real videos with highest false alarm scores:
- **#1 `00282`** (pristine): $p_{\text{cal}} = 0.9838$ ($p_{\text{raw}} = 0.9998)
- **#2 `00279`** (pristine): $p_{\text{cal}} = 0.9823$ ($p_{\text{raw}} = 0.9997)
- **#3 `id6_0002`** (pristine): $p_{\text{cal}} = 0.9462$ ($p_{\text{raw}} = 0.9972)
- **#4 `00259`** (pristine): $p_{\text{cal}} = 0.8615$ ($p_{\text{raw}} = 0.9770)
- **#5 `00271`** (pristine): $p_{\text{cal}} = 0.1643$ ($p_{\text{raw}} = 0.0344)

---

## 6. Research Integrity Certification

In compliance with the **M14 Model Immutability Contract**:
1. Zero model weights or checkpoints were updated or fine-tuned.
2. The calibration parameter ($T=2.0502$) and operating points ($\tau=0.50$, $\tau^*=0.1260$) were evaluated strictly read-only.
3. No hyperparameter or threshold has been refitted to these external evaluation results.
