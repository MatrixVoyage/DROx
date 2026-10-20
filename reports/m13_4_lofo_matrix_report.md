# Milestone 13.4 Phase E: 4-Fold LOFO Cross-Family Generalization Report

**Timestamp**: `2026-09-25T20:20:25.710079+00:00`  
**M12 Baseline Hash**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (Immutable Reference)  
**M13.2b Champion Hash**: `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611` (Principal Comparison Baseline)  

---

## 1. The 4-Fold Leave-One-Family-Out (LOFO) Matrix

| Fold | Held-Out Family | Seen Training Families | M12 Baseline (AUC / Det Rate) | M13.2b Champion (AUC / Det Rate) | M13.4 Candidate (AUC / Det Rate) | Candidate vs M13.2b Delta AUC |
|---|---|---|---|---|---|---|
| **LOFO-F2F** | **Face2Face** | Deepfakes + FaceSwap + NeuralTextures | 0.7787 (15.5%) | 0.7830 (51.7%) | 0.7903 (37.9%) | +0.0073 |
| **LOFO-NT** | **NeuralTextures** | Deepfakes + FaceSwap + Face2Face | 0.9298 (60.3%) | 0.9244 (84.5%) | 0.9499 (89.7%) | +0.0255 |
| **LOFO-FS** | **FaceSwap** | Deepfakes + NeuralTextures + Face2Face | 0.9402 (56.9%) | 0.9474 (91.4%) | 0.9435 (89.7%) | -0.0039 |
| **LOFO-DF** | **Deepfakes** | FaceSwap + NeuralTextures + Face2Face | 0.9863 (81.0%) | 0.9806 (98.3%) | 0.9824 (100.0%) | +0.0018 |
| **Macro Mean** | **All 4 Families** | Cross-Family Generalization | **0.9087** | **0.9089** | **0.9165** | **+0.0077** |

---

## 2. Milestone 13.4 Required Decision Categories

1. **Seed Stability**: **STABLE**
   - Cross-seed ROC-AUC across Seeds 42, 1337, 2026 is `0.7918 ± 0.0025` (Standard Error: `±0.0015`).
   - Clean Real Specificity: `91.84% ± 2.04%` (Standard Error: `±1.18%`).

2. **Operating-Point Validity**: **VALIDATED**
   - Validation-derived threshold policy (`Spec >= 90%`, `tau = 0.0200`) recovers **43.10% (25/58)** of Face2Face samples on the untouched test benchmark with **93.88% (92/98)** Clean Specificity without test-set leakage.

3. **Calibration**: **CALIBRATED**
   - Expected Calibration Error (ECE) reduced from `0.2729` (raw) to `0.0866` (temperature scaling, $T=2.0502$) and `0.0619` (isotonic calibration). Brier score reduced from `0.2683` to `0.1848`.

4. **Cross-Family Transfer**: **DEMONSTRATED**
   - Evaluation across the 4-fold matrix (LOFO-F2F, LOFO-NT, LOFO-FS, LOFO-DF) tests whether the multi-domain representation generalizes beyond Face2Face. Macro mean AUC: `0.9165` vs M13.2b `0.9089`.
