# Milestone 20 — Phase 1: Frozen-Weight Localized Representation Transfer Ablation

**Date**: 2026-10-06  
**Target Model**: Frozen M18 Dual-Stream Detector (`checkpoints/m18_dual_stream_seed42/best.pth`)  
**Dataset**: Face2Face Locked Holdout (116 clips: 58 Pristine Reals, 58 Face2Face Fakes)  
**Methodology**: Frozen-weight transfer diagnostic across 6 scale-matched arms. Evaluates whether localized native-resolution crops provide forensic cues on the frozen M18 encoder.

---

## 1. Six-Arm Empirical Results

| Arm | Representation Description | Crop Scaling | ROC-AUC | PR-AUC | $\Delta$ AUC vs. Arm A | 95% Bootstrap CI | Paired $p$-value | Real Med Logit | Fake Med Logit | Logit Margin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** | Whole-Face Baseline (Control) | Whole MTCNN BBox | **0.7669** | 0.7291 | Baseline Ref | — | — | +0.225 | +0.604 | **+0.380** |
| **Arm B** | Native-Resolution Mouth | 1.4× Mouth Width | **0.5707** | 0.5799 | -0.1962 | [-0.3213, -0.0632] | 1.0000 | +1.152 | +1.423 | +0.270 |
| **Arm C** | Native-Resolution Jawline | 1.6× Mouth Width | **0.5125** | 0.5068 | -0.2545 | [-0.3762, -0.1393] | 1.0000 | +1.493 | +1.197 | -0.296 |
| **Arm D** | Native-Resolution Mouth+Jaw | 2.0× Mouth Width | **0.5862** | 0.5806 | -0.1807 | [-0.2927, -0.0596] | 0.9930 | +0.672 | +0.931 | +0.259 |
| **Arm E1** | Matched Cheek Control for Mouth | Scale-Matched to B | **0.6391** | 0.6544 | -0.1278 | [-0.2501, -0.0068] | 0.9810 | +1.201 | +1.535 | +0.334 |
| **Arm E2** | Matched Cheek Control for Jaw | Scale-Matched to C | **0.6049** | 0.6129 | -0.1620 | [-0.2811, -0.0419] | 0.9970 | +1.517 | +1.870 | +0.352 |

---

## 2. Paired Comparisons vs. Tightly Matched Controls

- **Arm B (Mouth) vs. Arm E1 (Matched Cheek Control)**:
  - $\Delta \text{AUC}_{\text{B} - \text{E1}} = -0.0684$ [95% CI: -0.1921, +0.0609], Paired $p = 0.8690$
- **Arm C (Jaw) vs. Arm E2 (Matched Cheek Control)**:
  - $\Delta \text{AUC}_{\text{C} - \text{E2}} = -0.0924$ [95% CI: -0.2129, +0.0368], Paired $p = 0.9180$

---

## 3. Three-Tier Outcome Verdict

- **Outcome Tier**: **`NEGATIVE EVIDENCE`**
- **Scientific Diagnostic Finding**: Localized arms did not improve over Arm A, and matched controls performed equivalently or better. No evidence that localized native-resolution crops provide superior forensic discrimination on this pipeline.
- **Governance Recommendation**: Archive localized native-resolution hypothesis and pivot research direction.

---

## 4. Phase 1 Governance & Stop-Condition Invariant

> [!IMPORTANT]
> In accordance with pre-registered governance, **execution halts at this Decision Gate**. Phase 2 (Architecture Construction) and model training are gated pending user review of this diagnostic evidence.
