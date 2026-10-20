# Milestone 20.1 — Whole-Face Multi-Scale Pyramidal Representation Diagnostic Report

**Date**: 2026-10-06  
**Target Model**: Frozen M18 Dual-Stream Detector (`checkpoints/m18_dual_stream_seed42/best.pth`)  
**Dataset**: Face2Face Locked Holdout (116 clips: 58 Pristine Reals, 58 Face2Face Fakes)  
**Methodology**: Whole-face multi-scale spatial decomposition (Gaussian base + Laplacian detail octaves) evaluated across 5 diagnostic arms.

---

## 1. Multi-Scale Pyramidal Diagnostic Results

| Arm | Representation Description | Pyramid Formulation | ROC-AUC | PR-AUC | $\Delta$ AUC vs. Base | 95% Bootstrap CI | Paired $p$-value | Real Med Logit | Fake Med Logit | Logit Margin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | **Whole-Face Base Control** | Standard $G_1$ (112×112) | **0.7063** | 0.6496 | Baseline Ref | — | — | +0.453 | +2.498 | **+2.045** |
| **Arm 2** | **Mid-Frequency Warping** | Laplacian $L_1$ (112×112) | **0.5722** | 0.6073 | -0.1341 | [-0.2614, -0.0074] | 0.9860 | +1.652 | +1.683 | +0.032 |
| **Arm 3** | **Native High-Frequency** | Laplacian $L_0 \to 112$ | **0.4839** | 0.5154 | -0.2224 | [-0.3617, -0.0830] | 0.9980 | +1.017 | +0.988 | -0.029 |
| **Arm 4a**| **Detail Injection ($\alpha=0.2$)** | $G_1 + 0.20 \cdot L_0$ | **0.7301** | 0.6887 | +0.0238 | [+0.0048, +0.0443] | 0.0050 | +0.297 | +2.105 | +1.809 |
| **Arm 4b**| **Detail Injection ($\alpha=0.5$)** | $G_1 + 0.50 \cdot L_0$ | **0.7565** | 0.7451 | +0.0502 | [+0.0168, +0.0852] | 0.0020 | +0.223 | +0.996 | +0.773 |

---

## 2. Diagnostic Analysis & Findings

1. **Reconstruction Integrity**:
   - Laplacian decomposition satisfies exact mathematical reconstruction: $G_1 = \text{Upsample}(G_2) + L_1$ with float error $< 10^{-5}$.
2. **Frequency-Band Isolation**:
   - Compares whether isolated mid-frequency ($L_1$) or native high-frequency ($L_0$) detail carries the reenactment signal.
3. **High-Frequency Detail Injection**:
   - Tests whether injecting native-resolution residual details directly into the whole-face crop at $112\times 112$ bypasses the downsampling information loss without breaking the spatial filter regime.

---

## 3. Diagnostic Outcome & Governance Recommendation

- **Outcome Tier**: **`SUPPORT`**
- **Scientific Diagnostic Finding**: Statistically supported improvement observed under Arm 4a (alpha=0.20) (Delta AUC = +0.0238, p = 0.0050). Injecting native high-pass detail directly into whole-face crops improves forensic discrimination.
- **Governance Recommendation**: Proceed to design and train full Whole-Face Multi-Scale Pyramidal Network (M20.2).

---

## 4. Gating Invariant & Stop Condition

> [!IMPORTANT]
> Milestone 20.1 is strictly a diagnostic evaluation. No multi-epoch model training or architectural modification is performed without user authorization following review of these diagnostic results.
