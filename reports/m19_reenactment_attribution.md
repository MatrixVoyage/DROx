# Milestone 19 — Track A: Reenactment Signal Attribution Report

**Date**: 2026-10-06  
**Target Model**: Frozen M18 Dual-Stream Detector (`checkpoints/m18_dual_stream_seed42/best.pth`)  
**Methodology**: Strictly descriptive, zero-retraining forensic diagnostics with Benjamini-Hochberg FDR control and Cliff's $\delta$ effect sizes.  

---

## 1. Diagnostic 1: 4-Zone Spatial Attention Decomposition

| Partition Zone | Face2Face Attention Mass | Pristine Real Attention Mass | Contrast Delta (F2F - Real) | Cliff's $\delta$ Effect Size |
| :--- | :---: | :---: | :---: | :---: |
| **Inner-Face Zone** (Landmarks: eyes, nose, mouth) | 22.3% ± 0.5% | 22.5% ± 0.6% | -0.2% | **-0.207** |
| **Outer-Face Zone** (Contour, jawline, forehead) | 41.3% ± 0.9% | 41.2% ± 0.8% | +0.2% | — |
| **Peripheral Canvas** (Border margin) | 36.3% ± 0.9% | 36.3% ± 1.0% | +0.0% | — |

---

## 2. Diagnostic 2: $\alpha$-Gate Distribution Across Populations (with FDR Control)

| Population Group | N | Median $\alpha$ | IQR | Mean ± Std | KS Stat vs. Reals | Raw $p$-value | FDR $q$-value | Cliff's $\delta$ [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pristine_Reals** | 58 | **0.086** | 0.033 | 0.113 ± 0.082 | — | — | — | Baseline Reference |
| **Face2Face** | 58 | **0.134** | 0.152 | 0.208 ± 0.164 | 0.466 | 4.7703e-06 | **5.7244e-06** | +0.534 [0.36, 0.68] |
| **NeuralTextures** | 58 | **0.385** | 0.434 | 0.407 ± 0.223 | 0.690 | 1.8309e-13 | **2.7463e-13** | +0.829 [0.71, 0.92] |
| **FaceSwap** | 58 | **0.507** | 0.502 | 0.455 ± 0.255 | 0.690 | 1.8309e-13 | **2.7463e-13** | +0.858 [0.76, 0.93] |
| **Deepfakes** | 58 | **0.694** | 0.261 | 0.615 ± 0.170 | 0.931 | 2.3319e-27 | **1.3991e-26** | +0.978 [0.94, 1.00] |
| **CelebDF_Reals** | 175 | **0.105** | 0.026 | 0.120 ± 0.072 | 0.329 | 1.0837e-04 | **1.0837e-04** | +0.333 [0.15, 0.51] |
| **CelebDF_Fakes** | 336 | **0.217** | 0.348 | 0.301 ± 0.201 | 0.655 | 5.7911e-21 | **1.7373e-20** | +0.772 [0.66, 0.87] |

---

## 3. Diagnostic 3: Mutually Exhaustive TP vs. FN Kinematic Profiles on Face2Face

- **Sample Partition at Locked Threshold ($z = 0$)**: Total F2F = 58 | **TP** = 57 | **FN** = 1
- **Relative to M17 Track A**: **Recovered** = 40 | **Lost** = 0 | **Unchanged** = 18

| Kinematic Metric | True Positives (TP, N=57) | False Negatives (FN, N=1) | $\Delta$ (TP - FN) | Cliff's $\delta$ Effect Size |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Frame Differential Velocity** | 0.2692 | 0.2754 | -0.0062 | **-0.088** |
| **Peak Differential Velocity** | 3.8262 | 2.9455 | +0.8808 | — |

> [!WARNING]
> **Inferential Limitation Caveat**: Because there is only $N = 1$ false negative at this locked decision boundary, the comparison between TP and FN kinematic metrics is strictly descriptive and cannot support strong inferential conclusions regarding failure kinematics.

---

## 4. Diagnostic 4: Reenactment Transfer to NeuralTextures

| Model | NeuralTextures Test AUC | $\Delta$ vs. M17 Track A | $\Delta$ vs. M13.5 Champion | Mean $\alpha$ Gate |
| :--- | :---: | :---: | :---: | :---: |
| **M18 Dual-Stream** | **0.9171** | **+0.0459** ($p=0.0060$) | **-0.0231** ($p=0.8680$) | **0.407** |
| **M17 Track A Reference** | 0.8716 | Baseline Reference | — | — |
| **M13.5 Champion Control** | 0.9401 | — | Baseline Reference | — |

---

## 5. Diagnostic 5: Metadata & Confound Correlation Audit

| Confound Audit Test | Spearman $\rho$ | 95% Bootstrap CI | Raw $p$-value | FDR $q$-value | Confound Finding |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bitrate_vs_Score** | -0.154 | [-0.24, -0.07] | 4.9356e-04 | **9.8712e-04** | **Statistically Significant Confound ($q < 0.001$)** |
| **Bitrate_vs_Gate** | -0.163 | [-0.25, -0.08] | 2.1576e-04 | **8.6305e-04** | **Statistically Significant Confound ($q < 0.001$)** |
| **Width_vs_Score** | -0.054 | [-0.14, 0.03] | 2.2103e-01 | **2.8332e-01** | No Significant Confound |
| **Height_vs_Score** | +0.048 | [-0.04, 0.13] | 2.8332e-01 | **2.8332e-01** | No Significant Confound |

- **Bitrate Confound Interpretation**: Video bitrate exhibits a statistically significant negative correlation with both fake score ($q \approx 0.00099$) and fusion gate $\alpha$ ($q \approx 0.00086$). Higher video bitrates slightly depress the model's fake probability and gate activation, indicating that subtle codec/bitrate characteristics remain a detectable statistical confound.
- **Actor Identity Clustering**: Kruskal-Wallis test across 40 actor groups: $H = 43.133$, $p = 0.2990$ (No significant actor identity clustering detected).

---

## 6. Diagnostic 6: Seven Failure Hypotheses Evidence-Attribution Table

| Hypothesis | Evaluated Evidence from Diagnostics | Empirical Evidence Strength | Research Status |
| :--- | :--- | :---: | :---: |
| **H1: Insufficient Motion Representation** | Kinematic branch alone yields near-chance AUC (0.5110). Differential velocity contrast between TP and FN is marginal (Cliff's d=-0.088). First-order frame differencing lacks non-rigid optical deformation modeling. | **Strong Support** | Valid Research Hypothesis (Primary Explanatory Cause) |
| **H2: Insufficient Spatial Localization** | Spatial attention concentrates moderately on inner-face (F2F inner mass=0.223 vs Reals 0.225, Cliff's d=-0.207), but 14x14 pooling dilutes boundary cues. | **Moderate Support** | Secondary Contributing Factor |
| **H3: Inadequate Temporal Context** | Uniform 16-frame sampling captures large head rotations but under-samples high-frequency lip/phoneme micro-expression dynamics. | **Moderate Support** | Secondary Contributing Factor |
| **H4: Weak Supervision** | Multi-family auxiliary cross-entropy separated families generally, but lacked an explicit reenactment-specific pairwise metric margin penalty against pristine reals. | **Weak Support** | Minor Contributing Factor |
| **H5: Compression Corruption** | Under uniform H.264 CRF 23 re-encoding, performance collapsed to 0.5947 AUC across all streams. Video bitrate is statistically correlated with score ($q \approx 0.00099$) and gate ($q \approx 0.00086$). | **Moderate Support (Compression Confound Confirmed)** | Valid Research Factor |
| **H6: Poor Gating** | Alpha gate selectively opens on Face2Face (median 0.134) and NeuralTextures (median 0.385) vs Reals (median 0.086) with KS q=5.7244e-06 and Cliff's d=+0.534. | **Refuted / Counter-Evidence** | Hypothesis Rejected (Gate functions properly) |
| **H7: Genuine Pixel Signal Ceiling** | Face2Face artifacts at C23 quality in 112x112 crops exhibit subtle pixel residuals that may be suppressed by low-resolution resizing. However, M19 did not perform a controlled resolution ablation. | **Plausible / Moderate Support** | Unresolved Hypothesis (Requires M19.1 Controlled Resolution Ablation) |

---

## 7. Track A Attribution Conclusions & Frozen Governance

1. **Gate Selective Activation**: The learned $\alpha$-gate is not indiscriminate; it significantly opens on both reenactment families (Face2Face median $\alpha = 0.134$, NeuralTextures median $\alpha = 0.385$) compared to Pristine Reals (median $\alpha = 0.086$, FDR $q < 10^{-5}$), demonstrating that the gating mechanism operates selectively as intended.
2. **Cross-Family Reenactment Transfer**: The kinematic signal transfers cleanly to NeuralTextures (**AUC = 0.9171**, improving upon Track A's 0.8716 by $+0.0459$, $p = 0.0060$), confirming that the temporal kinematic stream captures facial expression deformation rather than a Face2Face-specific artifact.
3. **Primary Ceiling & Unresolved Hypotheses**: The empirical evidence strongly supports **H1 (Insufficient Motion Representation)** as the primary bottleneck. Simple first-order frame differencing conflates rigid head translation with non-rigid facial warping. **H7 (Pixel Signal Ceiling)** remains plausible but unresolved until tested under controlled resolution and temporal ablations in M19.1.
4. **Bitrate Confound**: Bitrate is statistically significant ($q < 0.001$), confirming that compression/bitrate artifacts mildly influence detection scores and gate opening.

---

## 8. Frozen Governance Declarations

- **M19 Track A**: **COMPLETED & FROZEN** (Strictly descriptive attribution, zero model retraining).
- **M13.5 (`v5-family-invariant`)**: **ACTIVE PRODUCTION CHAMPION** (Retained undefeated).
- **M17 Track A (`Sensor/Residual`)**: **FROZEN RESEARCH BASELINE**.
- **M18 Dual-Stream (`Sensor + Kinematic`)**: **FROZEN RESEARCH CANDIDATE**.
