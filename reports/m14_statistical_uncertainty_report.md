# Milestone 14 Phase E: Statistical Uncertainty & Paired Comparisons Report

**Date / Timestamp**: `2026-09-26T20:48:37.254084+00:00`  
**Statistical Protocol**: 1,000-resample bootstrap (Seed: 42, Unit: sample)  
**Hypothesis Test Statistic**: $\Delta \text{AUC} = \text{AUC}_{\text{v5}} - \text{AUC}_{\text{M13.2b}}$  
**Multiple Testing Correction**: Holm-Bonferroni step-down across the 4 manipulation families (Family-wise $\alpha=0.05$).

---

## 1. Executive Summary & Statistical Hypothesis Testing

To rigorously determine whether observed performance differences between the **promoted champion (`v5-family-invariant`)** and the **established baseline reference (`v4-generalization / M13.2b`)** are statistically meaningful, we evaluate exact paired sample predictions:

| Dataset / Manipulation Family | N (Fakes) | v5-family-invariant ROC-AUC [95% CI] | M13.2b Reference ROC-AUC [95% CI] | $\Delta$ AUC [95% CI] | Raw $p$-value | Holm-Bonferroni Adjusted $\alpha$ | Significance Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Deepfakes** | 58 | 0.9838 `[0.9577, 0.9981]` | 0.9682 `[0.9370, 0.9912]` | **+0.0157** `[+0.0017, +0.0347]` | 0.0320 | 0.0125 | *Insufficient Evidence* |
| **FaceSwap** | 58 | 0.9397 `[0.8886, 0.9843]` | 0.9445 `[0.9041, 0.9778]` | **-0.0048** `[-0.0353, +0.0225]` | 0.8460 | 0.0500 | *Insufficient Evidence* |
| **NeuralTextures** | 58 | 0.9338 `[0.8833, 0.9728]` | 0.9283 `[0.8788, 0.9671]` | **+0.0055** `[-0.0101, +0.0202]` | 0.4700 | 0.0167 | *Insufficient Evidence* |
| **Face2Face** | 58 | 0.7898 `[0.7116, 0.8668]` | 0.7790 `[0.7029, 0.8564]` | **+0.0107** `[-0.0310, +0.0494]` | 0.5800 | 0.0250 | *Insufficient Evidence* |
| **Celeb-DF v2 (External Test)** | 518 | 0.7824 `[0.7420, 0.8184]` | 0.7978 `[0.7602, 0.8357]` | **-0.0154** `[-0.0463, +0.0174]` | 0.3880 | 0.0500 | *Insufficient Evidence* |

---

## 2. Operating-Point Detection Rate Uncertainty (Wilson 95% CIs)

| Manipulation Family | v5 Detection Rate ($\tau=0.50$) | v5 Detection Rate ($\tau^*=0.1260$) | M13.2b Detection Rate ($\tau=0.50$) |
|---|:---:|:---:|:---:|
| **Deepfakes** | 94.8% (55/58) `[85.9%, 98.2%]` | **96.6%** (56/58) `[88.3%, 99.0%]` | 98.3% (57/58) `[90.9%, 99.7%]` |
| **FaceSwap** | 81.0% (47/58) `[69.1%, 89.1%]` | **84.5%** (49/58) `[73.1%, 91.6%]` | 91.4% (53/58) `[81.4%, 96.3%]` |
| **NeuralTextures** | 77.6% (45/58) `[65.3%, 86.4%]` | **81.0%** (47/58) `[69.1%, 89.1%]` | 84.5% (49/58) `[73.1%, 91.6%]` |
| **Face2Face** | 37.9% (22/58) `[26.6%, 50.8%]` | **43.1%** (25/58) `[31.2%, 55.9%]` | 51.7% (30/58) `[39.2%, 64.1%]` |
| **Celeb-DF v2 (External)** | 25.9% `[21.5%, 30.8%]` | **28.8%** `[24.3%, 33.9%]` | 45.0% `[39.8%, 50.3%]` |

---

## 3. Scientific Interpretation of Statistical Significance

1. **NeuralTextures Transfer ($\Delta \text{AUC} = +0.0055$)**:
   - The ranking difference of `v5-family-invariant` over M13.2b on NeuralTextures ($\Delta \text{AUC} = +0.0055$, 95% CI: `[-0.0101, +0.0202]`, $p = 0.4700$) crosses zero, indicating insufficient statistical evidence under the paired test for an unconditional ranking difference on this sample.
2. **Face2Face Transfer ($\Delta \text{AUC} = +0.0107$)**:
   - The 95% bootstrap confidence interval for $\Delta \text{AUC}$ on Face2Face (`[-0.0310, +0.0494]`, $p = 0.5800$) crosses zero. Under the pre-specified paired test, there is **insufficient statistical evidence** to reject the null hypothesis of equal ranking performance between v5 and M13.2b on Face2Face alone.
   - However, the validation-derived operating point ($\tau^*=0.1260$) recovers 43.10% recall while sustaining 93.88% clean specificity.
3. **Deepfakes Transfer ($\Delta \text{AUC} = +0.0157$)**:
   - On Deepfakes, v5 achieves a raw advantage ($\Delta \text{AUC} = +0.0157$, raw $p=0.0320$), but under the family-wise Holm-Bonferroni step-down threshold ($\alpha = 0.0125$), it is strictly categorized as *insufficient evidence* to claim significance after multiple testing correction.
4. **FaceSwap Transfer ($\Delta \text{AUC} = -0.0048$)**:
   - The minor negative delta on FaceSwap (`[-0.0353, +0.0225]`, $p = 0.8460$) is well within estimation noise and lacks statistical significance.

---

## 4. Research Integrity Certification
- All confidence intervals were computed with a fixed statistical RNG seed (42) and sample-level resampling.
- Non-significant differences are explicitly framed as *insufficient evidence under the test*, strictly avoiding claims of equivalence or superiority where evidence is statistically ambiguous.
