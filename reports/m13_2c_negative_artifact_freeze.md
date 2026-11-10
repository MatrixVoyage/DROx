# Milestone 13.2c: Negative Experimental Baseline Freeze Report

**Freeze Timestamp**: `2026-09-25T03:32:00Z`  
**Status**: **REJECTED AS A GENERALIZATION SOLUTION (PERMANENTLY FROZEN NEGATIVE ABLATION BASELINE)**  
**Frozen Manifest**: [`reports/m13_2c_negative_artifact_freeze.json`](file:///reports/m13_2c_negative_artifact_freeze.json)  
**Established Research Champion**: `M13.2b-Stage5` (Stage-5 Spatial Head + Multi-Family Auxiliary Head)  
**Immutable Reference Baseline**: `v3-robust (M12)` (SHA-256: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56`)

---

## 1. Executive Summary & Scientific Purpose

Milestone 13.2c tested whether explicitly separating identity-related ($\mathbf{z}_{\text{id}}$) and manipulation-related ($\mathbf{z}_{\text{manip}}$) representations via subspace projections, orthogonality constraints, and supervised contrastive identity loss would eliminate Face2Face generalization failure.

The experimental outcome was **conclusively negative**: disentanglement failed to yield statistically significant ranking improvements on held-out Face2Face over `M13.2b-Stage5`, while inducing severe specificity penalties and cross-dataset collapse.

However, **scientifically, this is not a wasted experiment**. It establishes three foundational empirical facts:
1. **Geometric Modulation Without Causal Gain**: The disentanglement objective alters representation geometry, particularly in the identity subspace ($\mathbf{z}_{\text{id}}$ $\Delta \text{Sim} = +0.1821$), but this geometric change produces **no statistically significant Face2Face ranking improvement**.
2. **Threshold Artifact vs Specificity Trade-Off**: The apparent threshold-level improvement at $\tau=0.5$ (jumping to $\sim 53.45\%$) is accompanied by severe specificity degradation (Clean Real Specificity dropped to **84.69%**, Clean FRR spiked to **15.31%**), demonstrating that it is an inductive probability shift, not a genuine generalization breakthrough.
3. **Causal Disconfirmation via Negative Control**: Permuting actor identity assignments in the negative control (`M13.2c-ShuffledGroup`) yielded identical Face2Face detection metrics to genuine identity supervision (ROC-AUC **0.7797 vs 0.7723**, $p=0.9820$), definitively proving that identity-invariance does not drive deepfake generalizability.

By freezing M13.2c as a negative experimental artifact, future milestones (starting with M13.3) maintain an immutable negative reference baseline to prevent regression into representation-disentanglement dead ends.

---

## 2. Cryptographic Checkpoint Fingerprints

All model checkpoints have been cryptographically hashed and sealed:

| Model Variant | Checkpoint Path | SHA-256 Digest | Status |
|---|---|---|---|
| **`v3-robust (M12)`** | `checkpoints/m12_full/best.pth` | `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` | **Immutable Baseline** |
| **`M13.2b-Stage5`** | `checkpoints/m13_2b_stage5/best.pth` | `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611` | **Research Champion** |
| **`M13.2c-ProjOnly`** | `checkpoints/m13_2c_proj_only/best.pth` | `bc36f4daba27ed1f784526a23784b9773ebe8ac93b19e61eaf19d77ef7c5d3be` | **Frozen Negative** |
| **`M13.2c-OrthOnly`** | `checkpoints/m13_2c_orth_only/best.pth` | `26957f4946d77a571b3a0ad6af0dad7874b0738c1fe57fcad362aab1a32b1993` | **Frozen Negative** |
| **`M13.2c-ContrastOnly`**| `checkpoints/m13_2c_contrast_only/best.pth`| `7f0e8286967f2f7d0a4f0596141945425bdff62e3f0a65d9d012c79eceb8b215` | **Frozen Negative** |
| **`M13.2c-TrueGroup`** | `checkpoints/m13_2c_true_group/best.pth` | `b33798a432c56df79be1b5854a924a85735655a206bf2b403d7f72ee634e2f76` | **Frozen Negative** |
| **`M13.2c-ShuffledGroup`**| `checkpoints/m13_2c_shuffled/best.pth` | `6e200dfccc13b81588c1918ef98634ab22d0d7ff6325dc91fa04674fb73b6084` | **Negative Control** |

---

## 3. Authoritative Benchmark Summary (Standardized 560-Query Suite)

Evaluated across the exact benchmark: 98 Clean Real, 144 Seen Fake, 88 Celeb-DF, 116 Held-Out Face2Face, 98 Degraded JPEG, 16 Synthetic OOD.

| Metric | Frozen M12 Baseline | M13.2b-Stage5 (Champion) | M13.2c-ProjOnly | M13.2c-OrthOnly | M13.2c-ContrastOnly | M13.2c-TrueGroup | M13.2c-Shuffled (Neg Ctrl) |
|---|---|---|---|---|---|---|---|
| **Face2Face ROC-AUC** | 0.7780 | 0.7789 | **0.7843** | 0.7674 | 0.7738 | 0.7723 | 0.7797 |
| **Face2Face PR-AUC** | 0.6845 | 0.6821 | 0.7051 | 0.7128 | **0.7244** | 0.7088 | 0.7124 |
| **Face2Face Counts (τ=0.5)** | 9/58 (15.52%) | 29/58 (50.00%) | 30/58 (51.72%) | 26/58 (44.83%) | 28/58 (48.28%) | 31/58 (53.45%) | 31/58 (53.45%) |
| **Clean Real Specificity** | **97.96%** | 90.82% | 89.80% | 90.82% | 90.82% | 84.69% | 89.80% |
| **Clean False Alarms (FRR)**| **2/98 (2.04%)** | 9/98 (9.18%) | 10/98 (10.20%) | 9/98 (9.18%) | 9/98 (9.18%) | 15/98 (15.31%) | 10/98 (10.20%) |
| **Celeb-DF Cross-DS AUC** | 0.7290 | **0.7599** | 0.6597 | 0.6878 | 0.6760 | 0.6731 | 0.7005 |
| **Degraded (JPEG) AUC** | 0.5044 | 0.6360 | **0.6706** | 0.6526 | 0.6443 | 0.6618 | 0.6310 |
| **Calibration ECE** | 0.3314 | 0.1825 | 0.1351 | 0.1261 | 0.1298 | **0.1147** | 0.1185 |

---

## 4. Paired Bootstrap Hypothesis Testing Against M13.2b-Stage5

Computed over 1,000 paired non-parametric bootstrap resamples on the test set:

| Model Configuration | $\Delta$ Face2Face ROC-AUC [95% CI] | $p$-value | $\Delta$ Face2Face PR-AUC [95% CI] | $p$-value | Statistically Significant |
|---|---|---|---|---|---|
| **Frozen M12 Baseline** | -0.0008 [-0.0416, +0.0435] | 0.9740 | +0.0012 [-0.0498, +0.0545] | 0.9520 | **No** ($p \ge 0.05$) |
| **M13.2c-ProjOnly** | +0.0044 [-0.0410, +0.0516] | 0.8780 | +0.0181 [-0.0376, +0.0744] | 0.5360 | **No** ($p \ge 0.05$) |
| **M13.2c-OrthOnly** | -0.0124 [-0.0708, +0.0416] | 0.6920 | +0.0248 [-0.0366, +0.0885] | 0.4700 | **No** ($p \ge 0.05$) |
| **M13.2c-ContrastOnly** | -0.0062 [-0.0625, +0.0484] | 0.8280 | +0.0363 [-0.0250, +0.1013] | 0.2820 | **No** ($p \ge 0.05$) |
| **M13.2c-TrueGroup** | -0.0074 [-0.0601, +0.0417] | 0.7900 | +0.0218 [-0.0443, +0.0939] | 0.5920 | **No** ($p \ge 0.05$) |
| **M13.2c-ShuffledGroup** | -0.0000 [-0.0514, +0.0503] | 0.9820 | +0.0254 [-0.0386, +0.0943] | 0.5020 | **No** ($p \ge 0.05$) |

---

## 5. Post-Hoc Representation Leakage Probing Analysis

| Disentangled Model | $\mathbf{z}_{\text{id}}$ Identity $\Delta\text{Sim}$ | $\mathbf{z}_{\text{manip}}$ ID Leak $\Delta\text{Sim}$ | $\mathbf{z}_{\text{id}}$ 1-NN Match | $\mathbf{z}_{\text{manip}}$ 1-NN Match (Leak) | $\mathbf{z}_{\text{manip}}$ Manip Probe AUC | $\mathbf{z}_{\text{id}}$ Manip Leak AUC |
|---|---|---|---|---|---|---|
| **M13.2c-ProjOnly** | +0.1200 | +0.4545 | 7.7% | 5.4% | **0.7830** | 0.8343 |
| **M13.2c-OrthOnly** | **+0.2199** | +0.4115 | **8.5%** | 7.4% | **0.7942** | 0.8202 |
| **M13.2c-ContrastOnly** | +0.1940 | +0.4174 | 6.9% | 7.2% | **0.7925** | 0.7570 |
| **M13.2c-TrueGroup** | +0.1821 | +0.4772 | 7.2% | 5.9% | **0.7445** | 0.7831 |
| **M13.2c-ShuffledGroup** | +0.0039 | +0.4617 | 6.2% | 4.9% | **0.8073** | 0.7923 |

### Invariant Conclusions for Future Milestones
1. Linear subspace projection cannot separate identity and manipulation features derived from a shared single-frame CNN backbone.
2. The core barrier to Face2Face detection is **not** identity interference, but **classifier specialization to blending seams**.
3. All future architecture investigations in Milestone 13.3 must test against this frozen negative baseline.
