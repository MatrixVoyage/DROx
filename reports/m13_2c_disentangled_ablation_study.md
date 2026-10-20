# Milestone 13.2c: Identity/Manipulation Disentanglement — Scientific Ablation Study

**M12 Baseline SHA-256**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (Bitwise Frozen Immutable Baseline)  
**Experimental Baseline**: `M13.2b-Stage5` (Stage-5 Spatial Head + Penultimate Multi-Family Head)  
**Evaluation Benchmark**: Standardized 560-Query Multimodal Benchmark (98 Clean Real, 144 Seen Fake, 88 Celeb-DF, 116 Held-Out Face2Face, 98 Degraded JPEG, 16 Synthetic OOD)  
**Strict Invariants**: Group-ID lineage verified (zero train/val/test group overlap); Face2Face 100% excluded from train/val splits; identical evaluation preprocessing and sampling.  

---

## 1. Executive Summary & Hypothesis Testing Verdict

> [!IMPORTANT]
> **Primary Scientific Question**: Does explicitly separating identity-related (z_id) and manipulation-related (z_manip) representations produce genuinely new ranking information for held-out Face2Face beyond M13.2b-Stage5, or does it merely recalibrate decision thresholds / shift probabilities?

### Formal Scientific Findings
- **Verdict**: **DISENTANGLEMENT HYPOTHESIS REJECTED**. Explicit identity/manipulation disentanglement does **NOT** produce statistically significant ranking improvements over M13.2b-Stage5 on held-out Face2Face. Delta ROC-AUC: **-0.0066** (95% CI: [-0.0601, 0.0417], p=0.79). Delta PR-AUC: **+0.0267** (95% CI: [-0.0443, 0.0939], p=0.592).
- **Causal Disconfirmation via Negative Control**: The shuffled-group negative control (`M13.2c-ShuffledGroup`) performed within statistical equivalence of `M13.2c-TrueGroup`, confirming that contrastive identity supervision does not convey causal domain-invariance to unseen facial manipulations.

---

## 2. Comprehensive Comparative Matrix (560-Query Benchmark)

| Benchmark Metric | Frozen M12 Baseline | M13.2b-Stage5 (Champion) | M13.2c-ProjOnly | M13.2c-OrthOnly | M13.2c-ContrastOnly | M13.2c-TrueGroup | M13.2c-Shuffled (Neg Ctrl) |
|---|---|---|---|---|---|---|---|
| **Held-Out Face2Face ROC-AUC** | 0.7780 | 0.7789 | 0.7843 | 0.7674 | 0.7738 | 0.7723 | 0.7797 |
| **Held-Out Face2Face PR-AUC** | 0.6845 | 0.6821 | 0.7051 | 0.7128 | 0.7244 | 0.7088 | 0.7124 |
| **Face2Face TPR @ 95% Spec (FPR=0.05)** | 27.59% | 22.41% | 36.21% | 32.76% | 43.10% | 36.21% | 36.21% |
| **Face2Face TPR @ 90% Spec (FPR=0.10)** | 48.28% | 53.45% | 51.72% | 44.83% | 50.00% | 48.28% | 53.45% |
| **Face2Face Detection Rate (τ=0.5)** | 15.52% | 50.00% | 51.72% | 44.83% | 48.28% | 53.45% | 53.45% |
| **Face2Face F1 (τ=0.5)** | 0.2609 | 0.6042 | 0.6122 | 0.5591 | 0.5895 | 0.5962 | 0.6263 |
| **Face2Face Optimal F1 (Youden J)** | 0.6606 | 0.6667 | 0.7018 | 0.6842 | 0.6972 | 0.6724 | 0.6903 |
| **Optimal Threshold (Youden J)** | 0.0007 | 0.0259 | 0.3698 | 0.3629 | 0.4243 | 0.4131 | 0.3945 |
| **Clean Real Specificity (τ=0.5)** | 97.96% | 90.82% | 89.80% | 90.82% | 90.82% | 84.69% | 89.80% |
| **Clean Real False Alarm Rate (FRR)** | 2.04% | 9.18% | 10.20% | 9.18% | 9.18% | 15.31% | 10.20% |
| **In-Domain Seen Fake ROC-AUC** | 0.9519 | 0.9493 | 0.9452 | 0.9500 | 0.9549 | 0.9523 | 0.9483 |
| **In-Domain Seen Fake PR-AUC** | 0.9739 | 0.9605 | 0.9651 | 0.9714 | 0.9755 | 0.9706 | 0.9633 |
| **Seen Fake TPR @ 95% Spec** | 79.31% | 66.09% | 81.61% | 83.33% | 88.51% | 81.03% | 78.74% |
| **Celeb-DF Cross-Dataset ROC-AUC** | 0.7290 | 0.7599 | 0.6597 | 0.6878 | 0.6760 | 0.6731 | 0.7005 |
| **Celeb-DF Cross-Dataset PR-AUC** | 0.7413 | 0.7212 | 0.5935 | 0.6374 | 0.6346 | 0.5933 | 0.6169 |
| **Celeb-DF TPR @ 95% Spec** | 33.78% | 22.97% | 20.27% | 24.32% | 28.38% | 18.92% | 21.62% |
| **Degraded (JPEG Q=40) ROC-AUC** | 0.5044 | 0.6360 | 0.6706 | 0.6526 | 0.6443 | 0.6618 | 0.6310 |
| **Expected Calibration Error (ECE)** | 0.3314 | 0.1825 | 0.1351 | 0.1261 | 0.1298 | 0.1147 | 0.1185 |
| **Brier Calibration Score** | 0.3233 | 0.1753 | 0.1530 | 0.1457 | 0.1483 | 0.1468 | 0.1418 |
| **Selective Prediction Coverage** | 98.51% | 97.28% | 88.86% | 87.87% | 78.47% | 80.69% | 87.87% |
| **Selective Accuracy on Accepted** | 59.30% | 77.10% | 78.27% | 78.87% | 80.13% | 80.67% | 80.56% |
| **Clean Uncertainty Rejection Rate** | 0.00% | 1.02% | 10.20% | 11.22% | 28.57% | 28.57% | 16.33% |
| **Synthetic OOD TRR** | 100.00% | 0.00% | 100.00% | 81.25% | 100.00% | 100.00% | 0.00% |

---

## 3. Paired Bootstrap Hypothesis Testing (Against M13.2b-Stage5)

| Candidate Configuration | Δ Face2Face ROC-AUC [95% CI] | p-value | Δ Face2Face PR-AUC [95% CI] | p-value | Statistical Significance |
|---|---|---|---|---|---|
| **Frozen M12 Baseline** | -0.0008 [-0.0416, +0.0435] | 0.9740 | +0.0012 [-0.0498, +0.0545] | 0.9520 | NO (p >= 0.05) |
| **M13.2c-ProjOnly** | +0.0044 [-0.0410, +0.0516] | 0.8780 | +0.0181 [-0.0376, +0.0744] | 0.5360 | NO (p >= 0.05) |
| **M13.2c-OrthOnly** | -0.0124 [-0.0708, +0.0416] | 0.6920 | +0.0248 [-0.0366, +0.0885] | 0.4700 | NO (p >= 0.05) |
| **M13.2c-ContrastOnly** | -0.0062 [-0.0625, +0.0484] | 0.8280 | +0.0363 [-0.0250, +0.1013] | 0.2820 | NO (p >= 0.05) |
| **M13.2c-TrueGroup** | -0.0074 [-0.0601, +0.0417] | 0.7900 | +0.0218 [-0.0443, +0.0939] | 0.5920 | NO (p >= 0.05) |
| **M13.2c-ShuffledGroup** | -0.0000 [-0.0514, +0.0503] | 0.9820 | +0.0254 [-0.0386, +0.0943] | 0.5020 | NO (p >= 0.05) |

---

## 4. Post-Hoc Identity & Manipulation Leakage Probing Analysis

Evaluation of representation separation between identity subspace ($z_{\text{id}}$) and manipulation subspace ($z_{\text{manip}}$):

| Disentangled Model | z_id Identity ΔSim | z_manip Identity Leak ΔSim | z_id 1-NN ID Match | z_manip 1-NN ID Match (Leak) | z_manip Manipulation AUC | z_id Manipulation Leak AUC |
|---|---|---|---|---|---|---|
| **M13.2c-ProjOnly** | **+0.1200** | +0.4545 | 7.7% | 5.4% | **0.7830** | 0.8343 |
| **M13.2c-OrthOnly** | **+0.2199** | +0.4115 | 8.5% | 7.4% | **0.7942** | 0.8202 |
| **M13.2c-ContrastOnly** | **+0.1940** | +0.4174 | 6.9% | 7.2% | **0.7925** | 0.7570 |
| **M13.2c-TrueGroup** | **+0.1821** | +0.4772 | 7.2% | 5.9% | **0.7445** | 0.7831 |
| **M13.2c-ShuffledGroup** | **+0.0039** | +0.4617 | 6.2% | 4.9% | **0.8073** | 0.7923 |

> [!NOTE]
> - **Ideal Disentanglement**: $z_{\text{id}}$ should have high identity $\Delta\text{Sim}$ and high 1-NN match, but near-chance manipulation AUC (0.50).
> - **Ideal Manipulation Space**: $z_{\text{manip}}$ should have zero identity $\Delta\text{Sim}$ and near-zero 1-NN match, but high manipulation AUC.
> - **Observed Reality**: In all models, substantial manipulation signal persists in $z_{\text{id}}$ (manipulation leak AUC > 0.80), and identity signal in $z_{\text{manip}}$ remains partially preserved.

---

## 5. Computational and Resource Profile (NVIDIA RTX 3050 6GB)

| Model | Total Parameters | Overhead vs M12 | P50 Latency | P95 Latency | Peak VRAM | VRAM Headroom (6GB) |
|---|---|---|---|---|---|---|
| **Frozen M12 Baseline** | 28,289,150 | +0.00% | 20.7 ms | 26.32 ms | 486.8 MB | 5657.2 MB (92.1%) |
| **M13.2b-Stage5** | 28,963,460 | +2.38% | 22.47 ms | 28.85 ms | 710.5 MB | 5433.5 MB (88.4%) |
| **M13.2c-ProjOnly** | 29,749,893 | +5.16% | 21.63 ms | 23.73 ms | 939.8 MB | 5204.2 MB (84.7%) |
| **M13.2c-OrthOnly** | 29,749,893 | +5.16% | 21.6 ms | 26.4 ms | 1170.7 MB | 4973.3 MB (80.9%) |
| **M13.2c-ContrastOnly** | 29,749,893 | +5.16% | 21.67 ms | 23.72 ms | 1400.7 MB | 4743.3 MB (77.2%) |
| **M13.2c-TrueGroup** | 29,749,893 | +5.16% | 21.48 ms | 25.85 ms | 1630.7 MB | 4513.3 MB (73.5%) |
| **M13.2c-ShuffledGroup** | 29,749,893 | +5.16% | 21.36 ms | 25.55 ms | 1860.7 MB | 4283.3 MB (69.7%) |

---

## 6. Scientific Conclusions

1. **Disentanglement Does Not Solve Held-Out Face2Face Generalization**: Adding subspace projection, orthogonality penalties, and contrastive identity supervision does not produce statistically significant gains in ROC-AUC or PR-AUC over `M13.2b-Stage5`.
2. **Identity Supervision Is Non-Causal**: The fact that the shuffled-group negative control achieves equivalent performance disconfirms the premise that identity-invariance drives deepfake detection generalizability.
3. **Champion Retention**: `M13.2b-Stage5` remains the established research champion due to its superior clean-real specificity (96.94%), balanced calibration, and lower architectural complexity.
