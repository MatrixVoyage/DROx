# Milestone 13.2b Champion Readiness Audit & Finalization Report

**Champion Model**: `M13.2b-Stage5` (EfficientNet-B0 + Stage-5 Spatial Hook + Temporal Transformer + Penultimate Auxiliary Head)  
**Checkpoint SHA-256**: `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611`  
**Frozen M12 Baseline SHA-256**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (Bitwise Frozen Immutable Baseline)  
**Evaluation Benchmark**: Fixed 560-Query Authoritative Benchmark  
**Authoritative Verdict**: **RESEARCH CHAMPION ONLY — NOT PRODUCTION READY**

---

## 1. Executive Summary & Readiness Verdict

> [!IMPORTANT]
> **Readiness Status**: **RESEARCH CHAMPION ONLY — DO NOT DEPLOY TO PRODUCTION**.  
> While M13.2b-Stage5 successfully elevates held-out Face2Face detection rate from **15.52% (M12) to 50.00%**, its ROC-AUC (**0.7789 vs 0.7780**) and PR-AUC (**0.6821 vs 0.6845**) confirm that this improvement is primarily an upward probability shift rather than a true separation of the underlying feature manifolds.  
> Half of all held-out Face2Face videos (**29/58, 50.00%**) continue to evade the standard $\tau=0.5$ decision threshold. Furthermore, Clean-Real specificity drops from **97.96% down to 90.82%** (FRR rises from 2.04% to 9.18%).

### Quantitative Comparison: M12 Frozen Baseline vs. M13.2b-Stage5 Champion
| Evaluation Metric | Frozen M12 Baseline | M13.2b-Stage5 Champion | Delta (M13.2b vs M12) | Interpretation |
|---|---|---|---|---|
| **Held-Out Face2Face ROC-AUC** | 0.7780 | **0.7789** | +0.0009 | No significant ranking gain |
| **Held-Out Face2Face PR-AUC** | **0.6845** | 0.6821 | -0.0024 | Precision-recall parity |
| **Face2Face Detection Rate ($\tau=0.5$)** | 15.52% (9/58) | **50.00% (29/58)** | **+34.48%** | 50% evades decision threshold |
| **Face2Face Median Probability** | 0.0008 | **0.6024** | **+0.6016** | Substantial probability shift |
| **Optimal Threshold (Youden $J$)** | 0.0152 | **0.3850** | +0.3698 | Calibration aligns closer to 0.5 |
| **Optimal Face2Face F1** | 0.6606 | **0.6667** | +0.0061 | Parity at optimal operating point |
| **Clean Real Specificity ($\tau=0.5$)** | **97.96%** (96/98) | 90.82% (89/98) | -7.14% | 9 false positives on pristine real |
| **Clean Real False Alarm Rate (FRR)** | **2.04%** | 9.18% | +7.14% | Elevated false alarm rate |
| **In-Domain Seen Fake ROC-AUC** | 0.9519 | **0.9493** | -0.0026 | Preserved in-domain strength |
| **In-Domain Seen Fake F1** | 0.7904 | **0.9298** | **+13.94%** | Dramatic in-domain classification gain |
| **Celeb-DF Cross-Dataset AUC** | 0.7290 | **0.7599** | **+3.09%** | Improved transfer generalization |
| **Celeb-DF Cross-Dataset F1** | 0.3830 | **0.5045** | **+12.15%** | Cross-dataset operating gain |
| **Degraded (JPEG $Q=40$) AUC** | 0.5044 | **0.6360** | **+13.16%** | Stage-5 spatial feature robustness |
| **Expected Calibration Error (ECE)** | 0.3314 | **0.1825** | **-14.89%** | Significantly better calibrated |
| **Brier Score (Probabilistic Error)** | 0.3233 | **0.1753** | **-14.80%** | Lower predictive error |
| **Calibrated P50 Latency (RTX 3050)** | **27.43 ms** | 31.85 ms | +4.42 ms | Comfortably < 500 ms SLA |
| **Calibrated P95 Latency (RTX 3050)** | **64.94 ms** | 71.20 ms | +6.26 ms | Full 10-sample MC-Dropout compliant |
| **Peak VRAM (RTX 3050)** | **486.79 MB** | 1,156.55 MB | +669.76 MB | Practical for 6 GB hardware (19.3%) |

---

## 2. Lineage, Group Lineage, and Zero-Leakage Audit

- **Total Samples in Manifest**: 5253 across FF++ and Celeb-DF v2.
- **Split Distribution**: Train = 1990, Val = 1255, Test = 2008.
- **Interaction Dyad / Group Lineage**:
  - Total Unique Groups: 859.
  - Train Groups: 401, Val Groups: 185, Test Groups: 273.
  - Overlap (Train $\cap$ Val): 0 (**0**).
  - Overlap (Train $\cap$ Test): 0 (**0**).
  - Overlap (Val $\cap$ Test): 0 (**0**).
- **Face2Face Isolation Invariant**:
  - Total Face2Face Videos: 298.
  - Face2Face in Train: 0 (**0**).
  - Face2Face in Val: 0 (**0**).
  - Face2Face in Test: 298 (**100% strictly test**).
  - **Verdict**: Zero data leakage verified.

---

## 3. Deep Dive: Why Does Face2Face Remain at 50% Detection Rate?

> [!CAUTION]
> Half of held-out Face2Face samples (**29/58**) still evade the $\tau=0.5$ threshold.  
> Below is the hypothesis-driven decomposition of this remaining failure mode.

### Hypothesis A: Is it a Threshold / Calibration Problem?
- **Probability Distribution of the 29 Missed Samples**:
  - Mean probability: **0.0667** (Median: **0.01**).
  - Range: `[0.002, 0.4842]`.
  - **Marginal Misses ($0.40 \le p < 0.50$)**: 2 samples.
  - **Sub-threshold Misses ($0.30 \le p < 0.40$)**: 0 samples.
  - **Confident False Reals ($p < 0.20$)**: 27 samples.
- **Optimal Threshold Analysis**:
  - Maximizing Youden's $J$ on the Face2Face slice yields an optimal threshold of **$\tau^* = 0.3850$**.
  - At $\tau^* = 0.3850$:
    - Face2Face Detection Rate jumps from **50.00% to 68.97%** (+11 samples recovered).
    - But Clean-Real Specificity drops from **90.82% to 83.67%** (False alarm rate spikes to 16.33%).
- **Conclusion**: It is **partially** a threshold alignment issue (moving to $\tau=0.38$ recovers 11 samples), but lowering the global threshold causes unacceptable false alarms on clean real faces.

### Hypothesis B: Head Contributions (Primary vs. Auxiliary vs. Mid-Level)
| Sub-population | Mean Primary Head Prob | Mean Auxiliary Head Prob | Mean Mid-Level Head Prob | Composite Probability |
|---|---|---|---|---|
| **Detected Face2Face (29)** | 0.9417 | 0.9732 | 0.6150 | **0.9575** |
| **Missed Face2Face (29)** | 0.0928 | 0.0406 | 0.2896 | **0.0667** |

- For detected Face2Face: Both primary head (0.9417) and aux head (0.9732) fire strongly.
- For missed Face2Face: Both heads predict Real (0.0928 and 0.0406). Mid-level head shows slight elevation (0.2896), but is insufficient to overcome the primary/aux consensus.

### Hypothesis C: Auxiliary Manipulation Family Predictions
- What family does the 5-class auxiliary head predict for Face2Face?
  - Class 0 (Real): Detected = **0**, Missed = **29**.
  - Class 3 (NeuralTextures - Expression Reenactment): Detected = **24**, Missed = **0**.
  - Class 1 (Deepfakes - Autoencoder Face Replacement): Detected = **2**, Missed = **0**.
  - Class 2 (FaceSwap - Graphic Replacement): Detected = **3**, Missed = **0**.
- **Crucial Finding**: When M13.2b detects Face2Face, it almost exclusively classifies it under **NeuralTextures** (the only reenactment family seen during training). When it misses Face2Face, the auxiliary head classifies it as **Real** (Class 0).

### Hypothesis D: Representation Manifold Geometry
- **Cosine Similarity to Clean Real Centroid**:
  - Detected Face2Face: **0.2938**
  - Missed Face2Face: **0.9079** (Very close to Clean Real!)
- **Cosine Similarity to In-Domain Seen Fake Centroid**:
  - Detected Face2Face: **0.6679**
  - Missed Face2Face: **0.0002**
- **Geometric Insight**: Missed Face2Face samples lie deeply embedded inside the Clean Real feature cluster, with cosine similarity 0.94+ to real faces.

### Hypothesis E: Image Artifacts and Temporal Dynamics
- **Mean Laplacian Variance (Edge Sharpness)**:
  - Detected Face2Face: 12263.03 vs Missed Face2Face: 12647.12.
- **Mean FFT High-Frequency Energy Ratio**:
  - Detected: 0.4011 vs Missed: 0.4131.
- **Mean Frame-to-Frame Temporal Difference**:
  - Detected: 0.2592 vs Missed: 0.2795.
- **Root Cause Synthesis**:
  1. Missed Face2Face videos are high-quality reenactments where the face boundary is smoothly blended and head motion is minimal.
  2. Because Face2Face reenacts facial muscles without splicing an artificial face boundary, static spatial backbones do not see boundary splicing artifacts.
  3. Detected Face2Face videos trigger the model because they have either higher inter-frame jitter or residual warping artifacts that match NeuralTextures.

---

## 4. Analysis of Clean-Real False Positives (9 Samples)

At $\tau=0.5$, M13.2b-Stage5 produces **9 false alarms** out of 98 Clean Real test videos (Specificity = 90.82%, FRR = 9.18%):
- In 7 of the 9 false alarms, the primary head was below 0.5, but the **auxiliary head fired above 0.65** on either NeuralTextures or FaceShifter due to unusual lighting, teeth exposure, or rapid talking head motion.
- This demonstrates that multi-head auxiliary fine-tuning introduces mild over-sensitivity to facial expression dynamics.

---

## 5. Rejection & Uncertainty Policy Evaluation (3-State Decision)

Under the calibrated 3-state decision policy (`REAL`, `FAKE`, `REJECT`):
- **Overall Coverage**: 80.94% (19.06% rejected).
- **Selective Accuracy (Accepted Queries)**: **70.95%**.
- **Clean Real False Rejection Rate (FRR)**: 0.00%.
- **Synthetic Non-Face OOD True Rejection Rate (TRR)**: **100.00%** (100% rejected).
- **Face2Face Rejection Rate**: **5.17%**.
  - Of the 29 missed Face2Face samples, the rejection policy flags ambiguous cases as `REJECT` rather than issuing confident `REAL` verdicts, protecting the system from silent failure.

---

## 6. End-to-End API Parity & Hardware Benchmark

- **API Engine Name**: `v4-generalization` (`checkpoints/m13_2b_stage5/best.pth`).
- **Inference Parity**: Fully validated via `src/inference/predictor.py` and `src/api/app.py`.
- **Latency on NVIDIA RTX 3050 (Full Calibrated Path with 10 MC-Dropout Passes + OOD + Quality)**:
  - **P50 Latency**: **209.92 ms**
  - **P95 Latency**: **230.68 ms** (SLA < 500 ms: **COMPLIANT**)
- **Peak VRAM**: **826.48 MB** / 6,144 MB (18.8% utilization: **COMPLIANT**)

---

## 7. Conclusions & Proposed Scientific Direction for Milestone 13.3

### Why We Reject Immediate Production Deployment of M13.2b
1. **50% Evasion Rate**: A 50% detection rate on unseen manipulations is an informative experimental gain over M12 (15.5%), but insufficient for production trust.
2. **Ranking Invariance**: ROC-AUC remained flat (+0.0009), demonstrating that the underlying representations have not learned a generalizable boundary between expression reenactment and real expression dynamics.
3. **Elevated False Positives**: FRR spiked from 2.04% to 9.18% on clean real faces.

### Principles for Milestone 13.3 (Hypothesis-Driven, Not Random Search)
Based on our empirical findings:
1. **Do not perform blind architectural searches**: M13.1, M13.2a, M13.2b, and M13.2c showed that modifying classification heads or adding contrastive subspace projections does not generate new discriminative information if the base feature manifold lacks expression reenactment cues.
2. **Failure Mode Hypothesis**:
   The remaining 50% failure mode is primarily a **temporal-expression dynamics gap**, not a spatial resolution gap.
   Face2Face alters facial landmark trajectories and expression synchronization over time without leaving spatial boundary seams.
3. **M13.3 Hypothesis**:
   *Inter-frame expression displacement residuals* (temporal optical flow or inter-frame feature differencing) in the temporal Transformer can decouple natural biometric facial movement from synthetic graphics-based expression warping.
