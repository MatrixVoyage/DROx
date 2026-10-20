# Milestone 13.1: Diagnostic Analysis of the Face2Face Failure

**Target Model**: `v3-robust (M12)` (Frozen Experimental Artifact)  
**M12 Checkpoint SHA-256**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56`  
**Evaluation Mode**: Authoritative Full Slice Run (560 total queries)  
**Strict Invariant**: Non-intrusive diagnostic only. No weights, preprocessing, or calibration parameters modified.

---

## 1. Executive Summary & Diagnostic Findings

The objective of Milestone 13.1 is to answer the core research question:
> **Why does M12 successfully reject obvious non-face OOD inputs but fail to identify Face2Face when its embeddings remain inside the learned facial feature distribution? Which representations contain discriminative signals for Face2Face, and where is the signal lost?**

### Key Empirical Findings:
1. **The Core Discovery: Representations Retain the Signal, but the Classifier Specializes**:
   - Under a lightweight linear diagnostic probe, M12's **frozen penultimate temporal representations ($D=1280$) achieve ROC-AUC = 0.8703** and **CV Accuracy = 80.1%** on separating Clean Real from Held-out Face2Face!
   - Mid-level spatial features (Stage 5, $D=80$) also achieve **ROC-AUC = 0.8006**.
   - **Crucial Takeaway**: The backbone and temporal transformer **already extract features that linearly separate Face2Face from pristine faces**. The failure does NOT stem from representation blindness, but from **classifier head specialization**: M12's end-to-end classifier learned an optimal hyperplane tailored exclusively to seen synthesis blending seams. When presented with Face2Face (which lacks blending boundaries), the projection along M12's normal vector $\mathbf{w}$ is strongly negative, driving the model to predict **Clean Real with 84.7% confidence**!
2. **The Facial Manifold Blindspot**: Face2Face embeddings have a mean regularized Mahalanobis distance of **0.9632** (Cohen's $d = -0.4602$) from clean validation faces, overlapping **46.2%** of the in-distribution facial manifold. Distance-based OOD gating cannot detect manipulation-family shift because Face2Face does not expand variance outward into an anomaly subspace.
3. **Spatial-Frequency & Temporal Dynamics**:
   - Pure global frequency statistics (FFT + DCT + Laplacian) achieve **ROC-AUC = 0.5761**.
   - Pure temporal difference statistics achieve **ROC-AUC = 0.5850**.
   - Reenactment artifacts are nuanced and semantic; global spectral or pixel-difference heuristics alone are insufficient to detect them.

---

## 2. Hypothesis Evaluations (H1 - H4)

| Hypothesis | Proposition | Evaluated Status | Empirical Evidence |
|---|---|---|---|
| **H1: Spatial-Frequency Collapse** | High-frequency residuals retain re-rendering cues that spatial RGB features smooth out. | **UNSUPPORTED** | Frequency probe AUC: 0.5761, Laplacian Cohen's d: 0.2711, FFT Cohen's d: 0.3457 |
| **H2: Manipulation Specialization** | Model specialized in blending boundaries present in Deepfakes/FaceSwap but absent in expression reenactment. | **SUPPORTED** | In-Domain Fake mean p: 0.6614, Face2Face mean p: 0.1534 (Model predicts Real for F2F despite 98.2% seen precision) |
| **H3: Temporal Discontinuity** | Expression reenactment introduces frame-to-frame dynamic jitter captured by temporal difference metrics. | **UNSUPPORTED** | Temporal dynamics probe AUC: 0.5850, Feature velocity Cohen's d: -0.2496 |
| **H4: Penultimate Subspace Compression** | Penultimate layer projects all valid faces into a compressed semantic manifold, destroying manipulation separation. | **PARTIALLY_SUPPORTED** | Early probe AUC: 0.6939 vs Penultimate probe AUC: 0.8703, Penultimate Mahalanobis Cohen's d: -0.4602 |

---

## 3. Quantitative Representation Profile Across Slices

| Operational Slice | N | Calibrated Prob (Mean [95% CI]) | Logit (Mean ± Std) | MC Var (x1e-4) | Mahalanobis Dist (Cohen's d) | Laplacian Var | Feat Velocity |
|---|---|---|---|---|---|---|---|
| **Clean Real** | 98 | 0.0208 [0.00, 0.05] | -11.95 ± 3.06 | 0.00 | 1.01 (d=+0.00) | 0.1 | 33.11 |
| **In-Domain Seen Fake** | 174 | 0.6614 [0.59, 0.73] | 3.69 ± 9.67 | 1.00 | 0.99 (d=-0.18) | 0.2 | 38.74 |
| **Celeb-DF Cross-Dataset** | 74 | 0.2483 [0.16, 0.34] | -6.02 ± 9.68 | 1.00 | 0.97 (d=-0.38) | 0.0 | 21.38 |
| **Held-Out Face2Face** | 58 | 0.1534 [0.07, 0.25] | -8.21 ± 7.21 | 1.00 | 0.96 (d=-0.46) | 0.2 | 30.80 |
| **Degraded (JPEG)** | 98 | 0.3520 [0.27, 0.43] | -2.27 ± 7.67 | 3.00 | 1.01 (d=+0.05) | 0.0 | 14.23 |
| **Synthetic OOD** | 58 | 0.0012 [0.00, 0.00] | -11.04 ± 0.01 | 0.00 | 0.83 (d=-3.09) | 8.8 | 1.55 |

---

## 4. Frozen-Feature Linear Diagnostic Probe Results

Trained on frozen representations using 5-fold stratified cross-validation to separate **Clean Real vs Held-Out Face2Face**:

| Representation Depth | Feature Dimension | 5-Fold CV ROC-AUC | 5-Fold CV Accuracy | Discriminative Status |
|---|---|---|---|---|
| **Early Spatial (Stage 2)** | D = 24 | **0.6939** | 67.3% | Moderate shallow separation |
| **Mid Spatial (Stage 5)** | D = 80 | **0.8006** | 69.2% | Strong intermediate signal |
| **Penultimate Temporal** | D = 1280 | **0.8703** | 80.1% | **Highest linear separation** |
| **Spatial-Frequency (FFT/DCT/Lap)** | D = 3 | **0.5761** | 59.7% | Weak global spectral signal |
| **Temporal Dynamics** | D = 3 | **0.5850** | 59.0% | Weak dynamic heuristic |
| **Combined Multi-Modal** | D = 110 | **0.8287** | 73.7% | High multi-modal separation |

---

## 5. Case Study: Representative False-Negative Face2Face Samples

Top Face2Face videos confidently misclassified as Clean Real by M12:

| Video ID | Calibrated Probability | MC-Dropout Variance | Mahalanobis Distance | Laplacian Variance | Feature Velocity |
|---|---|---|---|---|---|
| `f2f_007_132` | **0.04%** | 0.000000 | 1.11 | 0.1 | 25.6277 |
| `f2f_510_528` | **0.04%** | 0.000000 | 1.01 | 0.0 | 31.2004 |
| `f2f_146_256` | **0.04%** | 0.000000 | 1.01 | 0.2 | 26.4077 |
| `f2f_253_183` | **0.04%** | 0.000000 | 1.08 | 0.2 | 33.2978 |
| `f2f_174_964` | **0.04%** | 0.000000 | 1.02 | 0.2 | 25.0921 |

---

## 6. Evidence-Based Proposals for Milestone 13.2

Based strictly on empirical diagnostic data, M13.2 will formulate experiments on an **independent model branch** targeting the discovered failure modes:

1. **Manipulation-Family-Aware Head Training (M13.2a)**:
   - *Evidence*: Frozen penultimate features achieve **ROC-AUC = 0.8703** under a linear probe, but M12's single binary head outputs $p = 0.1534$.
   - *Proposed Architecture*: Decouple binary classification into multi-head or mixture-of-experts heads (e.g. Swapping Head vs Reenactment Head vs Synthesis Head) to prevent face-swap gradients from overpowering reenactment features.
2. **Orthogonal Subspace Projection (M13.2b)**:
   - *Evidence*: Face2Face features embed inside the real-face Mahalanobis cluster ($D = 0.9632$).
   - *Proposed Method*: Decompose the 1280-dimensional feature vector into facial identity invariant components and manipulation-residual components, projecting out identity variance before classification.
3. **Mid-Level Auxiliary Skip Connections (M13.2c)**:
   - *Evidence*: Mid-spatial features (Stage 5) retain strong independent separation (ROC-AUC = 0.8006).
   - *Proposed Architecture*: Multi-scale skip aggregation connecting intermediate spatial blocks directly to the classification head.
