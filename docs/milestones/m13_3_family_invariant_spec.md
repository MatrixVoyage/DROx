# Milestone 13.3: Family-Invariant Artifact Representation — Specification & Research Plan

**Milestone Identifier**: `M13.3`  
**Phase**: Experimental Research Branch  
**Target Hardware Envelope**: NVIDIA RTX 3050 Laptop GPU (6,144 MB VRAM, 45W TGP)  
**Primary Reference Paper**: *Deepfake Video Detection using ResNeXt-50 and LSTM* (Scientific Reports)  
**Frozen Baselines**:
- `v3-robust (M12)` (SHA-256: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56`) — Bitwise Immutable Baseline
- `M13.2b-Stage5` (SHA-256: `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611`) — Established Empirical Research Champion
- `M13.2c-Ablation` ([`reports/m13_2c_negative_artifact_freeze.json`](file:///g:/MyProjects/DeepFake%20Video%20Detection/reports/m13_2c_negative_artifact_freeze.json)) — Immutable Negative Experimental Baseline

---

## 1. Scientific Motivation & Core Research Question

### Retrospective from Milestone 13.2c
Milestone 13.2c definitively proved that identity subspace disentanglement does **not** provide generalizable manipulation ranking power:
1. `M13.2c-TrueGroup` ($\Delta \text{ROC-AUC} = -0.0074$, $p=0.79$) and `M13.2c-ProjOnly` ($\Delta \text{ROC-AUC} = +0.0044$, $p=0.88$) yielded no statistically significant ranking improvements over `M13.2b-Stage5`.
2. The negative control (`M13.2c-ShuffledGroup`) performed identically to genuine identity contrastive supervision ($\Delta = -0.0000, p=0.982$), proving that identity-invariance is non-causal for deepfake generalization.
3. Disentanglement caused severe clean-real specificity degradation (FRR spiked to 15.31%) and cross-dataset collapse on Celeb-DF (ROC-AUC dropped from 0.7599 to 0.6731).

The research question therefore shifts away from:
> *"Is identity contamination masking manipulation artifacts?"* (Disproven)

To the fundamental open problem of visual forensics:
> **Core Research Question**:  
> *What information in the frozen M13.1/M13.2 representations distinguishes manipulation families, and how can a detector be trained to preserve that information without specializing to the artifact signature of known families?*

---

## 2. Theoretical Framing: The Specialization Hypothesis

Our diagnostic findings from M13.1 and M13.2 demonstrate:
- **Representation Richness**: Penultimate features already linearly separate Face2Face from Clean Real with **ROC-AUC = 0.8703** under linear probes.
- **Classifier Specialization**: Standard ERM training causes classifier weights $\mathbf{w}$ to align exclusively with **spatial blending seams** characteristic of Deepfakes and FaceSwap.
- **Face2Face Anomaly**: Face2Face is an expression/mouth reenactment that warps interior facial pixels without introducing blending boundary discontinuities. Consequently, Face2Face samples project into the negative half-space of $\mathbf{w}$, with 27 of 29 missed samples predicted as Real ($p < 0.20$, median $p = 0.0100$).
- **Sample Count Context**: The test benchmark contains 58 Face2Face samples. A threshold detection shift of $+3.45\%$ (from 50.00% [29/58] to 53.45% [31/58]) represents literally **two videos**. Ranking metrics (ROC-AUC, PR-AUC) with paired bootstrap confidence intervals are mandatory to avoid false discovery.

---

## 3. Structural Priorities for Milestone 13.3

### Priority A: Learned Local Frequency Residual Branch
*Distinction from M13.1*: M13.1 demonstrated that *global* spectral statistics (such as frame-wide Laplacian variance and 2D FFT energy distribution) are identical between detected and missed samples. However, **spatially localized frequency representations** can expose local interpolation, resampling, and pixel synthesis inconsistencies.

#### Architectural Implementations:
1. **Block DCT Feature Maps**:
   - Compute $8 \times 8$ local Discrete Cosine Transform across the $112 \times 112$ face crops.
   - Partition DCT coefficients into low-, mid-, and high-frequency channels to highlight localized high-frequency spatial grid artifacts.
2. **Learnable Steerable High-Pass Residuals**:
   - Constrained convolutional kernels (e.g., SRM / Bayar constrained filters) that suppress semantic facial content and force the representation to attend to high-frequency sensor noise and resampling residuals.
3. **Dual-Domain Feature Fusion**:
   - Concatenate localized spatial frequency residuals with Stage-5 intermediate spatial features before the temporal Transformer.

---

### Priority B: Local Artifact & Region-Specific Texture Branch
Face2Face modifies interior facial landmarks (mouth interior, nasolabial folds, jawline reenactment) while keeping the outer facial boundary pristine. Standard CNN backbones downsample these subtle textural cues.

#### Architectural Implementations:
1. **Multi-Region Semantic ROIs**:
   - Extract localized sub-crops focused on:
     * **Mouth / Lip region** (phoneme-expression articulation)
     * **Periorbital region / Eyes** (gaze, blink, eyelid synthesis)
     * **Outer facial boundary** (affine blending border)
2. **Texture-to-Geometry Inconsistency Module**:
   - Compute residual difference between expected facial landmark geometry and actual pixel texture gradients.
   - Test whether expression warping generates subtle local shear strains absent in natural biometric movement.

---

### Priority C: Multi-Family Leave-One-Family-Out (LOFO) Matrix
*The Most Critical Methodological Advance*: Evaluating exclusively on `Train: [DF, FS, NT] -> Test: F2F` cannot distinguish whether **Face2Face is uniquely challenging** or whether **the detector systematically overfits to the artifact signatures of seen families**.

#### The 4-Fold LOFO Experimental Matrix:

| Fold Identifier | Training Manipulation Families | Held-Out Manipulation Family | Primary Diagnostic Objective |
|---|---|---|---|
| **Fold 1 (`LOFO-F2F`)** | Deepfakes + FaceSwap + NeuralTextures | **Face2Face** | Standard held-out expression reenactment benchmark |
| **Fold 2 (`LOFO-NT`)** | Deepfakes + FaceSwap + Face2Face | **NeuralTextures** | Expression reenactment without NeuralTextures guidance |
| **Fold 3 (`LOFO-FS`)** | Deepfakes + NeuralTextures + Face2Face | **FaceSwap** | Seam-heavy identity swap without FaceSwap supervision |
| **Fold 4 (`LOFO-DF`)** | FaceSwap + NeuralTextures + Face2Face | **Deepfakes** | Autoencoder synthesis without Deepfakes supervision |

#### Hypothesis Disambiguation:
- If a candidate architecture improves generalization across **all 4 folds**, it has learned **family-invariant visual evidence**.
- If a candidate architecture only improves on Face2Face while degrading on NeuralTextures or FaceSwap, it has merely shifted family-specific inductive biases.

---

## 4. Evaluation and Statistical Protocol

To guarantee mathematical reproducibility and prevent false claims:

1. **Ranking Metrics Primary**:
   - **ROC-AUC** and **PR-AUC** are the primary decision endpoints. Threshold detection rate at $\tau=0.5$ is strictly diagnostic.
2. **Fixed-Specificity Operating Points**:
   - Report $\text{TPR}_{@\text{FPR}=0.05}$ (Sensitivity at 95% Clean Real Specificity).
   - Report $\text{TPR}_{@\text{FPR}=0.10}$ (Sensitivity at 90% Clean Real Specificity).
3. **Explicit Integer Counts**:
   - Every rate must state exact counts (e.g., $29/58$ detected, $9/98$ false alarms) to maintain transparency regarding sample sizes.
4. **Paired Bootstrap Confidence Intervals**:
   - 1,000 non-parametric bootstrap resamples for paired difference distributions ($\Delta \text{ROC-AUC}$, $\Delta \text{PR-AUC}$) with 95% CIs and two-tailed empirical $p$-values against `M13.2b-Stage5`.
5. **Multi-Seed Stability**:
   - Where model training is conducted, run across 3 fixed random seeds (`seed=42, 1337, 2026`) and report mean $\pm$ standard error.
6. **Invariance to M12 & M13.2 Checkpoints**:
   - Cryptographic verification of `checkpoints/m12_full/best.pth` (`c8265f63...`) and `checkpoints/m13_2b_stage5/best.pth` (`d6813861...`).

---

## 5. Implementation Roadmap for M13.3

```text
Milestone 13.3 Roadmap
├── Phase 1: Diagnostic Feature Probing of Manipulation Families
│   ├── Analyze linear separability of DF, FS, NT, F2F in frozen M13.1 features
│   └── Measure cross-family cosine angles in Stage-5 and penultimate spaces
├── Phase 2: LOFO Manifest Splitting Infrastructure
│   ├── Construct manifests for LOFO-F2F, LOFO-NT, LOFO-FS, LOFO-DF
│   └── Verify zero actor group overlap and strict family isolation
├── Phase 3: Local Frequency Residual Branch Implementation
│   ├── Implement spatial DCT block decomposition module
│   └── Implement learnable high-pass residual filter module
├── Phase 4: Local Texture ROI Branch Implementation
│   ├── Landmark-aligned mouth and boundary patch extractor
│   └── Local texture consistency loss
├── Phase 5: Systematic 4-Fold LOFO Evaluation & Multi-Seed Benchmarking
│   ├── Train candidate variants across all 4 LOFO folds
│   └── Paired bootstrap hypothesis testing against M13.2b-Stage5
└── Phase 6: Authoritative Reporting & Baseline Resolution
    └── Publish formal M13.3 comparative report and decision artifact
```
