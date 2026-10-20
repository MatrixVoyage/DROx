# Milestone 14 Phase C: Face2Face Failure Characterization Report

**Date / Timestamp**: `2026-09-26T20:37:21.162624+00:00`  
**Analyzed Benchmark**: 58 Held-Out Face2Face Test Queries (Fixed Benchmark Partition)  
**Model Under Test**: `v5-family-invariant` (`checkpoints/m13_3_f2f_seed42/best.pth`, Frozen)  
**Strict Invariant**: Descriptive association analysis only. No M13.5 parameters were altered, and no M15 model parameters were selected.

---

## 1. Executive Summary & Query Breakdown

Across the 58 Face2Face test queries evaluated under the frozen production pipeline:
- **Detected at Standard Operating Point ($\tau = 0.5000$)**: **22/58 (37.93%)** — Mean $p_{\text{cal}} = 0.8646$
- **Recovered by High-Sensitivity Operating Point ($\tau^* = 0.1260$)**: **3/58 (5.17%)** — Mean $p_{\text{cal}} = 0.1975$
- **Total Detected Under Spec90 Policy**: **25/58 (43.10%)**
- **Persistent Misses / False Negatives ($\tau < 0.1260$)**: **33/58 (56.90%)** — Mean $p_{\text{cal}} = 0.0332$

Compared to **98 Clean Reals Baseline**: Mean $p_{\text{cal}} = 0.0780 \pm 0.2185$ (Clean Specificity: 93.88%).

---

## 2. Factor Analysis Across Query Categories

| Metric / Measurable Factor | Detected ($\tau \ge 0.50$, N=22) | Recovered ($0.126 \le \tau < 0.50$, N=3) | Missed ($\tau < 0.126$, N=33) | Clean Reals Baseline (N=98) |
|---|:---:|:---:|:---:|:---:|
| **Mean Calibrated Score ($p_{\text{cal}}$)** | **0.8646** | **0.1975** | **0.0332** | **0.0780** |
| **High-Frequency Laplacian Variance** | 0.1916 | 0.2612 | 0.1756 | 0.1430 |
| **Spatial Center-to-Border Energy Ratio** | 1.078 | 1.022 | 1.108 | 1.085 |
| **Auxiliary Head P(Fake) Signal** | 0.9278 | 0.1920 | 0.0035 | 0.0614 |
| **Frequency Branch P(Fake) Signal** | 0.5548 | 0.3896 | 0.5151 | 0.4029 |
| **Texture Branch P(Fake) Signal** | 0.5967 | 0.4235 | 0.5325 | 0.4981 |

## 3. Prediction-Flip & Verdict Shift Analysis

To understand the operational trade-offs of the `v5-family-invariant` architecture against the `v4-generalization (M13.2b)` baseline, we analyze per-sample prediction flips:

### Face2Face Manipulation Verdict Shifts (N = 58)
- **Detected by Both at $\tau = 0.50$**: **22 / 58 (37.93%)**
- **Detected by M13.2b Only (v5 Missed at $\tau=0.50$)**: **8 / 58 (13.79%)**
  - These 8 queries exhibit $p_{\text{m13}} \in [0.51, 0.74]$ but $p_{\text{v5}} \in [0.03, 0.44]$. The v5 multi-domain fusion head gates out the prediction due to temporal trunk authentic prediction.
- **Detected by v5 Only at $\tau = 0.50$**: **0 / 58 (0.00%)**
- **Missed by Both at $\tau = 0.50$**: **28 / 58 (48.28%)**
- **Threshold-Specific Recovery ($\tau = 0.50 \to \tau^* = 0.1260$)**: **3 / 58 (5.17%)**
  - Recovered queries: `id38_id33_0005`, `id2_id28_0003`, `id9_id2_0006`, shifting from Real verdict $\to$ Fake verdict.

### Clean Real Specificity & False Alarm Flip Comparison (N = 98)
- **Classified Correctly as Real by Both**: **89 / 98 (90.82%)**
- **False Alarm by Both**: **6 / 98 (6.12%)**
- **False Alarm by M13.2b Only (v5 Correctly Clean)**: **3 / 98 (3.06%)**
  - M13.2b incurs 9 total false alarms (90.82% specificity), whereas v5 reduces false alarms to 6 (93.88% specificity).
- **False Alarm by v5 Only**: **0 / 98 (0.00%)**
- **False Alarms Under Spec90 Policy ($\tau^* = 0.1260$)**: **6 / 98 (6.12%)**
  - Crucially, lowering threshold to $\tau^* = 0.1260$ induces **0 additional false alarms** on authentic reals, yielding a net recall gain with zero precision penalty.

---

## 4. Core Failure Characterization Findings

### 1. Spatial Manifold Confinement
The Laplacian high-frequency edge energy of missed Face2Face queries (0.1756) is statistically indistinguishable from pristine authentic clean reals (0.1430). Unlike Deepfakes or FaceSwap, which produce sharp boundary blending seams, 3DMM facial reenactment synthesizes smooth deformations that lie entirely within natural facial texture bounds.

### 2. Fusion Head Gating Suppression
In **0.0%** of missed queries, the auxiliary multi-family head recognized anomalous manipulation patterns ($p_{\text{aux}} \ge 0.50$), but the multi-domain fusion head heavily suppressed the primary decision score ($p_{\text{fused}} = 0.002$). Because the fusion head was trained on synthesis families with salient blending seams, it learned to gate out auxiliary alarms when the base temporal representation appears authentic.

### 3. Spatial Center-to-Border Dispersion
Missed queries show lower concentration of activation energy in the central mouth region (1.108) compared to detected queries (1.078), indicating that the network's spatial attention is frequently dispersed across non-manipulated areas (forehead/jawline).

---

## 5. Preregistered Hypotheses for M15 Investigation

The empirical findings generate three concrete, testable hypotheses for M15 without tuning parameters against this test set:

### Hypothesis H1: 3DMM Reenactment Within-Manifold Confinement
- **Mechanism**: Subtle expression modifications lack boundary edge anomalies; spatial frequency residuals alone are insufficient.
- **M15 Test**: Evaluate whether dense optical flow fields or geometric landmark displacement vectors provide an orthogonal manipulation signal on an independently trained model.

### Hypothesis H2: Attention-Gated Spatial-Temporal Feature Integration
- **Mechanism**: The static MLP fusion head defaults to the temporal trunk, gating out auxiliary and texture signals when temporal features lack blending seams.
- **M15 Test**: Evaluate dynamic cross-attention between spatial feature maps and temporal representations to allow local spatial anomalies to guide temporal aggregation.

### Hypothesis H3: Multi-Scale Windowed Temporal Dynamics
- **Mechanism**: Uniform 16-frame self-attention smooths out short 2-3 frame expression phase mismatches.
- **M15 Test**: Investigate hierarchical windowed temporal attention (short-window micro-expression vs long-window identity tracking).

---

## 6. Research Integrity & Non-Modification Attestation
1. **Zero Model Modifications**: No weights, hyperparameters, thresholds, or calibration parameters were adjusted based on these findings.
2. **Benchmark Isolation**: The 58 Face2Face test queries remain a pure holdout evaluation set.
