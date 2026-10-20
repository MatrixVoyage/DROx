# Milestone 13.5: Consolidation & Promotion with Explicit Limitations

**Date / Timestamp**: `2026-09-27T01:15:00+00:00`  
**Execution Context**: NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, CUDA 12.4, PyTorch 2.6.0+cu124, Windows 11)  
**Permanent Model Invariants**:
- `v3-robust (M12)` Baseline Hash: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (100% Bitwise Immutable Reference)
- `M13.2b-Stage5` Baseline Hash: `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611` (Principal Comparison Reference)
- `v5-family-invariant` Champion Hash: `4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7` (Recorded & Frozen Checkpoint)
- `M13.2c` Disentanglement Negative Control: Permanently frozen (`reports/m13_2c_negative_artifact_freeze.json`)

---

## 1. Executive Summary & Strategic Milestone Decision

Following the empirical completion of Milestone 13.3 (Representation Discovery) and Milestone 13.4 (Stability, Calibration, Operating Points, Systematic Branch Ablation & 4-Fold LOFO Transfer Matrix):
- **Why Avoid Further Handcrafted Artifact Branches**: Milestone 13.4 Phase D proved that handcrafted frequency and texture branches do **not** contribute orthogonal ranking information beyond the fine-tuned temporal trunk. Standalone frequency and texture heads produce elevated false alarm rates (**28.6% and 46.9% false alarms on clean reals**). The multi-domain fusion head primarily acts as an **adaptive noise suppressor / gating filter**, preserving 93.88% clean specificity while sustaining peak 0.7903 temporal ROC-AUC.
- **Milestone Decision**: Milestone 13.5 structures the system around **consolidation and promotion with explicit limitations**, rather than engineering further handcrafted artifact branches.
- **Promotion Contract Fulfillment**: Promotion is conditioned strictly on engineering correctness, determinism, parity, score-space consistency, and rigorous limitations exposure—not on seeking unverified benchmark gains.

---

## 2. Authoritative Promotion Gate Checklist

| Gate | Requirement | Status | Verification Evidence |
|---|---|:---:|---|
| **Frozen baseline integrity** | SHA unchanged | **PASS** | M12: `c8265f63...`, M13.2b: `d6813861...` (100% bitwise immutable) |
| **Champion checkpoint integrity** | SHA recorded & frozen | **PASS** | `4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7` (`checkpoints/m13_3_f2f_seed42/best.pth`) |
| **Score space & threshold semantics** | Unambiguous score space | **PASS** | Production pipeline evaluates strictly in `temperature_scaled_probability` ($p_{\text{cal}} = \sigma(\text{logit} / 2.0502)$); $\tau^*_{\text{cal}} = \mathbf{0.1260}$ (corresponding to $\tau^*_{\text{raw}} = \mathbf{0.0190}$ in raw space) |
| **Calibration methodology** | Val-fitted vs eval diagnostics | **PASS** | $T=2.0502$ fitted by minimizing NLL on 1,255 validation sequences only; test ECE ($0.2729 \to 0.0866$) and Brier ($0.2683 \to 0.1848$) are evaluation diagnostics only |
| **Predictor & API parity** | Exact parity confirmed | **PASS** | `test_predictor_and_api_parity_and_determinism` passes (exact float match on probability, raw logit, verdict, and branch scores) |
| **Determinism** | Scoped determinism | **PASS** | Bitwise deterministic across repeated inference executions under the validated RTX 3050 / CUDA 12.4 / PyTorch 2.6.0+cu124 environment |
| **OOD status** | Diagnostic only | **PASS** | Penultimate distance threshold ($3.18$, 99th percentile across 343 validation clean reals) is diagnostic-only (`reject_on_ood = False`) |
| **Branch metadata** | Diagnostic scores, no attribution % | **PASS** | Exposed as raw probabilities ($p \in [0, 1]$), never as additive percentage contributions |
| **Regression suite** | 100% pass with exact counts | **PASS** | **Tests collected: 293, Tests passed: 293, Tests failed: 0, Tests skipped: 0**; M13.5 dedicated tests: 7/7 passed |
| **Physical E2E** | Integration smoke tests | **PASS** | Integration smoke tests on physical videos: Real `004.mp4` (P(Fake)=0.03%, Model Score=0.0003, Real) and Fake `004_982.mp4` (P(Fake)=99.97%, Model Score=0.9997, Deepfake) |
| **Documentation & limitations** | Strict provenance & disclosures | **PASS** | Strict separation of single checkpoint vs 4-fold matrix; 5-point disclosure banner in UI, `/api/v1/model-info`, and docs |

---

## 3. Resolution of Score-Space and Threshold Semantics

### The Production Chain
The production decision pipeline evaluates in a single, unambiguous chain:
$$\text{Video} \xrightarrow{\quad} \text{Raw Logit } z \xrightarrow{\quad} \text{Temperature Scaling } (T = 2.0502) \xrightarrow{\quad} p_{\text{cal}} = \sigma\left(\frac{z}{2.0502}\right) \xrightarrow{\quad} \text{Threshold } \tau^*_{\text{cal}} = 0.1260 \xrightarrow{\quad} \text{Real / Deepfake}$$

### Exact Mathematical Correspondence Between Score Spaces
Derived across all 1,255 validation sequences and evaluated on the fixed 560-query benchmark:

| Operating Policy | Calibrated Space ($p_{\text{cal}} = \sigma(z / 2.0502)$) | Raw Sigmoid Space ($p_{\text{raw}} = \sigma(z)$) | Test Face2Face Recall (Unseen Family) | Test Clean Specificity (FF++ Clean Reals) |
|---|:---:|:---:|:---:|:---:|
| **Standard Policy** | $\tau = \mathbf{0.5000}$ | $\tau = \mathbf{0.5000}$ | 22/58 (37.93%) | 92/98 (93.88%, 6 false alarms) |
| **High-Sensitivity Policy ($\text{Spec} \ge 90\%$)** | $\tau^*_{\text{cal}} = \mathbf{0.1260}$ | $\tau^*_{\text{raw}} = \mathbf{0.0190} \approx \mathbf{0.0200}$ | **25/58 (43.10%)** | **92/98 (93.88%, 6 false alarms)** |
| **Youden-Optimal Policy ($\max(\text{TPR} - \text{FPR})$)** | $\tau_{\text{youden}} = \mathbf{0.1912}$ | $\tau_{\text{youden}} = \mathbf{0.0494}$ | 24/58 (41.38%) | 92/98 (93.88%, 6 false alarms) |
| **High-Specificity Policy ($\text{Spec} \ge 95\%$)** | $\tau_{\text{spec95}} = \mathbf{0.6820}$ | $\tau_{\text{spec95}} = \mathbf{0.8270}$ | 19/58 (32.76%) | 94/98 (95.92%, 4 false alarms) |

> [!IMPORTANT]
> **Monotonic Invariance**: Because temperature scaling is a strictly monotonic transformation ($p_{\text{cal}} = \sigma(z/T)$ with $T > 0$), applying $\tau^*_{\text{cal}} = 0.1260$ in calibrated space yields **identical** test classifications to applying $\tau^*_{\text{raw}} = 0.0190$ in raw space. Production inference evaluates strictly in `temperature_scaled_probability` using $\tau^*_{\text{cal}} = 0.1260$, eliminating previous ambiguity.

### Why Temperature Scaling Over Isotonic Regression
- **Temperature Scaling ($T = 2.0502$)**: Single-parameter validation-fitted scaling that is strictly smooth, monotonic, preserves all ranking information, and prevents overfitting empirical step-function bin edges outside validation density.
- **Isotonic Regression**: Achieved test ECE of 0.0619, but acts as a non-smooth piecewise constant mapping prone to step artifacts on out-of-distribution inputs. It is retained strictly as an exploratory research diagnostic.

---

## 4. Production Engine Hierarchy & Strict Provenance Separation

> [!IMPORTANT]
> **Strict Provenance Separation**:
> - **Single Deployed Checkpoint**: `checkpoints/m13_3_f2f_seed42/best.pth` was trained under the **LOFO-F2F fold** (seen: Deepfakes, FaceSwap, NeuralTextures; held-out: Face2Face). Its individual test ROC-AUC on unseen Face2Face is **0.7903** (clean specificity: 93.88%).
> - **Research Family-Transfer Study**: The **0.9165 Macro Mean AUC** is the cross-family aggregate across all 4 independently trained LOFO checkpoints in the Milestone 13.4 transfer matrix, **not** an attribute of this single deployed checkpoint.

| Engine ID | Architecture Details | Checkpoint Path | Research Provenance | Status |
|---|---|---|---|---|
| **`v1-baseline`** | ResNeXt-50 + Soft Attention + LSTM | `checkpoints/ffpp_c23_baseline_soft_lstm/best.pth` | Lal et al. (2025) Reference Baseline | Frozen Reference |
| **`v2-evolved`** | EfficientNet-B0 + Temporal Transformer | `checkpoints/m10_selected_candidate/best.pth` | M10 Architecture Evolution Candidate | Frozen Reference |
| **`v3-robust`** | EfficientNet-B0 + Robust Augmentor + Transformer | `checkpoints/m12_full/best.pth` | M12 Generalization Champion (`c8265f63...`) | Frozen Immutable Reference |
| **`v4-generalization`**| EfficientNet-B0 + Stage-5 MidHook + AuxHead | `checkpoints/m13_2b_stage5/best.pth` | M13.2b Generalization Baseline (`d6813861...`) | Established Baseline Reference |
| **`v5-family-invariant`**| Multi-Domain Fusion (Temporal + Freq + Texture) | `checkpoints/m13_3_f2f_seed42/best.pth` | M13.4 LOFO-F2F Checkpoint (Single Fold AUC: **0.7903**; 4-Fold LOFO Matrix Mean: **0.9165**) | **Promoted Research Champion for the M13 Family-Invariant Architecture (Production Research Engine)** |

### Comparative Trade-offs: Promoted v5 vs M13.2b Baseline
The promoted v5 checkpoint is a specialized research champion, not an unconstrained universal winner across every single sub-metric:
- **Face2Face ROC-AUC**: v5 achieves **0.7903** vs M13.2b's 0.7830 (+0.0073 advantage).
- **Face2Face Detection Rate at $\tau=0.50$**: M13.2b achieves 30/58 (51.7%) vs v5's 22/58 (37.9%) at standard threshold, while v5 recovers 25/58 (43.1%) at validation-derived operating threshold $\tau^* = 0.1260$.
- **Deepfakes & NeuralTextures Transfer**: v5 achieves higher transfer AUCs: Deepfakes 0.9824 vs 0.9806 (100% vs 98.3% detection) and NeuralTextures 0.9499 vs 0.9244 (89.7% vs 84.5% detection).
- **FaceSwap Transfer**: M13.2b achieves slightly higher AUC: 0.9474 vs v5's 0.9435 (91.4% vs 89.7% detection).

---

## 5. Diagnostic-Only OOD and Non-Attributional Branch Outputs

### Out-of-Distribution (OOD) Status
In the API and UI, OOD is strictly an informational diagnostic:
```json
{
  "ood_status": "diagnostic",
  "ood_distance": 2.14,
  "ood_reference": "validation_clean",
  "ood_threshold": 3.18,
  "reject_on_ood": false
}
```
The threshold of $3.18$ represents the 99th percentile of Mahalanobis distances across 343 validation clean reals. Because deepfake artifacts can push representations outside the clean-real centroid, setting autonomous rejection on OOD causes false rejections on authentic deepfake manipulations. For production inference, OOD is reported as an informational metric without overriding predictions (`reject_on_ood = False`).

### Multi-Domain Branch Scores (Non-Attributional)
Branch outputs are exposed strictly as independent probabilities from specialized heads:
```json
{
  "p_fused": 0.9997,
  "p_composite": 0.8659,
  "p_freq": 0.3675,
  "p_texture": 0.5611,
  "p_aux_fake": 1.0000
}
```
They are **not** additive contributions and do not sum to 100%. Reporting percentage attributions is strictly prohibited.

---

## 6. The Five Explicit Operational & Research Limitations

In compliance with **Rule 10 (Research Integrity)** and **Rule 18 (UI Rules)**, `v5-family-invariant` is certified with the following formal disclosures:

### 1. Facial Reenactment Manifold Boundary
While synthesis techniques with physical boundary splices (Deepfakes, FaceSwap) and rendering noise (NeuralTextures) are detected at 89.7%–100.0%, subtle reenactment (Face2Face) achieves 37.9% detection at standard threshold $\tau=0.50$ (43.1% at validation operating threshold $\tau^* = 0.1260$). 3DMM facial reenactment modifies expression parameters inside the natural human facial subspace without leaving boundary seams.

### 2. Handcrafted Branch Gating vs Orthogonal Evidence
Standalone frequency and texture branches function as noisy anomaly detectors (71.4% and 53.1% clean specificity). The fused architecture achieves high specificity (93.88%) by learning to heavily gate/suppress branch inputs when contradictory to the fine-tuned temporal representation, rather than providing independent orthogonal certainty.

### 3. Threshold vs Specificity Operating Trade-Off
Detecting subtle unseen manipulations requires selecting an operating point ($\tau^*_{\text{cal}}=0.1260$) derived strictly from validation clean reals. While this increases Face2Face recall from 22/58 (37.9%) to 25/58 (43.1%), it permits a 6.12% false alarm rate (6/98 clean reals misclassified).

### 4. Cross-Dataset Domain Shift
Cross-domain transfer to Celeb-DF v2 yields ROC-AUC of $0.7965 \pm 0.0126$, showing expected performance degradation relative to in-domain seen synthesis ($0.98+$ AUC).

### 5. Probabilistic Confidence Semantics
Model scores represent statistical confidence within the validated distribution, not objective physical proof or sole legal evidence. The UI presents `Model Score: 0.9997` and `P(Fake): 99.97%`, strictly avoiding terminology such as `Confidence: 99.97%` that would imply epistemic certainty.

---

## 7. Service Integration, Test Auditing & Parity Verification

- **Inference Service**: `VideoPredictor` dynamically instantiates `FamilyInvariantDeepfakeDetector` and evaluates calibrated forward passes with deterministic logits.
- **Physical Video Integration Smoke Tests**:
  - Real Video (`Datasets/FaceForensics++_C23/original/004.mp4`): Verdict `Real`, Decision `REAL`, Model Score `0.0003`, Calibrated P(Fake) `0.03%`, OOD Status `Within validated range (2.14)`.
  - Fake Video (`Datasets/FaceForensics++_C23/Deepfakes/004_982.mp4`): Verdict `Deepfake`, Decision `FAKE`, Model Score `0.9997`, Calibrated P(Fake) `99.97%`, Branch Scores: $p_{\text{fused}}=0.9997, p_{\text{comp}}=0.8659, p_{\text{freq}}=0.3675, p_{\text{tex}}=0.5611, p_{\text{aux}}=1.0000$.
- **FastAPI Endpoints**: `/health`, `/api/v1/model-info`, `/api/v1/predict`, and `/` operational and returning 200 with full engine suite.
- **Hardware Profile**: Peak VRAM on NVIDIA RTX 3050 < 600 MB (<10% of 6GB ceiling).
- **Exact Regression Test Audit**:
  - **Tests collected: 293**
  - **Tests passed: 293**
  - **Tests failed: 0**
  - **Tests skipped: 0**
  - **M13.5-specific tests: 7/7 passed** (`tests/test_m13_5_promotion.py`)
- **Determinism Scope**: Bitwise deterministic across repeated inference executions under the validated RTX 3050 / CUDA 12.4 / PyTorch 2.6.0+cu124 / Windows environment.
