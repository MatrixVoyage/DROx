# Milestone 14 Phase F: Architecture Decision Memo & M15 Preregistered Hypotheses

**Date / Timestamp**: `2026-09-27T02:05:00+00:00`  
**Author**: Research & Coding Agent  
**Subject**: Formal Architectural Decision on Milestone 15 Feasibility  
**Frozen Evaluation Target**: `v5-family-invariant` (`checkpoints/m13_3_f2f_seed42/best.pth`, SHA-256: `4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7`)  
**Guiding Non-Modification Rule**: Milestone 14 serves strictly to measure, characterize, and generate hypotheses—not to modify M13.5 or iteratively select M15 hyperparameters against holdout test data.

---

## 1. Executive Summary & Research Question

Milestone 14 addressed the primary research question:
> *Can the frozen M13.5 family-invariant detector maintain calibrated, low-false-alarm behavior under unseen datasets and perturbations, and what measurable failure mechanism explains the remaining Face2Face reenactment gap?*

Through five systematic, strictly read-only empirical investigations:
1. **Phase A0 & A**: Application stack verified and frozen; 100% bitwise hash integrity verified across M12, M13.2b, and v5 champion checkpoints.
2. **Phase B**: Independent external evaluation on the official Celeb-DF v2 benchmark test set (518 videos: 340 deepfakes, 178 reals).
3. **Phase C**: In-depth forensic failure characterization across the 58 Face2Face holdout queries.
4. **Phase D**: Controlled robustness stress testing across 23 One-Factor-At-A-Time (OFAT) perturbation conditions on the 560-query benchmark.
5. **Phase E**: Formal statistical uncertainty quantification (1,000-resample bootstrap CIs and paired hypothesis testing vs M13.2b reference with Holm-Bonferroni correction).

---

## 2. Synthesis of Empirical Findings Across M14 Phases

### Finding 1: Cross-Dataset Generalization & Calibration Stability (Phase B)
- On the official 518-video Celeb-DF v2 test benchmark, `v5-family-invariant` achieves stable generalization without retraining.
- Frozen temperature scaling ($T=2.0502$) generalizes out-of-domain, effectively reducing calibration error.
- The high-sensitivity operating point ($\tau^*=0.1260$) derived strictly on validation clean reals maintains elevated specificity while improving cross-dataset recall.

### Finding 2: The Face2Face Failure Mechanism (Phase C)
- **Within-Manifold Confinement**: The high-frequency edge energy (Laplacian variance) of missed Face2Face reenactments is statistically indistinguishable from pristine authentic clean reals. 3DMM expression modifications generate smooth deformations that avoid physical blending seams.
- **Fusion Head Gating**: The auxiliary multi-family competition head recognizes anomalies in a substantial fraction of missed queries ($p_{\text{aux}} \ge 0.50$), but the static MLP fusion head suppresses the primary decision score because the base temporal trunk predicts authentic.

### Finding 3: Perturbation Resilience & Semantic Real-World Shifts (Phase D)
- The detector demonstrates strong ranking resilience against mild-to-moderate JPEG compression (Q=50), spatial resizing (0.75x, 0.50x), and photometric shifts.
- Under extreme compression (CRF 42, JPEG Q=25), false alarms on authentic reals increase due to compression blocking artifacts triggering the frequency head. This represents a known physical property of lossy video codecs rather than pure model fragility.
- Temporal stride subsampling confirms the model's reliance on inter-frame temporal continuity.

### Finding 4: Statistical Significance and Estimation Noise (Phase E)
- **Face2Face**: The $\Delta \text{AUC} = +0.0107$ delta over M13.2b has a 95% bootstrap confidence interval (`[-0.0310, +0.0494]`, $p = 0.5800$) crossing zero. Under formal paired testing, there is **insufficient statistical evidence** to reject the null hypothesis of equal ranking performance between v5 and M13.2b on Face2Face alone. However, the validation-derived operating point ($\tau^*=0.1260$) recovers 43.10% recall at 93.88% clean specificity.
- **Deepfakes & NeuralTextures**: Deepfakes shows $\Delta \text{AUC} = +0.0157$ (raw $p=0.0320$, crossing Holm-Bonferroni adjusted $\alpha=0.0125$) and NeuralTextures shows $\Delta \text{AUC} = +0.0055$ (95% CI: `[-0.0101, +0.0202]`, $p=0.4700$). Under the pre-specified multiple testing correction, all family-level pairwise ranking differences represent insufficient evidence to establish unconditional superiority on these sample sizes.
- **Celeb-DF v2**: External transfer on 518 queries yields $\Delta \text{AUC} = -0.0154$ (95% CI: `[-0.0463, +0.0174]`, $p=0.3880$), also indicating insufficient statistical evidence of a difference.

---

## 3. Architecture Decision Tree Evaluation

```text
                                M13.5 Frozen Engine
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
       Face2Face Failure Analysis             Perturbation & External Datasets
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                           Synthesized Evidence Base
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
[Branch 1: Temporal Deficit]   [Branch 2: Domain Shift]    [Branch 3: Ambiguous / Noise]
Evidence: Frame dynamics are   Evidence: Feature space     Evidence: No clear failure
masked by global pooling;      collapses across datasets;   mode; delta vs baseline
inter-frame delta informative. spatial features robust.     statistically insignificant.
         │                               │                               │
         ▼                               ▼                               ▼
Generate M15 Hypothesis:       Generate M15 Hypothesis:    Retain Frozen v5 Engine;
Attention-Gated Spatial-       Domain-Invariant Training   Do NOT Add Complexity
Temporal Modeling
```

### Evaluation of Decision Branches:
1. **Branch 1 (Temporal & Spatial Integration Deficit) — SUPPORTED FOR HYPOTHESIS GENERATION**:
   - The forensic analysis indicates that the temporal trunk dominates the static fusion head, suppressing auxiliary spatial signals when temporal features lack blending seams.
   - *Verdict*: Justifies formulating an M15 research hypothesis around **attention-gated cross-modal feature integration**.
2. **Branch 2 (Domain Shift) — PARTIALLY SUPPORTED**:
   - Cross-dataset evaluation confirms expected domain gap on Celeb-DF v2, but the gap is stable and bounded.
3. **Branch 3 (Complexity Restraint) — ACTIVE GOVERNING CONSTRAINT**:
   - Because the Face2Face AUC delta is within bootstrap estimation bounds, immediate hyperparameter tuning or arbitrary branch additions are strictly barred.

---

## 4. Preregistered Research Hypotheses for Milestone 15

If an architectural exploration is authorized for Milestone 15, it must be governed by the following preregistered hypotheses and protocol:

### Preregistered Hypothesis H-M15-1: Attention-Gated Cross-Modal Fusion
- **Hypothesis**: Replacing the static concatenation MLP with a cross-attention gating mechanism (where temporal queries attend to Stage-5 spatial keys/values) will prevent the temporal trunk from unconditionally suppressing localized facial expression anomalies.
- **Preregistration Constraint**: Training must occur on a newly partitioned training split with independent validation; the M13/M14 Face2Face test benchmark will remain strictly a read-only evaluation holdout.

### Preregistered Hypothesis H-M15-2: Multi-Scale Temporal Phase Modeling
- **Hypothesis**: Hierarchical short-window (2-4 frame) temporal self-attention will capture transient expression phase transitions (e.g. phoneme and blink desynchronization) that uniform 16-frame self-attention smooths out.

---

## 5. Non-Modification & Integrity Attestation
1. `v5-family-invariant` was evaluated strictly read-only throughout Milestone 14.
2. Zero model weights, thresholds, calibration parameters, or preprocessing options were altered.
3. The Face2Face benchmark was preserved as a pristine holdout evaluation set.
