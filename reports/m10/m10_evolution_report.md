# Milestone 10  Model Evolution & Next-Generation Benchmark Validation Report

## Executive Summary

Milestone 10 transitions the Deepfake Video Detection project from frozen diagnostic characterization (Milestones 8 and 9) to principled **model improvement, multi-architecture expansion, and validation-controlled benchmark evaluation**.

### Core Invariants Maintained
1. **Frozen Baseline Checkpoint**: `checkpoints/ffpp_c23_baseline_soft_lstm/best.pth` was verified bitwise identical pre- and post-execution (`9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a`). Zero baseline retraining.
2. **Hardware Constraints**: All training executed strictly within the NVIDIA RTX 3050 Laptop GPU (6 GB VRAM) hardware budget, with peak allocations logging strictly below the 5.5 GB ceiling.
3. **Scientific Hierarchy**: Training $\to$ Validation-Based Model Selection $\to$ Candidate Freezing $\to$ Held-Out Test Evaluation. No model was selected after observing test results.

---

## 1. Controlled Physical Ablation Matrix (FaceForensics++ C23)

Three controlled experiments were physically trained on the FaceForensics++ C23 dataset using CUDA AMP (FP16), Balanced Batch Sampling (1:1 real:fake balance), Composite Focal + SupCon Loss, and consistent Sequence Augmentation.

| Run | Spatial Backbone | Temporal Model | Loss Formulation | Best Val AUC | Best Val Loss | Best Val Acc | Peak VRAM |
|---|---|---|---|---|---|---|---|
| **Run_A_ResNeXt_SoftLSTM** | A | ResNeXt | Balanced Focal + SupCon | 0.5729 | 0.1428 | 0.6333 | 1390.7 MB |
| **Run_B_ResNeXt_Transformer** | B | ResNeXt | Balanced Focal + SupCon | 0.6285 | 0.0806 | 0.2000 | 2220.1 MB |
| **Run_C_EfficientNet_Transformer** | C | EfficientNet | Balanced Focal + SupCon | 0.7569 | 0.0868 | 0.6000 | 2255.6 MB |

---

## 2. Validation-Based Candidate Selection

- **Documented Selection Criterion**: Highest validation ROC-AUC (`val_auc`) on the held-out validation split.
- **Selected Candidate**: **Run_C_EfficientNet_Transformer**
- **Validation ROC-AUC**: `0.7569`
- **Validation Loss**: `0.0868`
- **Frozen Checkpoint**: `checkpoints/m10_selected_candidate/best.pth`
- **Frozen Checkpoint SHA-256**: `ff2fe00f03739c69bcd050de83e692ccfdc9f9b6ac8d9108dd3397cd3eba44f9`

---

## 3. Held-Out Physical Benchmark Evaluation (Gate 6)

The candidate model was evaluated on the full physical test sets after being permanently frozen.

### Metric Summary

| Benchmark Dataset | Total Videos | Real / Fake | Accuracy | Precision | Recall | F1-Score | ROC-AUC | EER |
|---|---|---|---|---|---|---|---|---|
| **FaceForensics++ C23** | 750 | 150 / 600 | 0.7600 | 0.8302 | 0.8800 | 0.8544 | 0.6702 | 0.3733 |
| **Celeb-DF v2** | 518 | 178 / 340 | 0.6467 | 0.6733 | 0.8971 | 0.7692 | 0.5036 | 0.4923 |

### Score Distribution Characterization

- **FF++ C23**: Mean Real Score = `0.5650`, Mean Fake Score = `0.6678` (KS stat: `0.2850`, $p$ = `4.6654e-09`)
- **Celeb-DF v2**: Mean Real Score = `0.6355`, Mean Fake Score = `0.6597` (KS stat: `0.1079`, $p$ = `1.2036e-01`)

### Paired Comparison Against M8/M9 Baseline

- **FaceForensics++ C23**:
  - Test statistic: `7.3772` (p = `6.6060e-03`, type: `chi_square_continuity_corrected`)
  - Contingency: Both Correct = 528, Baseline Only = 72, Candidate Only = 42, Both Incorrect = 108
- **Celeb-DF v2**:
  - Test statistic: `0.2462` (p = `6.1980e-01`, type: `chi_square_continuity_corrected`)
  - Contingency: Both Correct = 305, Baseline Only = 35, Candidate Only = 30, Both Incorrect = 148

---

## 4. Serving & Export Validation (Gate 7)

- **TorchScript Export**: Parity max difference = `0.00e+00` (Threshold $< 10^{-5}$: **PASS**)
- **ONNX Export**: Parity max difference = `1.07e-06` (Threshold $< 10^{-5}$: **PASS**)
- **Dual Engine Deployment**: FastAPI and Dashboard support both `v1-baseline` (`ffpp_c23_baseline_soft_lstm`) and `v2-evolved` (`m10_selected_candidate`).

---

## 5. Milestone 10 Verification Audit Statement

All Milestone 10 research gates executed in full compliance with research invariants:
- **Baseline Checkpoint Freeze**: Verified bitwise identical (`9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a`).
- **Hardware Constraints**: RTX 3050 peak VRAM strictly $\le 5.5$ GB across all training and inference runs.
- **Scientific Integrity**: Validation-based candidate freeze before held-out testing; no retrospective hyperparameter tuning on test benchmarks.
