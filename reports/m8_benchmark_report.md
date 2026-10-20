# Milestone 8 Consolidated Benchmark & Robustness Report

**Execution Date**: 2026-09-21 16:24:05  
**Environment**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.0 GB VRAM)  
**Software**: PyTorch 2.6.0+cu124 | CUDA 12.4  
**Checkpoint Status**: Frozen (`9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a`) — Verified Untouched  

## 1. Full-Scale Benchmark Comparison (Gate 1)

| Benchmark | Split | Evaluated Videos | Real / Fake | Accuracy | ROC-AUC | EER |
|---|---|---|---|---|---|---|
| **FF++ C23 (In-Domain)** | Canonical Test (75 groups) | 750 | 150 / 600 | 0.8000 | 0.4734 | 0.5133 |
| **Celeb-DF v2 (Cross-Dataset)** | Official Test Split | 518 | 178 / 340 | 0.6564 | 0.5244 | 0.5029 |

### Cross-Dataset Generalization Drop

- **Δ AUC**: `+0.0509`
- **Δ EER**: `-0.0104`

## 2. Manipulation Family Breakdown (Gate 2)

| Manipulation Family | Fake Samples | Pristine Samples | Accuracy | ROC-AUC | EER |
|---|---|---|---|---|---|
| **Deepfakes** | 150 | 150 | 0.5000 | 0.4447 | 0.5267 |
| **Face2Face** | 150 | 150 | 0.5000 | 0.5121 | 0.5000 |
| **FaceSwap** | 150 | 150 | 0.5000 | 0.4710 | 0.5267 |
| **NeuralTextures** | 150 | 150 | 0.5000 | 0.4638 | 0.5133 |

## 3. Probability Calibration & Optimal Threshold Policies (Gate 3)

- **Fitted Temperature**: `T = 1.148` (fitted strictly on validation logits)
- **Youden's J Threshold**: `θ = 0.8915` (Validation J = 0.0667)
- **F1-Maximizing Threshold**: `θ = 0.01` (Validation F1 = 0.8889)
- **Validation ECE**: `0.1081` → `0.1020`

### FF++ Test Split Under Different Threshold Policies

| Policy | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| `threshold_0.50_baseline` | 0.5 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.4734 |
| `threshold_youden_0.8915` | 0.8915 | 0.4493 | 0.7842 | 0.4300 | 0.5554 | 0.4734 |
| `threshold_f1max_0.01` | 0.01 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.4734 |
| `calibrated_prob_threshold_0.50` | 0.5 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.4734 |

## 4. Video Robustness under Controlled Perturbations (Gate 4)

| Perturbation | Description | Accuracy | ROC-AUC | Δ AUC vs Baseline |
|---|---|---|---|---|
| **Baseline (Clean)** | Unperturbed 112x112 baseline | 0.5000 | 0.3867 | `+0.0000` |
| **Spatial 56x56** | Downsample 56x56 -> Upsample 112x112 | 0.5000 | 0.4578 | `+0.0711` |
| **Spatial 28x28** | Downsample 28x28 -> Upsample 112x112 | 0.5000 | 0.5289 | `+0.1422` |
| **Gaussian Blur s=0.5** | Gaussian blur sigma=0.5 | 0.5000 | 0.3733 | `-0.0133` |
| **Gaussian Blur s=1.0** | Gaussian blur sigma=1.0 | 0.5000 | 0.4444 | `+0.0578` |
| **Gaussian Blur s=1.5** | Gaussian blur sigma=1.5 | 0.5000 | 0.4444 | `+0.0578` |
| **Gaussian Noise s=0.05** | Additive noise std=0.05 | 0.5000 | 0.3822 | `-0.0044` |
| **Gaussian Noise s=0.10** | Additive noise std=0.10 | 0.5000 | 0.4933 | `+0.1067` |
| **Contrast Red. f=0.50** | 50% Contrast reduction | 0.5000 | 0.5111 | `+0.1244` |
| **Contrast Red. f=0.25** | 75% Contrast reduction | 0.5000 | 0.5644 | `+0.1778` |
| **JPEG Quality 30** | Simulated block compression Q=30 | 0.5000 | 0.3422 | `-0.0444` |
| **JPEG Quality 10** | Simulated block compression Q=10 | 0.5000 | 0.3822 | `-0.0044` |

## 5. Systematic Error Analysis (Gate 5)

- Total Test Samples Evaluated: `1268`
- False Positive Count (Real misclassified as Fake): `328` (FPR: `1.0000`)
- False Negative Count (Fake misclassified as Real): `0` (FNR: `0.0000`)

### Face Detection Associations (Strictly Non-Causal)

- **mtcnn_confidence**: Empirical data shows mtcnn_confidence is negatively correlated (r=-0.045) with prediction error; mean for correct predictions is 0.950 vs 0.950 for error cases. This observed relationship does not establish direct causality.
- **face_bounding_box_area**: Empirical data shows face_bounding_box_area is negatively correlated (r=0.000) with prediction error; mean for correct predictions is 4900.000 vs 4900.000 for error cases. This observed relationship does not establish direct causality.
- **face_aspect_ratio**: Empirical data shows face_aspect_ratio is negatively correlated (r=0.000) with prediction error; mean for correct predictions is 1.000 vs 1.000 for error cases. This observed relationship does not establish direct causality.
- **fallback_detector_ratio**: Empirical data shows fallback_detector_ratio is negatively correlated (r=0.000) with prediction error; mean for correct predictions is 0.000 vs 0.000 for error cases. This observed relationship does not establish direct causality.

## 6. Spatial Attention Attribution (Gate 6)

- Attention Tensor Shape: `[16, 4, 4]`
- Method: Soft spatial attention 1x1 convolution bottleneck introspection
- Note: Attention weights reflect intermediate spatiotemporal feature weighting within the ResNeXt-50 to LSTM bottleneck; they do not represent verified pixel-level manipulation masks.

## 7. DFDC Standing Gate (Gate 7)

- Status: **PENDING**
- Reason: PENDING — DATA ACCESS UNAVAILABLE. Dataset requires official competition credentials.

## 8. LaTeX Publication Snippet

```latex
\begin{table}[ht]
\centering
\small
\begin{tabular}{lccccc}
\toprule
Benchmark & Split & $N$ & Acc (\%) & ROC-AUC & EER (\%) \\
\midrule
FF++ C23 & Canonical Test & 750 & 80.00 & 0.4734 & 51.33 \\
Celeb-DF v2 & Official Test & 518 & 65.64 & 0.5244 & 50.29 \\
\bottomrule
\end{tabular}
\caption{Benchmark evaluation of frozen ResNeXt-50 + Soft Attention + LSTM detector.}
\label{tab:benchmark_results}
\end{table}
```
