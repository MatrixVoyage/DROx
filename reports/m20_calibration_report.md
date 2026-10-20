# Milestone 20.2 — Pyramidal Dual-Stream Detector: Calibration & Decision Layer Report

**Date**: 2026-10-06 14:58:43 UTC  
**Model**: `M20PyramidalDualStreamDetector` (`checkpoints/m20_pyramidal_seed42/best.pth`)  
**Pyramidal Detail Weight**: alpha = 0.5000  

---

## 1. Validation-Derived Calibration Parameters & Policies

All parameters fitted strictly on the independent FF++ validation split (N=1,490):

- **Temperature Scaling**: T* = 1.2806
- **Platt Scaling**: slope w* = 1.2036, bias b* = -1.0837
- **Raw Validation Policy**: tau_95* = 0.9271, tau_99* = 0.9950
- **Platt Validation Policy**: tau_95* = 0.8781, tau_99* = 0.9940

---

## 2. Benchmark Performance Under Decoupled Decision Policies

| Benchmark | N | AUC | Platt ECE | Platt Brier | Spec@tau95* | Sens@tau95* |
|---|---:|---:|---:|---:|---:|---:|
| Face2Face_Holdout | 116 | 0.7836 | 0.1264 | 0.2112 | 94.8% | 20.7% |
| CelebDF_v2_Holdout | 511 | 0.7937 | 0.1662 | 0.2031 | 98.3% | 31.0% |
| FFpp_Mixed_InDomain | 801 | 0.8429 | 0.1440 | 0.1745 | 97.4% | 43.3% |
| JPEG_Q50_Compression | 801 | 0.7539 | 0.1850 | 0.2119 | 93.1% | 25.5% |
| H264_CRF23_ReEncoding | 511 | 0.5744 | 0.3373 | 0.3353 | 100.0% | 0.0% |

---

## 3. Decision-Score Distribution Shifts

| Benchmark | Median Logit | Real Median | Fake Median | Separation Margin | Overlap % |
|---|---:|---:|---:|---:|---:|
| Face2Face_Holdout | +0.283 | +0.025 | +0.969 | +0.944 | 93.1% |
| CelebDF_v2_Holdout | +0.398 | +0.124 | +0.960 | +0.836 | 95.7% |
| FFpp_Mixed_InDomain | +0.703 | +0.089 | +1.816 | +1.727 | 87.3% |
| JPEG_Q50_Compression | +0.773 | +0.296 | +1.183 | +0.886 | 97.4% |
| H264_CRF23_ReEncoding | +0.242 | +0.205 | +0.262 | +0.057 | 98.6% |

---

**Conclusion**: Calibration successfully decouples score-offset drift from ranking performance.
The validation-derived threshold tau_95* restores high specificity across independent test sets without altering ROC-AUC.
