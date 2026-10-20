# Deepfake Video Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Research Only](https://img.shields.io/badge/license-Research%20Only-yellow.svg)](PRD.md)
[![Hardware: RTX 3050 6GB](https://img.shields.io/badge/hardware-RTX%203050%206GB-green.svg)](context/architecture.md)

A research-grade, production-calibrated visual deepfake video detection system. Based on the spatial-attention and temporal-modeling architecture of *Lal et al. (Scientific Reports, 2025)* and evolved through multi-domain invariant representations, independent external cross-dataset validation, controlled perturbation stress testing, and preregistered cross-attention ablation experiments.

---

## Table of Contents

- [1. System Architecture & Data Flow](#1-system-architecture--data-flow)
- [2. Model Zoo & Architecture Versions](#2-model-zoo--architecture-versions)
- [3. Benchmark Results & Scientific Synthesis](#3-benchmark-results--scientific-synthesis)
- [4. Hardware Constraints & Efficiency](#4-hardware-constraints--efficiency)
- [5. Installation & Setup](#5-installation--setup)
- [6. Quick Start & CLI Usage](#6-quick-start--cli-usage)
- [7. REST API & Web Technical Dashboard](#7-rest-api--web-technical-dashboard)
- [8. Calibration, Thresholds & Scoring Semantics](#8-calibration-thresholds--scoring-semantics)
- [9. Testing & Reproducibility](#9-testing--reproducibility)
- [10. Limitations & Responsible Forensics Policy](#10-limitations--responsible-forensics-policy)

---

## 1. System Architecture & Data Flow

The system processes video inputs through a strictly modular, face-centric pipeline optimized for commodity GPU environments:

```
Input Video (.mp4, .avi, .mov, .mkv)
   │
   ▼
Deterministic Frame Sampling (T=16 uniform frames, per-video SHA-256 seed)
   │
   ▼
MTCNN Face Detection & Primary Selection (Largest area / highest confidence)
   │
   ▼
Bounding Box Smoothing & Robust Fallback (BBox → Previous Frame → Center Crop → Reject)
   │
   ▼
112×112 Face Crop & RGB Preprocessing (ImageNet mean/std normalization)
   │
   ▼
Spatial Feature Extraction (EfficientNet-B0 backbone / ResNeXt-50 baseline)
   │
   ▼
Spatial-Attention & Representation Hooks (CBAM / Soft-Attention / Stage-5 hooks)
   │
   ▼
Temporal Modeling (Global Transformer Encoder / BiLSTM sequence aggregation)
   │
   ▼
Multi-Domain Feature Invariance & Classification (Frequency + Texture + Primary Heads)
   │
   ▼
Calibrated Probabilities & Forensics Verdict (Temperature Scaling T=2.0502, tau*=0.1260)
```

---

## 2. Model Zoo & Architecture Versions

The project maintains an immutable, cryptographically verified hierarchy of models serving specialized research and operational roles:

| Model Identifier | Architecture Description | Role / Status | Checkpoint / Operating Policy |
|---|---|---|---|
| **`v5-family-invariant`** | EfficientNet-B0 + Frequency Residual + Local Texture Patch + TempTrans | **Active Production Champion** | $\tau^* = 0.5000$ (Macro AUC: 0.9165) |
| **`m22-ensemble`** | Log-Odds Platt Fusion ($0.50 z_{13} + 0.50 z_{18}$) | **Highest-Accuracy Research Champion** | $\tau_{95}^* = 0.9191$ (0.8122 F2F / 0.8214 Celeb-DF) |
| **`m18-dual-stream`** | M17 Track A Trunk + 3D Differential Kinematics + Gated Fusion | Reenactment Research Baseline | $\tau_{95}^* = 0.8821$ (Raw) / $0.8501$ (Platt) |
| **`m17-sensor-residuals`**| ResNeXt-50 + Multi-Scale SRM Residuals (Scales 1 & 2) | Sensor Noise Research Baseline | $\tau^* = 0.5000$ |
| **`v4-generalization`** | EfficientNet-B0 + Auxiliary Manipulation Family Classifier | Generalization Baseline (M13.2b) | $\tau^* = 0.5000$ |
| **`v3-robust` (M12)** | EfficientNet-B0 + Soft Attention + Transformer (Full FF++ C23) | Permanently Frozen Control | `c8265f634880c56929e5d82be8e4fc808fc7776d...` |
| **`v2-evolved` (M10)** | EfficientNet-B0 + Temporal Transformer | Evolutionary Checkpoint | `ff2fe00f03739c69bcd050de83e692ccfdc9f9...` |
| **`v1-baseline` (M7)** | ResNeXt-50 (32x4d) + Soft Attention + LSTM (Paper Reproduction) | Reference Baseline Engine | `9d22df92bdab20a35e1311d627d9fa487c757b...` |

---

## 3. Benchmark Results & Scientific Synthesis

### A. Full 4-Fold Leave-One-Family-Out (LOFO) Matrix Benchmark

Evaluates cross-manipulation transfer where one manipulation family is completely held out during training:

| Matrix Fold | Held-Out Family | Training Families | M12 Baseline AUC | M13.2b Champion AUC | v5 Production Champion AUC |
|---|---|---|:---:|:---:|:---:|
| **LOFO-F2F** | **Face2Face** | DF + FS + NT | 0.7787 | 0.7830 | **0.7903** (+0.0073) |
| **LOFO-NT** | **NeuralTextures** | DF + FS + F2F | 0.9298 | 0.9244 | **0.9499** (+0.0255) |
| **LOFO-FS** | **FaceSwap** | DF + NT + F2F | 0.9402 | **0.9474** | 0.9435 (-0.0039) |
| **LOFO-DF** | **Deepfakes** | FS + NT + F2F | 0.9863 | 0.9806 | **0.9824** (+0.0018) |
| **Macro Mean** | **All 4 Families** | Cross-Family Generalization | 0.9087 | 0.9089 | **0.9165** (+0.0076) |

### B. Independent External Generalization (Celeb-DF v2 Benchmark)

Evaluated strictly on the 518 official test queries (178 authentic celebrity videos, 340 deepfakes) without retraining:

- **ROC-AUC**: **0.7824** (Production Champion) → **0.8275** (M15 Arm B) → **0.8723** (M15 Arm C)
- **PR-AUC**: **0.8774**
- **Clean Specificity Floor**: **97.75%** (only 4 false alarms on 178 authentic videos, FAR: 2.25%)

### C. Milestone 15 Four-Arm Preregistered Architecture Experiment

Comprehensive comparative synthesis across all 4 experimental arms:

| Metric / Benchmark | Arm A (Frozen Champion) | Arm B (TS-CrossAttn) | Arm C (RobustGen) | Arm D (Ablation-NoGate) |
|---|:---:|:---:|:---:|:---:|
| **Face2Face Holdout AUC** | **0.8104** | 0.6980 | 0.7053 | 0.7284 |
| **Celeb-DF v2 Holdout AUC** | 0.7808 | 0.8275 | **0.8723** | 0.7902 |
| **Clean FF++ Specificity** | 93.88% | 93.99% | 81.55% | **94.85%** |
| **Clean Celeb-DF Specificity** | **97.75%** | 94.86% | 82.29% | 94.86% |
| **JPEG Q=50 ROC-AUC** | **0.6360** | 0.4994 | 0.5900 | 0.5625 |
| **P50 Inference Latency** | 24.95 ms | **20.48 ms** | 21.18 ms | 25.39 ms |
| **Peak GPU VRAM** | 4.85 GB | **0.51 GB** | 0.74 GB | 0.82 GB |
| **Final Promotion Decision** | **RETAIN CHAMPION** | RETAIN CHAMPION | RETAIN CHAMPION | RETAIN CHAMPION |

**Key Scientific Takeaway**: Dynamic cross-modal attention gating accounts for **+3.73%** external AUC transfer (Arm B vs Arm D), while multi-scale frequency consistency recovers compression robustness (+9.06% AUC) and boosts transfer to **0.8723 AUC**. However, because candidate arms failed clean specificity safety ceilings or Face2Face gains, `v5-family-invariant` is retained as the production engine under strict governance.

---

## 4. Hardware Constraints & Efficiency

All models and inference pipelines are validated to operate within commodity GPU constraints:

- **Target Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM).
- **Physical Peak VRAM**: **539.9 MB** during full video inference (< 10% VRAM utilization).
- **P50 Forward Latency**: **20.48 – 24.95 ms** per 16-frame sequence.
- **Throughput**: ~510 frames/sec (forward pass) / ~2.2 s total end-to-end video pipeline.
- **CPU Fallback**: Graceful, fully verified CPU execution mode if CUDA is unavailable.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.11+
- CUDA 12.x & compatible NVIDIA drivers

### Installation Steps

1. **Clone the repository and enter the directory**:
```powershell
git clone https://github.com/YourOrg/deepfake-video-detection.git
cd "deepfake-video-detection"
```

2. **Create and activate a virtual environment**:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
pip install -e .
```

4. **Verify environment and hardware diagnostics**:
```powershell
python -m src.cli verify-env
```

---

## 6. Quick Start & CLI Usage

The system provides a unified command-line interface:

### 1. Spatiotemporal Forensic Audit (Milestone 24)
Extracts temporal windows, computes multi-scale Gaussian/Laplacian edge anomalies, evaluates 3D kinematic differentials, and calculates calibrated 95% confidence intervals:
```powershell
python -m src.cli forensic-audit --video path/to/sample.mp4 --windows 4 --frames-per-window 16 --output reports/forensic_report.json
```

### 2. Run Single-Engine Inference
```powershell
python -m src.cli predict --video path/to/sample.mp4 --engine v5-family-invariant --threshold 0.50
```
*Available engines:* `v5-family-invariant`, `m22-ensemble`, `m18-dual-stream`, `m17-sensor-residuals`, `v4-generalization`, `v3-robust`, `v2-evolved`, `v1-baseline`.

### 3. Launch the Technical Web Dashboard
```powershell
python -m src.cli serve --host 127.0.0.1 --port 8000
```
Navigate to `http://localhost:8000` to interact with the real-time forensics dashboard.

### 4. Export Model to TorchScript or ONNX
```powershell
python -m src.cli export --format onnx --output exports/detector.onnx
python -m src.cli export --format torchscript --output exports/detector.pt
```

### 5. Benchmark Inference Latency & VRAM
```powershell
python -m src.cli benchmark-latency --batch-size 1 --frames 16 --runs 100
```

### 6. Preprocess Video to Cached Facial Tensors
```powershell
python -m src.cli preprocess --video path/to/video.mp4 --output-dir processed --split test
```

### 7. Indivisible Group-Stratified Manifest Splitting
```powershell
python -m src.cli manifest split --input data/manifest.csv --output data/manifest_split.csv --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

---

## 7. REST API & Web Technical Dashboard

### FastAPI Endpoints

1. **`POST /api/v1/forensic-audit` (Spatiotemporal Forensic Pipeline)**:
   Performs multi-window evaluation, spatial Laplacian boundary analysis, 3D kinematic energy derivation, and base64 JET saliency heatmap generation. Returns:
   ```json
   {
     "video_filename": "sample.mp4",
     "verdict": "AUTHENTIC",
     "verdict_confidence": "HIGH",
     "video_fake_probability": 0.4750,
     "confidence_interval": [0.4710, 0.4800],
     "temporal_consistency_score": 0.9980,
     "kinematic_residual_energy": 9.08,
     "saliency_heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAAAH...",
     "windows_analyzed": [
       {
         "window_id": 0,
         "start_sec": 0.0,
         "end_sec": 2.0,
         "frame_count": 16,
         "calibrated_fake_probability": 0.4720,
         "kinematic_residual_energy": 8.95
       }
     ]
   }
   ```

2. **`POST /api/v1/compare` (Cross-Engine Attribution Arbitration)**:
   Arbitrates Champion M13.5 (static spatial/texture), Baseline M18 (dynamic 3D kinematics), and Fused M22 (calibrated log-odds ensemble). Returns consensus classification (`CONCORDANT_FAKE`, `CONCORDANT_REAL`, or `DISCORDANT`) and modality attribution analysis.

3. **`POST /api/v1/predict` (Single-Engine Calibrated Inference)**:
   Single-model forward pass with MC-Dropout epistemic uncertainty, out-of-distribution (OOD) distance diagnostics, and execution latency waterfall.

4. **`GET /api/v1/model-info`**: Architecture specifications, provenance hashes, and operating thresholds for all 8 engines.

5. **`GET /health`**: System status, CUDA GPU device, and model preload verification.

### Interactive Technical Dashboard (`src/ui/`)
- **Visual Design**: Dark glassmorphic aesthetic (`#030408` base, `#0E1320` surface) adhering to `context/ui-context.md`.
- **Modes of Analysis**:
  1. **Single-Engine Diagnosis**: Real-time video player, interactive threshold slider, latency waterfall, and OOD badges.
  2. **Cross-Engine Comparison**: 3-column side-by-side card grid arbitrating static spatial vs kinematic cues.
  3. **Spatiotemporal Forensic Audit**: Calibrated 95% CI verdict card, 2D spatial Laplacian anomaly heatmap with continuous JET legend, kinematic motion energy telemetry, and dynamic temporal window decomposition timeline.
- **Background**: Dynamic high-performance ThreeUI predictive arc particle canvas.

---

## 8. Calibration, Thresholds & Scoring Semantics

To prevent misinterpretation of raw neural network outputs:

1. **Temperature Scaling ($T = 2.0502$)**:
   $$\hat{p}_{\text{cal}} = \sigma\left(\frac{z}{2.0502}\right)$$
   Fitted strictly on 1,255 validation sequences by minimizing negative log-likelihood (ECE reduced from $0.2729 \to 0.0866$).
2. **Operating Threshold Policy**:
   - **Standard Operating Point**: $\tau = 0.50$ (balanced precision and recall).
   - **High-Sensitivity Policy**: $\tau^* = 0.1260$ (guarantees $\ge 90.0\%$ specificity on clean authentic media while maximizing recall on challenging manipulation types).
3. **Out-of-Distribution (OOD) Metric**:
   Penultimate feature Euclidean distance evaluated against a validation manifold threshold ($3.18$ at 99th percentile).

---

## 9. Testing & Reproducibility

Full workspace test suite enforcing zero data leakage, shape contracts, and hash immutability:

```powershell
pytest tests/ -v
```

- **Regression Status**: **406/406 tests passing (100% pass rate)**.
- **Contract Tests**: Verifies MTCNN face detection, bounding box fallbacks, tensor dimensions, TorchScript/ONNX numerical parity ($< 1.2 \times 10^{-7}$), bitwise checkpoint hashes, spatiotemporal window extraction, and multi-engine serving.
- **Determinism**: Scoped deterministic inference verified under fixed random seeds across CPU and CUDA.

---

## 10. Limitations & Responsible Forensics Policy

Per project governance guidelines and Rule 18:

> [!WARNING]
> ### Forensic Transparency & Limitations Disclosure
> 1. **Not Absolute Proof**: A high deepfake score indicates statistical anomaly relative to training distributions; it is not legal proof of synthetic generation.
> 2. **Compression Vulnerability**: Heavy compression (JPEG $Q < 25$, CRF $> 35$) and severe resizing induce prediction drift and potential false alarms.
> 3. **Within-Manifold Forgeries**: Manipulations lacking facial boundary seams (e.g., Face2Face expression modifications) exhibit lower detection confidence than whole-face replacements.
> 4. **Single-Face Focus**: The pipeline currently evaluates the primary (largest area) face detected in the video sequence.
> 5. **OOD Flagging**: Inputs flagged as Out-of-Distribution must be verified by a human forensic analyst.

---

## Citation & References

```bibtex
@article{lal2025deepfake,
  title={Deepfake video deception detection using visual attention-based method},
  author={Lal, et al.},
  journal={Scientific Reports},
  volume={15},
  pages={23920},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-025-23920-0}
}
```
