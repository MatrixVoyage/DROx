# Deepfake Video Detection: Multi-Modal Spatiotemporal Forensics & Calibrated Neural Inspection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![FastAPI 0.115+](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Pytest 100% Pass](https://img.shields.io/badge/tests-436%20passed%20%28100%25%29-brightgreen.svg)](tests/)
[![Hardware: RTX 3050 6GB](https://img.shields.io/badge/hardware-RTX%203050%206GB-green.svg)](README.md#hardware-constraints--efficiency)
[![License: Research Only](https://img.shields.io/badge/license-Research%20Only-yellow.svg)](README.md#license--citation)

A research-oriented, production-calibrated visual deepfake video detection platform. Grounded in the spatial-attention and temporal-modeling architecture of *Lal et al. (Scientific Reports, 2025)* and evolved into a multi-modal inspection framework combining deep neural models (EfficientNet-Transformer, 3D Kinematics, Log-Odds Platt Ensembles), deterministic forensic signals (Laplacian boundary seams, Farneback optical flow velocity divergence, Eye Aspect Ratio blink kinematics, 2D-FFT azimuthal spectral decay), statistical calibration ($T=2.0502$), and an evidence-grounded explainable AI forensic analyst.

---

## Live System Architecture & Interactive Dataflow

The end-to-end detection pipeline processes unconstrained video containers via a decoupled dual-track architecture engineered to operate within commodity GPU constraints (NVIDIA RTX 3050 6GB VRAM, peak memory < 540 MB during inference):

<p align="center">
  <img src="reports/figures/interactive_dataflow.svg" alt="Deepfake Detection Live Dataflow Diagram" width="100%"/>
</p>

### Comprehensive System Architecture

<p align="center">
  <img src="reports/figures/system_architecture_comprehensive.png" alt="Comprehensive Dual-Track System Architecture" width="100%"/>
</p>
<p align="center"><em>Figure 1: Complete system architecture faithfully representing the repository implementation: Decoupled Ingestion & Face Tracking, Dual-Track Neural (Track A) and Deterministic Forensics (Track B), Statistical Calibration & OOD Gating, and the Multi-Engine Serving Layer with Groq AI Analyst.</em></p>

---

## Native Interactive Tensor Flowchart

```mermaid
flowchart TD
    subgraph S1["1. VIDEO INGESTION & TRACKING"]
        V["Raw Video File<br/>(MP4 / AVI / MOV)"] --> DEC["OpenCV Demuxer<br/>(PTS Timestamp Check)"]
        DEC --> SMP["Deterministic Sampler<br/>(T=16 frames, SHA-256)"]
        SMP --> MTCNN["MTCNN Face Cascade<br/>(3-Stage Detection & 1.30 Margin)"]
        MTCNN --> CROPS["Normalized Face Crops<br/>(B, 16, 3, 112, 112)"]
    end

    subgraph S2["2. DUAL-TRACK ANALYSIS"]
        CROPS --> TRK_A["TRACK A: NEURAL ENGINES"]
        CROPS --> TRK_B["TRACK B: FORENSIC SIGNALS"]

        subgraph S2A["Track A: Deep Learning Models"]
            TRK_A --> V5["v5-family-invariant<br/>(EfficientNet-B0 + SoftAttn + Transformer)"]
            TRK_A --> M18["m18-dual-stream<br/>(Appearance + 3D Differential Kinematics)"]
            V5 -->|z13 logit| M22["m22-ensemble<br/>(Platt Log-Odds Fusion)"]
            M18 -->|z18 logit| M22
        end

        subgraph S2B["Track B: Physical & Biological Telemetry"]
            TRK_B --> LAP["Laplacian Seam (L0)<br/>Boundary vs Landmark Ratio"]
            TRK_B --> OF["Farneback Optical Flow<br/>Velocity Divergence & Acceleration"]
            TRK_B --> EAR["EAR Blink Kinematics<br/>Ocular Motion Anomalies"]
            TRK_B --> FFT["2D-FFT Spectrum<br/>Azimuthal High-Freq Power Decay"]
        end
    end

    subgraph S3["3. CALIBRATION & DECISION ENGINE"]
        V5 -->|Logit z| CALIB["Temperature Scaling<br/>P = sigmoid(z / 2.0502)"]
        M22 -->|Fused Logit| CALIB
        CROPS -->|Latents| OOD["Mahalanobis OOD Gating<br/>d_OOD > 3.5 -> Flag OOD"]
        CALIB --> DEC_ENG["Operational Policy Engine<br/>tau=0.50 Standard | tau=0.1912 Youden-Optimal"]
        OOD --> DEC_ENG
        LAP & OF & EAR & FFT --> DEC_ENG
    end

    subgraph S4["4. SERVING & EXPLAINABILITY LAYER"]
        DEC_ENG --> API["FastAPI REST Endpoints<br/>(/api/v1/predict, /compare, /forensic-audit)"]
        DEC_ENG --> UI["Technical Dashboard<br/>(Glassmorphic Web UI + Heatmaps)"]
        DEC_ENG --> GROQ["Groq AI Analyst Briefing<br/>(Evidence-Gated Causal Guardrails)"]
    end

    classDef stage1 fill:#0F172A,stroke:#0284C7,stroke-width:1.5px,color:#E2E8F0;
    classDef stage2 fill:#0F172A,stroke:#7C3AED,stroke-width:1.5px,color:#E2E8F0;
    classDef stage3 fill:#0F172A,stroke:#D97706,stroke-width:1.5px,color:#E2E8F0;
    classDef stage4 fill:#0F172A,stroke:#E11D48,stroke-width:1.5px,color:#E2E8F0;

    class S1 stage1;
    class S2 stage2;
    class S3 stage3;
    class S4 stage4;
```

---

## Empirical Model Comparison & Pareto Frontier

This repository implements, calibrates, and benchmarks **8 distinct model engines**. Below is the empirical performance comparison measured directly across in-domain datasets (**FaceForensics++ C23**), unseen external holdout benchmarks (**Celeb-DF v2**), challenging reenactment manipulations (**Face2Face**), and heavy JPEG compression perturbations ($Q=50$):

<p align="center">
  <img src="reports/figures/model_comparison_benchmark.png" alt="Empirical Model Comparison and Hardware Pareto Frontier" width="100%"/>
</p>
<p align="center"><em>Figure 2: (Left) Empirical detection AUROC across model generations and test conditions. (Right) Hardware Pareto Frontier (Inference Latency vs. Macro AUROC) on NVIDIA RTX 3050 Laptop GPU, illustrating the 45 ms real-time serving boundary.</em></p>

### Comprehensive Model Benchmark Matrix

| Model Identifier | Architecture Description | Feature Dim | FF++ C23 (In-Domain) | Celeb-DF v2 (Unseen OOD) | Face2Face (Reenactment) | JPEG Q50 Robustness | P50 Latency (RTX 3050) | Peak VRAM | Role & Deployment Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`v5-family-invariant`** | EfficientNet-B0 + Soft Spatial Attention + 2-Layer Transformer | 1280 | **0.9165** | 0.7824 | **0.7947** | 0.6360 | **24.95 ms** | **539.9 MB** | **Active Production Champion** (Best balance of latency & accuracy) |
| **`m22-ensemble`** | Log-Odds Platt Ensemble ($0.50 z_{13} + 0.50 z_{18}$) | 1280/2048 | **0.9230** | **0.8213** | **0.8118** | **0.8944** | 72.58 ms | 680.0 MB | **Highest-Accuracy Research Champion** (Best cross-dataset transfer) |
| **`m18-dual-stream`** | Static Trunk + 3D Differential Kinematics | 1280 | 0.8950 | **0.8166** | 0.7858 | **0.8822** | 48.20 ms | 620.0 MB | Reenactment Research Baseline |
| **`v3-robust` (M12)** | EfficientNet-B0 + Soft Attention + Transformer (Full FF++) | 1280 | 0.9087 | 0.7780 | 0.7787 | 0.6120 | 24.50 ms | 490.0 MB | Permanently Frozen Control Checkpoint |
| **`v2-evolved` (M10)** | EfficientNet-B0 + Temporal Transformer | 1280 | 0.9020 | 0.7450 | 0.7150 | 0.5420 | **22.10 ms** | **420.0 MB** | Lightweight Edge Baseline |
| **`v1-baseline` (M7)** | ResNeXt-50 (32×4d) + Soft Attention + 1-Layer LSTM | 2048 | 0.8840 | 0.7120 | 0.6820 | 0.5120 | 27.40 ms | 485.0 MB | Reference Paper Reproduction (*Lal et al. 2025*) |

---

## 4-Fold Leave-One-Family-Out (LOFO) Matrix

To evaluate true generalization to unseen manipulation types, the model was evaluated under a strict 4-fold cross-validation scheme where one entire manipulation family was withheld during training:

| Matrix Fold | Held-Out Manipulation | In-Training Sets | Baseline AUC | Champion AUC | Clean Specificity |
|---|---|---|:---:|:---:|:---:|
| **LOFO-F2F** | **Face2Face** (Facial Reenactment) | DF + FS + NT | 0.7787 | **0.7903** (+0.0116) | 93.88% |
| **LOFO-NT** | **NeuralTextures** (Facial GAN) | DF + FS + F2F | 0.9298 | **0.9499** (+0.0201) | 94.20% |
| **LOFO-FS** | **FaceSwap** (Identity Replacement) | DF + NT + F2F | 0.9402 | **0.9435** (+0.0033) | 95.10% |
| **LOFO-DF** | **Deepfakes** (Autoencoder) | FS + NT + F2F | 0.9806 | **0.9824** (+0.0018) | 96.50% |
| **Macro Mean** | **All 4 Families** | Cross-Family Generalization | 0.9087 | **0.9165** (+0.0078) | **94.92%** |

---

## Spatial Attribution & Blending Seam Localization

The system produces spatial attention heatmaps revealing exactly where the neural backbone focuses its decision:

<p align="center">
  <img src="reports/figures/attribution/true_positive_fake_attribution.png" alt="True Positive Deepfake Attribution" width="420"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="reports/figures/attribution/true_negative_real_attribution.png" alt="True Negative Authentic Attribution" width="420"/>
</p>
<p align="center"><em>Figure 3: (Left) True-positive deepfake showing elevated attention concentration along the synthetic blending seam around the chin and jawline. (Right) True-negative authentic face exhibiting uniform, diffuse anatomical attention across ocular and nasal structures.</em></p>

---

## Key Technical Innovations

### 1. Multi-Modal Forensic Verification (Track B)
Rather than trusting raw neural network scores alone, the system extracts deterministic physical and biological measurements:
- **Spatial Laplacian Seam Analysis**: Pyramidal band decomposition ($L_0 = \text{Frame} - \text{Gaussian}(\text{Frame})$) quantifying the energy ratio between outer facial crop boundaries and inner facial landmarks.
- **Optical Flow Velocity Divergence**: Dense Farneback flow tracking vector field motion differentials, detecting unnatural synthetic smoothing along facial perimeters.
- **Eye Aspect Ratio (EAR) Blink Kinematics**: 68-point facial landmark geometry tracking involuntary ocular blink frequencies and duration anomalies.
- **2D-FFT Azimuthal Power Spectrum**: Fast Fourier Transform analyzing radial spectral decay to detect high-frequency GAN upsampling grid artifacts.

### 2. Statistical Calibration & Decision Science
Neural networks trained with standard cross-entropy produce overconfident, uncalibrated scores. The system strictly separates **logits**, **probabilities**, and **operational thresholds**:
$$\hat{p}_{\text{cal}} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + \exp(-z / 2.0502)}$$
- **Temperature Scaling ($T = 2.0502$)**: Optimized via Negative Log-Likelihood (NLL) on 1,255 validation sequences. Reduces Expected Calibration Error (ECE) from $0.2729 \to 0.0866$ while preserving 100% of ranking AUROC.
- **Operating Threshold Policies**:
  - `Standard` ($\tau = 0.5000$): Balanced decision threshold for general screening.
  - `Youden-Optimal` ($\tau = 0.1912$): Mathematically minimizes error trade-off ($J = \text{TPR} - \text{FPR}$).
  - `High-Sensitivity` ($\tau^* = 0.1260$): Maximizes detection recall on compressed videos while enforcing $\ge 90\%$ specificity on clean authentic media.
  - `High-Specificity` ($\tau = 0.6820$): Enforces $95\%$ specificity for high-consequence forensic investigations.

### 3. Epistemic Uncertainty & Out-of-Distribution (OOD) Guardrails
- **Monte Carlo Dropout**: Computes score mean and dispersion variance across 10 stochastic forward passes to quantify model uncertainty.
- **Penultimate Manifold OOD Detection**: Mahalanobis centroid distance in the 1280-dimensional latent space. Samples with $d_{\text{OOD}} > 3.5$ are flagged as Out-of-Distribution.

### 4. Explainable AI Forensic Analyst (Groq Integration)
An optional generative synthesis engine transforms structured telemetry into human-friendly, evidence-grounded forensic briefings:
- **Strict Causal Guardrails**: Algorithmic measurements are never described as proven physical splicing or tampering.
- **Post-Generation Consistency Validation**: Validates that every metric, timestamp, and verdict in the narrative exactly matches underlying telemetry before delivery.
- **Deterministic Safe Fallback**: If the LLM times out or fails validation, a deterministic rule-based narrative is substituted automatically.

---

## Hardware Constraints & Efficiency

All models and inference pipelines are validated to operate within commodity GPU environments:

- **Target Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM).
- **Peak VRAM Consumption**: **539.9 MB** during full video inference (< 10% VRAM utilization).
- **Forward Pass Latency**: **24.95 ms** per 16-frame sequence.
- **Deep Model Throughput**: **511 frames/sec** on CUDA Tensor Cores with FP16 mixed precision.
- **End-to-End Clip Runtime**: **2.23 seconds** (including OpenCV demuxing, MTCNN face detection, normalization, inference, and signal calculation).
- **CPU Fallback**: Automatic, fully tested CPU execution mode if CUDA is unavailable.

---

## Quickstart & Local Execution

### 1. Prerequisites
- Python 3.10 or 3.11
- Git
- NVIDIA GPU with CUDA 12.x drivers (recommended; CPU mode is supported)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/deepfake-video-detection.git
cd deepfake-video-detection
```

### 3. Create and Activate Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 5. Configure Optional Environment Variables
```bash
cp .env.example .env
```
*(Optional: Open `.env` and configure your `GROQ_API_KEY` to enable natural-language AI analyst briefings).*

### 6. Verify Environment and Hardware
```bash
python -m src.cli verify-env
```
Expected output:
```text
=== Deepfake Detection Environment Diagnostics ===
Python Version: 3.11.x
PyTorch Version: 2.6.0+cu124
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Total VRAM: 6.00 GB
MTCNN: Available
OpenCV Version: 5.0.0
==================================================
```

---

## Running the Application

### Option A: Launch Interactive Web UI & REST API

Start the FastAPI application and glassmorphic technical dashboard:

```bash
python -m src.cli serve --host 127.0.0.1 --port 8000
```
Open **`http://localhost:8000`** in your browser.

#### Technical Dashboard Features:
- **Interactive Video Upload**: Drag-and-drop video player with instantaneous preview.
- **Real-Time Threshold Slider**: Dynamically adjusts decision thresholds between Youden-optimal ($\tau=0.1912$), Standard ($\tau=0.50$), and High-Specificity ($\tau=0.6820$).
- **2D Spatial Laplacian Anomaly Map**: Visualizes high-frequency boundary seams with continuous JET colormap.
- **Multi-Window Timeline**: Inspects temporal consistency across sliding 2-second observation windows.
- **Structured Telemetry Tab**: Displays exact mathematical features, OOD distance, dispersion intervals, and container metadata.
- **Explainable AI Analyst Tab**: Renders structured forensic briefings synthesized via Groq LLM with strict grounding verification.

---

### Option B: Command-Line Interface (CLI)

#### 1. Full Spatiotemporal Forensic Audit
```bash
python -m src.cli forensic-audit --video path/to/sample.mp4 --windows 4 --frames-per-window 16 --output reports/forensic_report.json
```

#### 2. Standalone Neural Model Inference
```bash
python -m src.cli predict --video path/to/sample.mp4 --engine v5-family-invariant --threshold 0.50
```

#### 3. Benchmark Inference Latency & Hardware Throughput
```bash
python -m src.cli benchmark-latency --batch-size 1 --frames 16 --runs 100
```

#### 4. Export Model to ONNX or TorchScript
```bash
python -m src.cli export --format onnx --output exports/detector.onnx
python -m src.cli export --format torchscript --output exports/detector.pt
```

---

### Option C: Docker Deployment

Run the complete platform inside an isolated container:

```bash
docker-compose up --build
```
Access the dashboard at `http://localhost:8000`.

---

## REST API Documentation

When the server is running, interactive OpenAPI/Swagger documentation is available at **`http://localhost:8000/docs`**.

### Primary Endpoints:

#### 1. `POST /api/v1/forensic-audit`
Performs end-to-end spatiotemporal forensic audit, returning multi-window scores, physical measurements, and base64 heatmap:

```json
{
  "video_filename": "sample.mp4",
  "verdict": "AUTHENTIC",
  "verdict_confidence": "HIGH",
  "video_fake_probability": 0.0472,
  "confidence_interval": [0.0381, 0.0563],
  "temporal_consistency_score": 0.9980,
  "laplacian_boundary_ratio": 1.042,
  "optical_flow_divergence": 0.0184,
  "saliency_heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "groq_analyst_report": "The submitted video was evaluated across 4 observation windows..."
}
```

#### 2. `POST /api/v1/predict`
Single-engine forward pass with Monte Carlo dropout epistemic uncertainty, out-of-distribution distance diagnostics, and execution latency breakdown.

#### 3. `POST /api/v1/compare`
Cross-engine arbitration comparing Champion M13.5 (spatial artifacts), Baseline M18 (kinematic motion), and Fused M22 (calibrated log-odds ensemble).

#### 4. `GET /health`
Returns system status, active CUDA device, and preloaded model verification.

---

## Testing & Verification

The repository enforces strict architectural contracts, shape expectations, bitwise checkpoint provenance, and regression tests:

```bash
pytest tests/ -v
```

- **Test Suite Status**: **436 passed, 0 failed (100% pass rate)**.
- **Coverage**:
  - `test_models.py`: Tensor shape invariants, ResNeXt cardinality, attention equations, LSTM recurrence.
  - `test_pipeline.py`: Frame sampling strides, MTCNN face crops, ImageNet normalization.
  - `test_calibration.py`: Temperature scaling NLL optimization, Platt logistic fitting, threshold derivation.
  - `test_video_specific_telemetry.py`: Physical signal determinism, baseline checks, attribution gating.
  - `test_groq_analyst.py`: Evidence grounding, causal guardrail validation, fallback cascading.
  - `test_export.py`: ONNX and TorchScript numerical parity ($< 1.2 \times 10^{-7}$).

---

## Repository Structure

```text
├── configs/                   # Versioned YAML configurations
│   ├── default.yaml           # Core training, sampling, and optimizer parameters
│   ├── experiments/           # Architecture ablations and cross-dataset evaluation configs
│   └── forensics/             # Empirical baseline distributions (FF++ C23)
├── src/                       # Production source code
│   ├── api/                   # FastAPI application server and route definitions
│   ├── data/                  # Video demuxer, MTCNN face detector, cropper, adapters
│   ├── models/                # Backbones (ResNeXt, EfficientNet), Attention, Temporal models
│   ├── training/              # Trainer, mixed precision, checkpoint manager
│   ├── evaluation/            # Calibration fitting, ROC-AUC, EER, cross-dataset benchmarks
│   ├── forensics/             # Spatial-temporal audit engine, physical signals, Groq analyst
│   ├── inference/             # VideoPredictor, uncertainty estimation, OOD detection
│   ├── ui/                    # Glassmorphic technical dashboard (HTML/CSS/JS)
│   └── utils/                 # Configuration loaders, seed management, logging
├── tests/                     # 436 automated pytest unit and integration tests
├── scripts/                   # Automation scripts for benchmarks and data manifests
├── reports/                   # Empirical benchmark logs, ROC curves, and figures
│   └── figures/               # System flowcharts, publication plots, and heatmaps
├── .env.example               # Environment template for optional Groq API key
├── pyproject.toml             # Package build specification and metadata
├── requirements.txt           # Pinned production and research dependencies
├── Dockerfile                 # Multi-stage production container definition
└── docker-compose.yml         # Container composition configuration
```

---

## Responsible Forensics & Limitations Disclosure

In accordance with scientific integrity guidelines:

> [!WARNING]
> ### Forensic Disclaimers & Operational Boundaries
> 1. **Statistical Score vs. Absolute Proof**: A high deepfake score indicates that facial features and temporal dynamics align statistically with known synthetic manipulation distributions. It does not constitute standalone legal proof of forgery.
> 2. **Compression Degradation**: Extreme video compression (H.264 CRF $> 35$, JPEG $Q < 25$) destroys subtle frequency and PRNU residuals, inducing score variance.
> 3. **Within-Manifold Forgeries**: Expression-only manipulations (e.g., Face2Face) operate largely within natural facial subspaces without boundary seams, exhibiting lower detection sensitivity than whole-face swaps.
> 4. **Facial Tracking Continuity**: The system enforces quality gates. If fewer than 8 face crops are extracted, or if tracking is lost, the system emits an explicit `LOW_QUALITY` rejection or `INCONCLUSIVE` verdict rather than asserting authenticity.
> 5. **Human-in-the-Loop Verification**: Predictions flagged as Out-of-Distribution (OOD) or exhibiting score dispersion $\ge 0.15$ must be reviewed by a qualified forensic analyst.

---

## License & Citation

This codebase is released for academic and research purposes.

If you utilize this architecture, benchmarks, or forensic components in your research, please cite:

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
