# Deepfake Video Detection

Research-grade visual deepfake video detection system based on the visual attention and temporal modeling architecture (*Lal et al., 2025*).

## Project Pipeline Overview

The system processes video inputs through a face-centric pipeline tailored for commodity GPU environments (NVIDIA RTX 3050, 6 GB VRAM):

```
Dataset Manifest
   ↓
Indivisible Group-Stratified Split (Zero Leakage)
   ↓
Random Frame Sampling (Paper-aligned default, deterministic per-video SHA-256)
   ↓
MTCNN Face Detection (Per-pipeline instance, torch.inference_mode())
   ↓
Deterministic Primary Face Selection (Largest area / highest confidence)
   ↓
Bounding Box Expansion & Fallback Handling (Detected → Previous BBox → Center Crop → Reject)
   ↓
112×112 Face Crop & RGB Preprocessing
   ↓
ImageNet / None Normalization
   ↓
(T, 3, 112, 112) float32 Tensor Output
   ↓
Atomic Persistence (.pt tensor + metadata .json)
```

## Hardware Constraint

- **Target GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
- **Design Invariants**: 112×112 face crops, small batch sizes (8–16), mixed-precision FP16, on-the-fly or cached frame processing.

## Setup & Environment

### Prerequisites
- Python 3.11+
- CUDA 12.x / NVIDIA GPU Drivers

### Installation

1. Create and activate a virtual environment:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
pip install -e .
```

3. Verify environment diagnostics:
```powershell
python -m src.cli verify-env
```

## CLI Usage

### 1. Environment Diagnostics
```powershell
python -m src.cli verify-env
```

### 2. Group-Stratified Manifest Splitting
Splits a video dataset manifest by indivisible `group_id` (e.g. source identity in FaceForensics++) ensuring zero data leakage:
```powershell
python -m src.cli manifest split --input data/manifest.csv --output data/manifest_split.csv --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

### 3. Video Preprocessing
Preprocess a single video into a facial sequence tensor and JSON metadata:
```powershell
python -m src.cli preprocess --video path/to/video.mp4 --output-dir processed --split test
```

## Testing

Run the automated test suite:
```powershell
pytest tests -v
```
