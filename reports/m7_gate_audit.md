# Milestone 7 — Video Inference Service, Model Export, Dashboard & Packaging Audit Report

**Date:** 2026-09-21  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)  
**Model Under Test:** ResNeXt-50 + Soft Attention + LSTM (`checkpoints/ffpp_c23_baseline_soft_lstm/best.pth`)  
**M7 Research Freeze:** STRICTLY OBSERVED (Zero retraining, fine-tuning, modification, or overwriting of checkpoint; SHA-256 verified)

---

## 1. Executive Gate Summary

| Gate | Name | Status | Evidence / Verification Key |
|---|---|---|---|
| **Gate 1** | Inference Correctness | `PASS` | `reports/gate1_inference_audit.json` — Physical real & fake video inference, deterministic repeated runs (diff = `0.00000000`). |
| **Gate 2** | Model Export | `PASS` | `reports/gate2_export_audit.json` — TorchScript diff = `0.0000000000`, ONNX diff = `0.0000001192` (fixed $T=16$, dynamic batch $B \ge 1$). |
| **Gate 3** | CLI Interface | `PASS` | `reports/gate3_cli_audit.json` — Subcommands `predict`, `export`, `benchmark-latency`, `serve`. Physical video CLI prediction executed. |
| **Gate 4** | API Service | `PASS` | `reports/gate4_api_audit.json` — FastAPI `/health`, `/api/v1/model-info`, `/api/v1/predict`. Real video uploaded with HTTP 200. |
| **Gate 5** | Technical Dashboard | `PASS` | `reports/gate5_dashboard_audit.json` — Dark technical UI adhering to `context/ui-context.md`. Waterfall latency, provenance, and research banner. |
| **Gate 6** | Performance Profiling | `PASS` | `reports/latency_profile_rtx3050.json` — Model forward: **31.27 ms** (511.7 fps). End-to-end: **2232.51 ms**. Interactive inference verified. |
| **Gate 7** | Containerization | `PENDING — DOCKER DAEMON OFFLINE` | `reports/gate7_container_audit.json` — `Dockerfile`, `docker-compose.yml` authored. Live daemon check confirmed offline; pending daemon launch. |
| **Gate 8** | Regression & Freeze | `PASS` | `reports/gate8_regression_audit.json` — 165/165 tests passed (100%). Checkpoint SHA-256 bitwise identical to pre-M7 state. |

---

## 2. Gate-by-Gate Verification Details

### Gate 1 — Inference Correctness (`PASS`)
- **Input Contract:** Exact $(16, 3, 112, 112)$ floating-point tensor sampled uniformly from video, faces localized via MTCNN, resized with bilinear interpolation, and normalized with ImageNet statistics ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
- **Physical Datasets Evaluated:**
  - Real Video: `Datasets/FaceForensics++_C23/original/004.mp4` $\implies$ Logit: `0.9930`, Prob: `0.7297`, Faces: `16/16`.
  - Manipulated Video: `Datasets/FaceForensics++_C23/Deepfakes/004_982.mp4` $\implies$ Logit: `1.0517`, Prob: `0.7411`, Faces: `16/16`.
- **Determinism Check:** Repeated forward passes on identical input produced max absolute difference of `0.00000000`.

### Gate 2 — Model Export (`PASS`)
- **TorchScript:** Exported via `torch.jit.trace` on dummy input $(1, 16, 3, 112, 112)$. Max absolute difference against eager PyTorch: `0.0000000000` (within atol $10^{-5}$).
- **ONNX Export & Invariants:** Exported via `torch.onnx.export` (opset 17). Dynamic batching ($B \ge 1$) was enabled while strictly constraining sequence length to $T=16$ as demanded by the recurrent LSTM hidden state contract.
  - Parity at $B=1$: Max absolute difference = `0.0000001192` (within atol $10^{-4}$).
  - Parity at $B=2$: Max absolute difference = `0.0000000000`.
- **Core Decoupling:** Standalone inference pipeline remains fully functional without ONNX dependencies.

### Gate 3 — CLI Interface (`PASS`)
- Supported and verified subcommands:
  - `predict`: Standalone video inference with configurable threshold, JSON export, and device selection.
  - `export`: Export checkpoint to TorchScript, ONNX, or both.
  - `benchmark-latency`: Dual-tier latency profiler.
  - `serve`: Launch FastAPI server and dashboard via uvicorn.
- Physical CLI execution on real video generated `reports/gate3_cli_predict_output.json`.

### Gate 4 — API Service (`PASS`)
- Implemented with FastAPI and uvicorn:
  - `GET /health`: Returns service health, model loaded status, active device, and version.
  - `GET /api/v1/model-info`: Returns architecture specification, training dataset, citation, and research limitations.
  - `POST /api/v1/predict`: Accepts multipart video upload, performs inference, returns verdict, probability, confidence, face stats, and latencies.
- Physically verified with real MP4 video upload through `TestClient`.

### Gate 5 — Technical Dashboard (`PASS`)
- Built in `src/ui/` with Vanilla CSS and JS following `context/ui-context.md` semantic color tokens (`--bg-base: #0B0F14`, `--bg-surface: #111821`, `--accent-primary: #6EA8FE`, etc.).
- **Features Verified:**
  - Dynamic verdict card ("REAL / AUTHENTIC" vs "DEEPFAKE DETECTED" using text and icons, compliant with Rule 18 and accessibility guidelines).
  - Model score progress bar and raw logit.
  - Interactive decision threshold slider (`0.05` to `0.95`).
  - Face extraction diagnostics (analyzed frames, detected faces, fallback ratio).
  - Latency waterfall breakdown (MTCNN preprocessing vs Model forward pass).
  - Full model provenance metadata table.
  - Prominent Research Limitations Callout Banner highlighting face dependence, 16-frame sampling, and cross-dataset domain shift.

### Gate 6 — Performance Profiling (`PASS`)
- Empirical benchmark executed on **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)**:
  - **Isolated Model Forward Pass ($B=1, T=16, 112 \times 112$):**
    - Mean Latency: **31.27 ms**
    - Median Latency: **30.75 ms**
    - 95th Percentile: **32.99 ms**
    - 99th Percentile: **33.54 ms**
    - Throughput: **511.7 frames/sec** (~32 video clips/sec)
  - **End-to-End Physical Video Pipeline:**
    - Frame extraction & MTCNN face detection: **2204.71 ms** (~98.8% of runtime)
    - Model forward inference: **27.37 ms** (~1.2% of runtime)
    - Total Video Processing Latency: **2232.51 ms** (~2.23 s)
  - **GPU Memory Footprint:**
    - Allocated: **231.49 MB**
    - Reserved: **642.00 MB**
    - Peak Memory: **373.02 MB** (< 10% of 6 GB VRAM budget)
  - **Inference Classification:** Strictly classified as **Interactive Video Clip Inference**. The claim of "real-time video streaming" is intentionally avoided and deferred, as end-to-end processing is dominated by per-frame MTCNN localization.

### Gate 7 — Containerization (`PENDING — DOCKER DAEMON OFFLINE`)
- Authoring completed:
  - `Dockerfile`: Multi-stage GPU runtime based on `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`, non-root user `appuser`, ffmpeg/opencv dependencies, and healthcheck.
  - `docker-compose.yml`: Configured with NVIDIA GPU device reservation (`capabilities: [gpu]`) and read-only checkpoint mounts.
  - `.dockerignore`: Strictly excludes datasets, virtualenvs, and sensitive run caches.
- **Daemon Inspection:** Executed live `docker info`. Docker CLI 28.4.0 is installed, but the Docker Desktop daemon service is currently offline. Per project policy, Gate 7 is recorded strictly as `PENDING — DOCKER DAEMON OFFLINE` until the daemon is started and physical GPU-in-container verification is performed.

### Gate 8 — Regression Testing & Research Freeze (`PASS`)
- **Automated Regression Suite:** 165 tests collected; **165 passed (100% pass rate)** in 204.96s.
  - M1–M6 Pre-existing Tests: 142/142 passed (100%).
  - M7 Newly Added Tests: 23/23 passed (100%).
- **Research Freeze Audit:**
  - Target: `checkpoints/ffpp_c23_baseline_soft_lstm/best.pth`
  - Pre-M7 SHA-256: `9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a`
  - Post-M7 SHA-256: `9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a`
  - Status: Bitwise identical; zero modification or retraining occurred.
