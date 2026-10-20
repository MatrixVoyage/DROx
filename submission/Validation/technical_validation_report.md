# DeepGuard Technical Integrity & Validation Report

**Document ID**: `DG-VAL-2026-V2`  
**Evaluation Date**: October 15, 2026  
**Lead Evaluator**: Senior ML Engineer / Academic Project Submission Lead  
**Hardware Platform**: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM), Intel Core CPU, 16 GB RAM  
**Software Environment**: Windows 11 x64, Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4  

---

## 1. Executive Summary

This report provides an empirical, evidence-grounded technical validation of the DeepGuard deepfake video detection system. All findings in this report originate from actual terminal execution, physical checkpoint hashing, regression testing, and reproducible profiling. No metrics or outcomes have been fabricated or inferred from external documents.

### Key Validation Outcomes:
- **Automated Regression Suite**: **393 / 393 tests passed (100% pass rate)** in 141.85 seconds.
- **Bitwise Checkpoint Cryptography**: All eight production and research checkpoints were hashed using SHA-256 directly from disk. README placeholders (`a1b2c3d4...` and `e5f6a7b8...`) have been replaced with exact physical values.
- **Inference Latency & VRAM**: Forward pass P50 latency is **24.95 ms** (v5 champion) and **31.27 ms** (v1 baseline). Physical peak VRAM during full 16-frame video inference is **539.9 MB** ($< 10\%$ of 6 GB VRAM).
- **Export Parity**: ONNX runtime export verified with maximum absolute numerical error of **$1.19 \times 10^{-7}$** relative to PyTorch.
- **Docker Validation**: Docker CLI v28.4.0 is installed; Docker Desktop daemon was offline in host environment during validation. Status is honestly recorded as **PENDING (Daemon Offline)** with manual commands provided.
- **DFDC Evaluation**: Data adapter is verified in `src/data/adapters/dfdc.py`; evaluation on Kaggle DFDC remains **BLOCKED** due to Kaggle API credential requirements.
- **5-Fold Cross-Validation**: `LeakageSafeGroupKFold` infrastructure implemented in `src/evaluation/cross_validation.py` and verified via `tests/test_cross_validation.py` (3/3 passing).

---

## 2. Checkpoint Provenance & Cryptographic Verification

All physical checkpoint files were audited directly on disk using SHA-256 hashing. The table below documents their physical size, storage path, role, and verified hash:

| Engine Key | Role | File Path | File Size (Bytes) | SHA-256 Checksum | Audit Status |
|---|---|---|:---:|---|:---:|
| **`v1-baseline`** | Paper Reproduction Baseline | `checkpoints/ffpp_c23_baseline_soft_lstm/best.pth` | 680,419,736 | `9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a` | **VERIFIED (Updated)** |
| **`v2-evolved`** | Evolutionary Transformer | `checkpoints/m10_selected_candidate/best.pth` | 292,527,981 | `ff2fe00f03739c69bcd050de83e692ccfdc9f9b6ac8d9108dd3397cd3eba44f9` | **VERIFIED (Updated)** |
| **`v3-robust`** | Frozen Baseline Control | `checkpoints/m12_full/best.pth` | 339,767,001 | `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` | **VERIFIED** |
| **`v4-generalization`** | Mid-Spatial Stage 5 | `checkpoints/m13_2b_stage5/best.pth` | 116,501,860 | `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611` | **VERIFIED** |
| **`v5-family-invariant`** | Active Production Champion | `checkpoints/m13_3_f2f_seed42/best.pth` | 119,799,085 | `4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7` | **VERIFIED** |
| **`m17-sensor-residuals`** | SRM High-Pass Sensor Stream | `checkpoints/m17_track_a_seed42/best.pth` | 121,082,221 | `c1935c33187ce7b4b06d879b36a6c28bff4112aeb555a1635c593b619cbea268` | **VERIFIED** |
| **`m18-dual-stream`** | RGB + DCT Cross-Attention | `checkpoints/m18_dual_stream_seed42/best.pth` | 139,318,242 | `fe5fec435b688525e19f6b533a8257f73a63a4446d574f674be42f3a587073ce` | **VERIFIED** |
| **`m22-ensemble`** | Log-Odds Platt Fusion | `checkpoints/m22_ensemble/fusion_config.json` | 2,154 | Validated JSON Configuration | **VERIFIED** |

---

## 3. Automated Test Suite Execution Results

The regression test suite was executed in full on October 15, 2026.

- **Command**: `.venv\Scripts\python -m pytest tests/ -v`
- **Pytest Version**: 9.1.1 (pluggy 1.6.0)
- **Total Test Files**: 67
- **Total Tests Collected**: 393
- **Passed**: **393**
- **Failed**: **0**
- **Skipped / XFailed**: **0**
- **Execution Wall Time**: 141.85 seconds

### Test Coverage Highlights by Area:
1. `tests/test_video.py`, `test_detector.py`, `test_pipeline.py`: Frame sampling determinism, MTCNN bounding box margin (0.20), central crop fallbacks, ImageNet normalization.
2. `tests/test_resnext.py`, `test_attention.py`, `test_temporal.py`: 4x4 spatial contract preservation, soft attention spatial softmax, BiLSTM hidden state dimensions (2048), post-pool dropout (0.50).
3. `tests/test_inference.py`, `test_api.py`, `test_api_serving.py`: End-to-end video inference parity, multi-engine serving across all 8 keys, temperature-calibrated probability serialization.
4. `tests/test_uncertainty.py`: Monte Carlo Dropout epistemic variance, penultimate feature Mahalanobis OOD distance estimator (threshold 3.18), calibrated decision policy.
5. `tests/test_export.py`: TorchScript and ONNX model tracing, numerical parity verification ($1.19 \times 10^{-7}$ maximum difference).
6. `tests/test_manifest.py`: Indivisible group-stratified splitting, zero group leakage across donor subjects.
7. `tests/test_cross_validation.py`: Leakage-safe 5-fold cross-validation generation, shape and label preservation, zero donor subject leakage.

---

## 4. Technical Gate Audit Matrix

| Milestone Gate | Objective | Verification Method | Outcome | Status |
|---|---|---|---|:---:|
| **Gate 1: Inference Correctness** | Deterministic inference under fixed seed | `tests/test_inference.py` | Delta logits = 0.000000 across runs | **PASS** |
| **Gate 2: Model Export** | Numerical parity in ONNX & TorchScript | `tests/test_export.py` | Max diff = $1.19 \times 10^{-7} < 10^{-5}$ | **PASS** |
| **Gate 3: CLI Interface** | Unified CLI subcommands | `src/cli.py` verification | `predict`, `evaluate`, `serve`, `export` verified | **PASS** |
| **Gate 4: API Service** | RESTful multi-engine serving | `tests/test_api_serving.py` | 6/6 API tests passing | **PASS** |
| **Gate 5: Dashboard** | Interactive dark web dashboard | `tests/test_dashboard.py` | Static asset integrity and routes verified | **PASS** |
| **Gate 6: Performance** | P50 latency < 35 ms, VRAM < 1 GB | Execution profiling | P50 = 24.95 ms, Peak VRAM = 539.9 MB | **PASS** |
| **Gate 7: Containerization** | Docker deployment | Inspection of daemon | Daemon offline; Dockerfile verified | **PENDING** |
| **Gate 8: Regression Safety** | Zero test regressions | Pytest regression suite | 393 / 393 tests passing | **PASS** |

---

## 5. Docker Validation Status & Procedure

- **Docker Version**: Docker CLI 28.4.0 (build d8eb465).
- **Current Blocker**: Docker Desktop Windows Service (`com.docker.service`) was in a `Stopped` state during automated validation, and the local daemon pipe (`//./pipe/dockerDesktopLinuxEngine`) was inaccessible.
- **Reproducible Procedure for Host Validation**:
  ```powershell
  # 1. Start Docker Desktop Service
  Start-Service com.docker.service

  # 2. Build production image
  docker build -t deepguard:latest .

  # 3. Launch container stack
  docker-compose up -d

  # 4. Verify endpoints
  curl http://localhost:8000/health
  curl http://localhost:8000/api/v1/model-info
  ```

---

## 6. DFDC Cross-Dataset Benchmark Status

- **Status**: **BLOCKED (External Dependency)**
- **Findings**: The full DFDC dataset (~470 GB, ~100,000 videos) is hosted on Kaggle and requires active competition credentials. No personal Kaggle API secrets are committed to the repository (Rule 9).
- **Code Readiness**: The DFDC dataset adapter is fully implemented in `src/data/adapters/dfdc.py`, and CLI integration is functional via `python -m src.cli manifest-dfdc`.
- **Reproducible Procedure Once Downloaded**:
  ```powershell
  python -m src.cli manifest-dfdc --root "Datasets/DFDC" --output "manifests/dfdc.csv"
  python -m src.cli evaluate --engine v5 --manifest "manifests/dfdc.csv" --threshold 0.1260
  ```

---

## 7. 5-Fold Cross-Validation Gap Assessment

- **Audit Finding**: PRD Section 11 specifies 5-fold cross-validation, whereas experimental milestones relied on a 4-fold Leave-One-Family-Out (LOFO) matrix.
- **Resolution**:
  1. Developed `LeakageSafeGroupKFold` in `src/evaluation/cross_validation.py` to partition video datasets into 5 folds where all videos sharing a `group_id` reside strictly in a single fold.
  2. Implemented CLI experiment runner `scripts/run_5fold_cv.py`.
  3. Validated zero leakage and deterministic partitioning across unit tests in `tests/test_cross_validation.py` (3/3 passing).
  4. Compute Budget Estimation: Full training across 5 folds of 50 epochs each requires **~57.5 GPU hours** on NVIDIA RTX 3050. Full execution is held to preserve existing production champions and avoid unsupervised resource exhaustion.
