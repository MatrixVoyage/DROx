# Milestone 23: Production Serving Architecture & Multi-Engine Comparison Dashboard

## 1. Executive Summary

Milestone 23 elevates the Deepfake Video Detection research framework into an accredited, production-grade serving engine and interactive forensic analysis suite. 

Building upon the empirical findings of **Milestone 22** (which established the M22 Log-Odds Platt Fusion Ensemble as the research champion with 0.8122 F2F AUC, while retaining M13.5 as the production champion under single-model edge resource constraints), Milestone 23:
1. **Unifies All 8 Project Engines** under a single lazy-loaded, thread-safe serving registry.
2. **Implements a Cross-Engine Comparative Arbitration Endpoint** (`POST /api/v1/compare`), computing concordant and discordant multi-domain forensic evidence across Champion M13.5 (static spatial/texture), Baseline M18 (dynamic 3D kinematics), and Fused M22 (calibrated log-odds ensemble).
3. **Upgrades the Interactive Dashboard UI** with side-by-side forensic card analysis, automated modality disagreement narratives, dynamic confidence gauges, and granular kinematic telemetry.
4. **Verifies 100% Repository-Wide Regression Pass Rate** across 382 automated test suites.

---

## 2. Supported Model Registry (`AVAILABLE_ENGINES`)

All model architectures developed across the research lifecycle are now accessible via runtime selection:

| Engine Key | Architecture & Description | Role | Operating Policy Threshold ($\tau$) |
|---|---|---|---|
| `v5-family-invariant` | EfficientNet-B0 + Frequency Residual + Local Texture Patch + TempTrans | **Active Production Champion** | $\tau^* = 0.5000$ |
| `m22-ensemble` | Validation Log-Odds Platt Fusion ($0.50 z_{13} + 0.50 z_{18}$) | **Highest-Accuracy Research Champion** | $\tau_{95}^* = 0.9191$ |
| `m18-dual-stream` | M17 Track A Sensor Trunk + 3D Differential Kinematics + Gated Fusion | Reenactment Research Baseline | $\tau_{95}^* = 0.8821$ (Raw) / $0.8501$ (Platt) |
| `m17-sensor-residuals` | ResNeXt-50 + Multi-Scale SRM Residuals (Scales 1 & 2) | Sensor Noise Research Baseline | $\tau^* = 0.5000$ |
| `v4-generalization` | EfficientNet-B0 + Auxiliary Manipulation Classifier | M13.2b Generalization Baseline | $\tau^* = 0.5000$ |
| `v3-robust` | EfficientNet-B0 + Temporal Transformer (Frozen Immutable) | M12 Accredited Production Baseline | $\tau^* = 0.5000$ |
| `v2-evolved` | EfficientNet-B0 + Temporal Transformer (Initial) | M10 Exploration Baseline | $\tau^* = 0.5000$ |
| `v1-baseline` | ResNeXt-50 + Soft Spatial Attention + BiLSTM | M7 Initial Research Paper Reproduction | $\tau^* = 0.5000$ |

---

## 3. Comparative Arbitration Specification (`POST /api/v1/compare`)

The comparative endpoint accepts a video upload, computes parallel feature extractions across the relevant model streams, and returns a structured consensus assessment:

### Consensus Taxonomies
- **`CONCORDANT_FAKE`**: All active engines evaluate the video above their calibrated decision thresholds ($d_{13} = 1 \land d_{18} = 1 \land d_{22} = 1$). Both spatial manipulation artifacts and kinematic motion anomalies are unanimously confirmed.
- **`CONCORDANT_REAL`**: All active engines evaluate the video below their calibrated decision thresholds ($d_{13} = 0 \land d_{18} = 0 \land d_{22} = 0$). Video exhibits authentic camera noise and natural biometric dynamics.
- **`DISCORDANT`**: Modalities diverge ($d_{13} \neq d_{18}$). Typically indicates subtle reenactment (where spatial blending is smooth but facial kinematics are perturbed) or compression artifacts (where high-frequency residuals are suppressed but face geometry remains genuine). Arbitrated decisively via the M22 log-odds fusion score.
- **`REJECT_OR_UNCERTAIN`**: One or more models trigger safety envelope rejections (e.g., face detection failure, extreme out-of-distribution distance, or elevated MC-dropout variance). Rejection does not imply fake; evidence is insufficient for accredited attribution.

---

## 4. Quality & Regression Verification

- **API Serving Suite** (`tests/test_api_serving.py`): 6/6 tests passing (100%).
- **Calibrated API Suite** (`tests/test_api_calibrated.py`): 3/3 tests passing (100%).
- **Standard API Suite** (`tests/test_api.py`): 5/5 tests passing (100%).
- **Comprehensive Project Regression** (`pytest tests/`): **382 passed, 0 failed, 100% pass rate**.

---

## 5. Deployment Instructions

To launch the unified serving engine locally:
```bash
.venv\Scripts\python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Then navigate to `http://localhost:8000` to interact with the Multi-Engine Technical Dashboard.
