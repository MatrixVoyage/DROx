# Milestone 13.2: Unseen-Manipulation Generalization — Experimental Matrix

**M12 Baseline SHA-256**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (Bitwise Frozen Benchmark)  
**Evaluation Benchmark**: Fixed 560-Query Authoritative Benchmark  
**Strict Invariant**: Zero test/Face2Face data leakage. Preprocessing & sampling identical across all models.

---

## 1. Executive Summary & Comparative Matrix

| Evaluation Metric | Frozen M12 Baseline | M13.2a-Frozen (Trunk Frozen) | M13.2a-Joint (Multi-Task Fine-Tuned) | M13.2b-Stage5 (Stage-5 + Penultimate) |
|---|---|---|---|---|
| **Held-Out Face2Face ROC-AUC** | 0.7780 | 0.8146 | 0.7920 | **0.7789** |
| **Face2Face Detection Rate (τ=0.5)** | 15.52% | 24.14% | 41.38% | **50.00%** |
| **In-Domain Seen Fake ROC-AUC** | 0.9519 | 0.9589 | 0.9583 | 0.9493 |
| **In-Domain Seen Fake F1** | 0.7904 | 0.8544 | 0.9129 | 0.9298 |
| **Celeb-DF Cross-Dataset AUC** | 0.7290 | 0.7270 | 0.7140 | 0.7599 |
| **Celeb-DF Cross-Dataset F1** | 0.3830 | 0.4286 | 0.5138 | 0.5045 |
| **Degraded (JPEG Q=40) AUC** | 0.5044 | 0.5065 | 0.5573 | 0.6360 |

---

## 2. Held-Out Face2Face Probability Distributions (Unseen Family Generalization)

| Model | Mean Prob [95% CI] | Median Prob | Detection Rate (τ=0.5) | Interpretation |
|---|---|---|---|---|
| **Frozen M12 Baseline** | 0.1534 [0.0692, 0.2417] | 0.0008 | 15.5% | Predicts Real (Failure) |
| **M13.2a-Frozen** | 0.3565 [0.2705, 0.4411] | 0.3380 | 24.1% | Generalized Detection |
| **M13.2a-Joint** | 0.4145 [0.2867, 0.5360] | 0.0711 | 41.4% | Generalized Detection |
| **M13.2b-Stage5** | 0.5097 [0.3732, 0.6334] | 0.6024 | 50.0% | Generalized Detection |

---

## 3. Computational and Latency Profile

| Model | Total Parameters | P50 Latency (ms) | P95 Latency (ms) | Peak VRAM (MB) |
|---|---|---|---|---|
| **Frozen M12 Baseline** | 28,289,150 | 27.43 ms | 64.94 ms | 486.79 MB |
| **M13.2a-Frozen** | 28,948,611 | 34.41 ms | 68.1 ms | 710.03 MB |
| **M13.2a-Joint** | 28,948,611 | 30.96 ms | 57.58 ms | 933.82 MB |
| **M13.2b-Stage5** | 28,963,460 | 24.95 ms | 38.66 ms | 1156.55 MB |
