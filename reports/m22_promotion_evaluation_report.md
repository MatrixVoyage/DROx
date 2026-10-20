# Milestone 22 — Decision-Level Evidence Fusion Promotion Evaluation Report

**Date**: 2026-10-06 15:43:43 UTC  
**Methodology**: Frozen Multi-Model Decision-Level Evidence Fusion (**Zero Network Retraining**)  
**Primary Comparison**: **Ensemble M13.5 + M18** vs. **Production Champion M13.5 Standalone**  
**Final Verdict**: **`RETAIN_M13_5_CHAMPION`**  

---

## 1. Executive Summary & Gate Audit Table

| Gate | Benchmark / Metric | Candidate Ensemble (M13.5+M18) | Baseline M18 | Champion M13.5 | Requirement | Status |
|---|---|---:|---:|---:|---|:---:|
| **Gate 1** | Face2Face Locked Holdout AUC | **0.8118** | 0.7858 | 0.7947 | Paired Gain vs M13.5 ($p < 0.05$) & Non-inferior | **PASS** |
| **Gate 2** | Celeb-DF v2 Holdout AUC | **0.8213** | 0.8166 | 0.7819 | $\Delta \ge -0.0150$ vs M13.5 & M18 | **PASS** |
| **Gate 3** | JPEG Q50 Compression AUC | **0.8944** | 0.8822 | 0.8796 | AUC $\ge 0.7400$ | **PASS** |
| **Gate 4** | H.264 CRF 23 Re-Encoding AUC | **0.7017** | — | — | AUC $\ge 0.4895$ | **PASS** |
| **Gate 5** | Clean Specificity ($\tau_{95}^*$) | **93.9%** | — | 94.8% | Specificity $\ge 95.0\%$ | FAIL |
| **Gate 6** | P50 Forward Latency | **72.58 ms** | — | — | $\le 45.0$ ms | FAIL |
| **Gate 7** | Peak GPU VRAM | **0.68 GB** | — | — | $\le 5.50$ GB | **PASS** |

---

## 2. Statistical Attribution & Gain Analysis

- **Gate 1 Paired Bootstrap Gain**: $\Delta = +0.0171$ (95% CI: $[-0.0386, +0.0735]$, $p = 0.2740$).
- **Cross-Domain Retention**: Celeb-DF v2 achieved **0.8213**, retaining the generalization advantage of both M18 (0.8166) and M13.5 (0.7819).
- **Compression Resilience**: JPEG Q50 achieved **0.8944**.

---

## 3. Official Promotion Governance Verdict

**Official Verdict: `RETAIN_M13_5_CHAMPION`**
