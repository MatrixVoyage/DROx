# Milestone 20.2 — Pyramidal Dual-Stream Detector: 7-Gate Promotion Report

**Date**: 2026-10-06 15:01:27 UTC  
**Audit Verdict**: `RETAIN_M13_5_CHAMPION`  
**Candidate Model**: `M20PyramidalDualStreamDetector` (`checkpoints/m20_pyramidal_seed42/best.pth`)  
**Production Champion**: `M13.5 (v5-family-invariant)` (`checkpoints/m13_3_f2f_seed42/best.pth`)  
**Research Baseline**: `M18 Dual-Stream` (`checkpoints/m18_dual_stream_seed42/best.pth`)  

---

## 1. Pre-Registered 7-Gate Audit Scorecard

| Gate | Benchmark / Metric | Candidate M20 | Baseline M18 | Champion M13.5 | Requirement | Status |
|---|---|---:|---:|---:|---|:---:|
| **Gate 1** | F2F ROC-AUC | 0.7836 | 0.7802 | 0.7946 | Paired Gain vs M18 (p<0.05) & Non-inferior vs M13.5 | FAIL |
| **Gate 2** | Celeb-DF v2 ROC-AUC | 0.7937 | 0.7945 | 0.7806 | Delta >= -0.0150 vs M13.5 & M18 | PASS |
| **Gate 3** | JPEG Q50 ROC-AUC | 0.7538 | 0.7535 | 0.7428 | AUC >= 0.7500 | PASS |
| **Gate 4** | H.264 CRF 23 ROC-AUC | 0.5742 | 0.5277 | 0.5984 | AUC >= 0.4895 | PASS |
| **Gate 5** | Clean Specificity (tau_95*) | 94.8% | 1.7% | 94.8% | Specificity >= 95.0% | FAIL |
| **Gate 6** | P50 Forward Latency | 58.81 ms | — | — | <= 35.0 ms | FAIL |
| **Gate 7** | Peak GPU VRAM | 0.76 GB | — | — | <= 5.50 GB | PASS |

---

## 2. Statistical Attribution & Significance

- **Gate 1 Delta vs M18 Baseline**: +0.0024 (95% CI: [-0.0314, +0.0380], p = 0.4430)
- **Gate 1 Delta vs Production Champion**: -0.0105 (95% CI: [-0.0914, +0.0694], p = 0.6070)
- **Gate 2 Delta vs M18 Baseline**: -0.0007 (95% CI: [-0.0130, +0.0112], p = 0.5430)
- **Gate 2 Delta vs Production Champion**: +0.0113 (95% CI: [-0.0310, +0.0550], p = 0.3030)

---

## 3. Governance Verdict & Next Steps

**Audit Verdict**: `RETAIN_M13_5_CHAMPION`

- If `PROMOTE_TO_CHAMPION`: M20.2 satisfies all 7 promotion gates, improving reenactment discrimination without regression on cross-domain, compression, or latency budgets.
- If `RETAIN_M13_5_CHAMPION`: M13.5 remains the undefeated active production champion. M20.2 is archived as a valuable empirical research milestone.
