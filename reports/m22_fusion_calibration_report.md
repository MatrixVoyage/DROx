# Milestone 22 — Decision-Level Evidence Fusion Calibration Report

**Date**: 2026-10-06 15:38:56 UTC  
**Methodology**: Zero Neural Network Retraining (Frozen Models Fitted on Validation Split)  
**Validation Dataset**: FF++ C23 Validation Split ($N = 1490$ clips)  

---

## 1. Standalone Validation Performance

- **Champion M13.5 Alone**: ROC-AUC = **0.8828**
- **Baseline M18 Alone**: ROC-AUC = **0.8745**

---

## 2. Evidence Fusion Strategies Benchmark on Validation

| Strategy | Model 1 Weight (M13.5) | Model 2 Weight (M18) | Bias $b^*$ | Val ROC-AUC | Val $\tau_{95}^*$ | Val $\tau_{99}^*$ |
|---|---:|---:|---:|---:|---:|---:|
| **1. Constrained Linear Logit** | 0.1433 | 0.8567 | +0.7567 | **0.8873** | 0.9367 | 0.9959 |
| **2. Calibrated Log-Odds Platt** | 0.4092 | 0.3590 | +1.4219 | **0.8911** | 0.9191 | 0.9878 |
| **3. Non-Parametric Rank Average** | 0.5000 | 0.5000 | 0.0000 | **0.8892** | 0.4761 | 0.7453 |
| **4. Hierarchical Gated Routing** | Gated ($p_{13} \ge 0.95$) | Gated ($p_{18} \le 0.05$) | — | **0.8844** | 0.9514 | 0.9810 |

---

## 3. Primary Selected Strategy: **strategy_2_log_odds_platt**
- Achieved highest validation ROC-AUC (**0.8911**).
- Locked operating thresholds: $\tau_{95}^* = 0.9191$, $\tau_{99}^* = 0.9878$.
- All fusion parameters locked for evaluation across the 7 pre-registered promotion gates.
