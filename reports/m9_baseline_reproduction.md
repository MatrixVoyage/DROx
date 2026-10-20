# Milestone 9 Gate 1 — Baseline Reproduction Report

**Execution Timestamp**: 2026-09-21T11:56:06.143259+00:00  
**Checkpoint SHA-256**: `9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a`  
**Status**: **PASS**  

## 1. M8 Cache Provenance & Exact Population

- Total Cached Records: `1418`
- FF++ C23 Canonical Test Split: `750` videos (75 groups)
- Celeb-DF v2 Official Test Split: `518` videos
- FF++ C23 Validation Split: `150` videos (15 groups)

## 2. Benchmark Metric Reproduction vs M8 Published Reports

| Benchmark | Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | EER |
|---|---|---|---|---|---|---|---|---|
| **FF++ C23** | Canonical Test | 750 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.4734 | 0.5133 |
| **Celeb-DF v2** | Official Test | 518 | 0.6564 | 0.6564 | 1.0000 | 0.7925 | 0.5244 | 0.5029 |

**Discrepancy Check**: `Recomputed metrics match M8 published baseline within tolerance (1e-4)` (Tolerance: 0.0001)

## 3. Physical Pipeline Verification Subset (10 Videos)

- All 10 physical predictions matched cache within $\epsilon < 0.0001$: **True**
