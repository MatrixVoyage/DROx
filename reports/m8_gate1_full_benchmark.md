# Milestone 8 Gate 1 — Large-Scale Physical Benchmark Report

**Execution Timestamp**: 2026-09-21 16:21:01  
**Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.0 GB VRAM)  
**PyTorch Version**: 2.6.0+cu124 (CUDA: 12.4)  
**Checkpoint SHA-256**: `9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a`  

## 1. FaceForensics++ C23 Canonical Test Split

- Total requested: 750 physical videos (75 groups)
- Evaluated: 750 (Real: 150, Fake: 600)
- Missing / Failed / Preprocessing errors: 0 / 0 / 0

| Metric | In-Domain FF++ Test |
|---|---|
| **LOSS** | None |
| **ACCURACY** | 0.8 |
| **PRECISION** | 0.8 |
| **RECALL** | 1.0 |
| **F1** | 0.888888888888889 |
| **ROC_AUC** | 0.4734166666666667 |
| **EER** | 0.5133333333333333 |
| **CONFUSION_MATRIX** | {'tp': 600, 'fp': 150, 'tn': 0, 'fn': 0} |
| **NUM_SAMPLES** | 750 |

## 2. Celeb-DF v2 Official Test Split

- Total requested: 518 physical videos (Official test manifest)
- Evaluated: 518 (Real: 178, Fake: 340)
- Missing / Failed / Preprocessing errors: 0 / 0 / 0

| Metric | Cross-Dataset Celeb-DF v2 |
|---|---|
| **LOSS** | None |
| **ACCURACY** | 0.6563706563706564 |
| **PRECISION** | 0.6563706563706564 |
| **RECALL** | 1.0 |
| **F1** | 0.7925407925407926 |
| **ROC_AUC** | 0.5243555849306014 |
| **EER** | 0.5029411764705882 |
| **CONFUSION_MATRIX** | {'tp': 340, 'fp': 178, 'tn': 0, 'fn': 0} |
| **NUM_SAMPLES** | 518 |
