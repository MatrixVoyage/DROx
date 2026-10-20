# Milestone 16 — Step 6 Holdout & OOD Benchmark Report

**Final Promotion Verdict**: `DO NOT PROMOTE -- return to Step 5`

| Dataset / Family | Model | AUROC (95% CI) | Detection Rate | Specificity | ECE | McNemar (b / c / p) | Status |
|---|---|---|---|---|---|---|---|
| `CelebDF_v2_OOD` | M13.5 | 0.7819 | 28.4% | 92.5% | 0.472 | — | — |
| | **M16** | **0.7345** (-0.0470 [-0.12, +0.03]) | **28.4%** | **92.5%** | **0.469** | b=0, c=0, p=1.000 | [FAIL] |
| `FF++_Holdout_Overall` | M13.5 | 0.9404 | 77.1% | 94.4% | 0.164 | — | — |
| | **M16** | **0.9381** (-0.0028 [-0.05, +0.05]) | **77.1%** | **94.4%** | **0.164** | b=0, c=0, p=1.000 | [PASS] |
