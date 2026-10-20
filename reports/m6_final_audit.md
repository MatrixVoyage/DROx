# Milestone 6 / M6.1 Final Engineering & Experimental Audit Report

**Audit Date:** 2026-09-21  
**Project:** Deepfake Video Detection Research Platform  
**Audited Artifacts:** Milestone 6 Orchestration, Milestone 6.1 Real-Data Validation Gate, Checkpoints, Manifests, Tensors, Metric Pipelines, Reports.

---

## 1. Audit Status

**STATUS:** **PASS WITH CLARIFICATIONS**

The current Milestone 6 / M6.1 implementation, experiment pipeline, and real-data validation results were subjected to a strict independent audit. All reported metrics were independently reproduced directly from saved tensors, manifest splits, and the trained model checkpoint (`best.pth`) without retraining. No code bugs, label inversion errors, or group leakages were detected. Clarifications and corrections were applied to reporting terminology to guarantee explicit distinction between physical subset validation experiments and full canonical benchmark runs.

---

## 2. Dataset Availability

| Dataset | Physical Files on Disk | Canonical Manifest | Subset Evaluated | Status |
|---|---|---|---|---|
| **FaceForensics++ C23** | 7,000 `.mp4` files (16.69 GB) | 5,000 videos | 160 videos (100 train, 30 val, 30 test) | **Physically Available & Validated** |
| **Celeb-DF v2** | 6,529 `.mp4` files (28.75 GB) | 6,529 videos | 8 videos (4 real, 4 fake) | **Physically Available & Validated** |
| **DFDC** | None (0 files) | N/A | None | **PENDING — DATA ACCESS UNAVAILABLE** |

**Enforcement of Rule 10:** No scraping, unverified mirrors, synthetic data substitution, or fabricated metrics were permitted for DFDC. DFDC remains explicitly marked as pending external access.

---

## 3. Manifest Verification

1. **Canonical FF++ C23 Manifest (`manifests/ffpp_c23.csv`):**
   - Contains 5,000 rows across pristine (1,000 original videos) and 4 primary manipulation methods (Deepfakes: 1,000; Face2Face: 1,000; FaceSwap: 1,000; NeuralTextures: 1,000).
   - Validated schema compliance via `validate_manifest_schema`.
   - Verified split distribution: 3,600 train, 700 validation, 700 test.
2. **FF++ C23 M6 Subset Manifest (`manifests/ffpp_c23_m6_subset.csv`):**
   - Contains exactly 160 rows.
   - Splits: 100 train (80 fake, 20 real), 30 validation (24 fake, 6 real), 30 test (24 fake, 6 real).
   - Perfect method balancing: each group contains 2 original videos and 8 manipulated derivatives (2 Deepfakes, 2 Face2Face, 2 FaceSwap, 2 NeuralTextures).
3. **Canonical Celeb-DF v2 Manifest (`manifests/celebdf_v2.csv`):**
   - Contains 6,529 rows (590 Celeb-real, 5,639 Celeb-synthesis, 300 YouTube-real).
   - Official test list (`List_of_testing_videos.txt`) confirmed at exactly 518 videos (178 real, 340 fake).
4. **Celeb-DF Gate 3 Subset Manifest (`manifests/celebdf_gate3_subset.csv`):**
   - Contains 8 official test videos: 4 real (YouTube-real / Celeb-real), 4 fake (Celeb-synthesis).

---

## 4. Split and Group Leakage Verification

Mutual-pair grouping was audited to guarantee that pristine source videos and their manipulated derivatives never appear across separate splits:

$$\text{group\_id} = \min(s, t)$$

Where $s$ and $t$ are the numerical source video indices extracted from the filename pattern (e.g. `001_002.mp4` $\implies \min(1, 2) = 1$).

### Recomputed Group Intersections:
- **Canonical FF++ (`manifests/ffpp_c23.csv`):**
  - $\text{Train Groups} \cap \text{Val Groups} = \emptyset$ (0 overlap)
  - $\text{Train Groups} \cap \text{Test Groups} = \emptyset$ (0 overlap)
  - $\text{Val Groups} \cap \text{Test Groups} = \emptyset$ (0 overlap)
- **Subset FF++ (`manifests/ffpp_c23_m6_subset.csv`):**
  - $\text{Train Groups} \cap \text{Val Groups} = \emptyset$ (0 overlap)
  - $\text{Train Groups} \cap \text{Test Groups} = \emptyset$ (0 overlap)
  - $\text{Val Groups} \cap \text{Test Groups} = \emptyset$ (0 overlap)

**Result:** **ZERO GROUP LEAKAGE DETECTED (PASS)**.

---

## 5. Training Protocol Verification

- **Experiment Run:** `runs/ffpp_c23_baseline_soft_lstm_20260921_003348`
- **Model:** `DeepfakeDetector` (ResNeXt-50 layer2 freeze + Soft Attention + 1-layer LSTM 2048-dim + Binary Linear Head).
- **Optimizer:** Adam ($\text{lr}=1\times 10^{-3}$, $\beta=(0.9, 0.999)$, $\text{weight\_decay}=1\times 10^{-5}$).
- **Scheduler:** `ReduceLROnPlateau` (mode: `max`, factor: 0.1, patience: 5, monitored: `val_auc`).
- **Precision:** Mixed Precision (CUDA AMP via `torch.amp.autocast`).
- **Batch Size:** 4 with 2 gradient accumulation steps (effective batch size: 8).
- **Training Duration:** Triggered early stopping at Epoch 12 (patience 5 without improvement beyond Epoch 7).
- **Data Isolation:** Verified that training loop and checkpoint selector accessed strictly `split == 'train'` and `split == 'val'`. The test split was completely excluded until post-training evaluation.

---

## 6. Checkpoint Verification

- **Checkpoint File:** `checkpoints/ffpp_c23_baseline_soft_lstm/best.pth`
- **Monitored Metric:** `val_auc`
- **Validation History Inspection:**
  - Epoch 1: $\text{val\_auc} = 0.3403$
  - Epoch 2: $\text{val\_auc} = 0.4306$
  - Epoch 4: $\text{val\_auc} = 0.5069$
  - Epoch 7: $\text{val\_auc} = 0.6389$ **[BEST]**
  - Epochs 8–12: Did not exceed 0.6389.
- **Checkpoint Metadata:** `best.pth` contains `epoch: 7`, `best_metric: 0.6388888888888888`, and valid model/optimizer/scaler states.
- **Verification:** `best.pth` was selected **strictly and exclusively** on validation set performance.

---

## 7. FF++ Metric Reproduction

The test evaluation was re-executed independently on the 30 physical test samples (`processed/ffpp_c23/test`) using `best.pth` in `torch.inference_mode()`:

| Metric | Stored / Reported | Independently Recomputed | Match Status |
|---|---|---|---|
| **Loss** | 0.5586 | 0.5586 | **EXACT MATCH** |
| **Accuracy** | 0.8000 | 0.8000 | **EXACT MATCH** |
| **Precision** | 0.8000 | 0.8000 | **EXACT MATCH** |
| **Recall** | 1.0000 | 1.0000 | **EXACT MATCH** |
| **F1-Score** | 0.8889 | 0.8889 | **EXACT MATCH** |
| **ROC-AUC** | 0.395833 | 0.395833 | **EXACT MATCH** |
| **EER** | 0.666667 | 0.666667 | **EXACT MATCH** |

---

## 8. Confusion Matrix Verification

### Saved vs Reproduced Confusion Matrix:
- **True Positives (TP):** 24
- **False Positives (FP):** 6
- **True Negatives (TN):** 0
- **False Negatives (FN):** 0
- **Total Samples:** 30 (24 Fake, 6 Real)

### Mathematical Consistency Check:
1. $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N} = \frac{24 + 0}{30} = \frac{24}{30} = 0.8000$
2. $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{24}{24 + 6} = \frac{24}{30} = 0.8000$
3. $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{24}{24 + 0} = \frac{24}{24} = 1.0000$
4. $\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.8000 \times 1.0000}{0.8000 + 1.0000} = \frac{1.6}{1.8} = \frac{8}{9} = 0.888888... \approx 0.8889$

**Finding:** The reported metrics follow with complete mathematical necessity from the confusion matrix.  
**Threshold Behavior:** At the default classification threshold ($\theta = 0.5$), the model predicted **all 30 test samples as Class 1 (Deepfake)** because all output probabilities fall in the range $[0.7086, 0.9143]$.

---

## 9. AUC Sanity Check & Deep-Dive Investigation

The audit rigorously investigated why the reported ROC-AUC is $0.3958$ despite an accuracy of $80\%$.

### Hypotheses Tested:
- **Label Inversion Bug?** **NO.** Class 0 is confirmed real, Class 1 is confirmed fake across dataset adapters, manifests, and data loaders.
- **Logit / Probability Formula Bug?** **NO.** Logits are converted via `torch.sigmoid(logits)`. No `1 - p` inversion exists.
- **Metric Input Bug?** **NO.** Raw continuous probabilities and binary labels are correctly passed to `sklearn.metrics.roc_auc_score(y_true, probs)`.

### Exact Mathematical Cause (Mann-Whitney U Analysis):
ROC-AUC measures the probability that a randomly selected fake sample has a higher score than a randomly selected real sample across all possible decision thresholds:

$$\text{AUC} = \frac{U}{N_{\text{fake}} \times N_{\text{real}}} = \frac{U}{24 \times 6} = \frac{U}{144}$$

Inspecting the actual continuous probabilities produced by `best.pth`:
- **Real samples ($N=6$, label 0):** Probabilities = `[0.7299, 0.7338, 0.9143, 0.9142, 0.9142, 0.9143]`, Mean = $0.8534$, Median = $0.9142$.
- **Fake samples ($N=24$, label 1):** Mean = $0.8408$, Median = $0.9122$.

Pairwise comparisons across all $24 \times 6 = 144$ combinations:
- **Concordant pairs ($p_{\text{fake}} > p_{\text{real}}$):** 57
- **Discordant pairs ($p_{\text{real}} > p_{\text{fake}}$):** 87
- **Tied pairs ($p_{\text{real}} = p_{\text{fake}}$):** 0

$$\text{AUC} = \frac{57}{144} = 0.3958333...$$

### Conclusion:
The baseline model, trained for only 12 epochs on a 100-video subset with 80% fake prevalence, developed an overall positive logit bias ($z > 0.88 \implies p > 0.70$). At threshold 0.5, all samples are predicted fake, matching the 80% class prevalence. However, on this specific 30-sample test set, 4 of the 6 real samples received very high probabilities (~0.914), causing real samples to slightly outrank fake samples in continuous score. 
**The value $\text{ROC-AUC} = 0.3958$ is mathematically correct and verified.**

---

## 10. EER Verification

- **Reported EER:** $0.6667$ (66.67%)
- **Recomputed EER:** $0.6667$ via `compute_eer` using linear crossing interpolation on the ROC curve ($d = \text{FPR} - \text{FNR} = 0$).
- **Explanation:** In an inverted-ranking regime ($\text{AUC} < 0.5$), the standard ROC operating curve shifts below the diagonal line, producing an EER above $0.5$ ($1.0 - \text{AUC}_{\text{approx}} \approx 0.60 - 0.67$). Verified mathematically.

---

## 11. Celeb-DF Cross-Dataset Verification

The audit independently verified the cross-dataset evaluation of `best.pth` on the 8 physical Celeb-DF test videos (`processed/celebdf_gate3/test`):

| Metric | Reported | Recomputed | Status |
|---|---|---|---|
| **Samples** | 8 (4 Real, 4 Fake) | 8 | **MATCH** |
| **Accuracy** | 0.5000 | 0.5000 | **MATCH** |
| **Precision** | 0.5000 | 0.5000 | **MATCH** |
| **Recall** | 1.0000 | 1.0000 | **MATCH** |
| **F1-Score** | 0.6667 | 0.6667 | **MATCH** |
| **ROC-AUC** | 0.4375 | 0.4375 | **MATCH** |
| **EER** | 0.5000 | 0.5000 | **MATCH** |
| **Confusion Matrix** | TP=4, FP=4, TN=0, FN=0 | TP=4, FP=4, TN=0, FN=0 | **MATCH** |

- **Inference Guarantee:** Executed in `torch.inference_mode()`. Zero optimizer steps, zero gradients, zero adaptation.

---

## 12. $\Delta\text{AUC}$ and $\Delta\text{EER}$ Verification

The audit identified an important reporting nuance in how generalizations deltas were calculated in `cross_eval_celebdf.json` vs `reports/benchmark_report.md`:

1. **Delta vs In-Domain Validation Metric (Stored in `best.pth`):**
   - In-Domain Validation AUC = $0.6389$
   - In-Domain Validation EER = $0.3333$
   $$\Delta\text{AUC}_{\text{val}} = \text{AUC}_{\text{celeb}} - \text{AUC}_{\text{val}} = 0.4375 - 0.6389 = -0.2014$$
   $$\Delta\text{EER}_{\text{val}} = \text{EER}_{\text{celeb}} - \text{EER}_{\text{val}} = 0.5000 - 0.3333 = +0.1667$$
   *(This exact computation was saved in `runs/cross_eval_celebdf.json`).*

2. **Delta vs In-Domain Test Subset Metric:**
   - In-Domain Test AUC = $0.3958$
   - In-Domain Test EER = $0.6667$
   $$\Delta\text{AUC}_{\text{test}} = \text{AUC}_{\text{celeb}} - \text{AUC}_{\text{test}} = 0.4375 - 0.3958 = +0.0417$$
   $$\Delta\text{EER}_{\text{test}} = \text{EER}_{\text{celeb}} - \text{EER}_{\text{test}} = 0.5000 - 0.6667 = -0.1667$$

**Correction Made:** Both delta baselines are now explicitly labeled in `reports/benchmark_report.md` to prevent any confusion.

---

## 13. Subset vs Full-Benchmark Classification

**CRITICAL DESIGNATION:**
- Canonical FF++ C23 contains 5,000 videos.
- Canonical Celeb-DF v2 contains 6,529 videos (with 518 official test videos).
- **Milestone 6.1 Executed Scale:**
  - **FF++ C23:** 160 physical videos preprocessed (100 train, 30 val, 30 test).
  - **Celeb-DF v2:** 8 physical videos evaluated.

**Classification:** This experiment is strictly classified as a:
> **"30-video physical validation subset run" / "Real-data pipeline validation"**

It must **never** be cited as the full FF++ C23 benchmark evaluation. All documents and tables have been updated with this explicit phrasing.

---

## 14. Provenance Verification

All experimental artifacts were verified against the runtime execution environment:
- **Python Version:** 3.11.9
- **PyTorch Version:** 2.6.0+cu124
- **CUDA Version:** 12.4
- **GPU Device:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (6,143.5 MB VRAM)
- **Random Seed:** 42
- **Git Commit:** Recorded as `unknown` (workspace was not initialized as a Git repository).
- **Run Directory:** `runs/ffpp_c23_baseline_soft_lstm_20260921_003348`
- **Config Hash / Snapshot:** Saved in `config.yaml` within the run directory.

---

## 15. Reproducibility Result

The independent audit script `scripts/audit_m6_independent.py` successfully loaded `best.pth` and evaluated the saved tensor inputs without retraining. All metrics, losses, and confusion matrix entries reproduced with **100% exact numerical match**.

---

## 16. Test Suite Result

The full automated regression suite was executed:
```pwsh
pytest tests/ -q
```
- **Total Tests:** 143
- **Passed:** 143
- **Failed:** 0
- **Pass Rate:** **100%**
- **Execution Time:** ~95.5 seconds

---

## 17. Corrections Made

1. **Benchmark Report Terminology (`reports/benchmark_report.md`):**
   - Replaced ambiguous "Project Run" label with "30-Video Physical Subset".
   - Added explicit callout noting the subset scale versus the 5,000-video canonical benchmark.
2. **Generalization Delta Clarification:**
   - Disentangled $\Delta\text{AUC}$ vs validation ($ -0.2014 $) from $\Delta\text{AUC}$ vs test ($ +0.0417 $) with explicit documentation in both markdown and JSON reports.
3. **Literature Reference Attribution:**
   - Enforced separate columns and strict `[Ref]` badges for *Lal et al. (2025)* published metrics.

---

## 18. Remaining Limitations

1. **Subset Bias:** Training for only 12 epochs on a small 100-video subset with 80% fake prevalence caused the model to develop an overall positive logit bias ($p > 0.5$ for all test samples).
2. **Sample Scale:** While sufficient to validate the entire data processing, GPU training, AMP, checkpointing, and evaluation pipeline end-to-end on real videos, full benchmark convergence will require training across the full 5,000-video canonical set.
3. **Celeb-DF Scale:** Evaluation on Celeb-DF v2 was conducted on 8 test samples to validate the cross-dataset pipeline; evaluating all 518 official test videos remains future compute work.

---

## 19. DFDC Status

**DFDC Status:** **PENDING — DATA ACCESS UNAVAILABLE**  
The physical DFDC dataset is not present locally on disk. Under the project research integrity rules, no synthetic data, unverified mirrors, or fabricated values were introduced. This does not invalidate the completed FF++ C23 $\to$ Celeb-DF v2 pipeline validation; it simply leaves the DFDC branch pending legitimate dataset acquisition.

---

## 20. Recommended Milestone 7 Handoff Status

**RECOMMENDATION:** **READY FOR MILESTONE 7**

All prerequisites for Milestone 7 handoff are satisfied:
1. Zero unresolved correctness issues or bugs in the M1–M6 pipeline.
2. Complete mathematical verification of all test and cross-dataset metrics.
3. Transparent, unambiguous labeling of subset experiments.
4. Zero fabricated or estimated data.
5. All 143 automated unit and regression tests passing.
6. The frozen M1–M5 contracts remain intact.

The project is ready to proceed to **Milestone 7: Video Inference Service, Export & Deployment Packaging**.
