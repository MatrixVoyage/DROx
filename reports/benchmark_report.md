# Deepfake Detection Benchmark Report - Milestone 6.1

> [!NOTE]
> - All entries tagged with **[Ref]** correspond to external published values reported in *Lal et al. (2025)* (Scientific Reports).
> - All project-generated values correspond strictly to executed code experiments on physical datasets with zero fabrication.
> - **Subset Distinction**: The project evaluation results below reflect a **30-video physical validation subset** (160-video physical train/val/test subset), NOT the full 5,000-video FF++ C23 benchmark.
> - **DFDC Status**: Marked as **PENDING — DATA ACCESS UNAVAILABLE**. External physical dataset access is pending; no fabrication, scraping, or synthetic replacement was performed.

## 1. In-Domain Benchmark: FaceForensics++ (FF++) C23

| Model / Source | Scale / Protocol | Accuracy | Precision | Recall | F1-Score | ROC-AUC | EER |
|---|---|---|---|---|---|---|---|
| Lal et al. (2025) Proposed [Ref] | Full Benchmark (Paper) | 0.9288 ± 0.012 | 0.940 ± 0.010 | 0.943 ± 0.009 | 0.944 ± 0.008 | 0.962 ± 0.008 | 4.2 ± 0.5% |
| ResNeXt-50 alone [Ref] | Full Benchmark (Paper) | 0.8900 ± 0.015 | 0.880 ± 0.020 | 0.870 ± 0.014 | 0.875 ± 0.012 | 0.912 ± 0.010 | 8.5 ± 1.0% |
| LSTM temporal only [Ref] | Full Benchmark (Paper) | 0.8920 ± 0.014 | 0.890 ± 0.015 | 0.882 ± 0.013 | 0.885 ± 0.011 | 0.920 ± 0.009 | 7.8 ± 0.9% |
| EfficientNet-B4 [Ref] | Full Benchmark (Paper) | 0.9240 ± 0.013 | 0.940 ± 0.010 | 0.908 ± 0.011 | 0.924 ± 0.009 | 0.950 ± 0.007 | 5.1 ± 0.6% |
| Vision Transformer (ViT) [Ref] | Full Benchmark (Paper) | 0.9150 ± 0.014 | 0.920 ± 0.012 | 0.910 ± 0.010 | 0.915 ± 0.009 | 0.945 ± 0.008 | 5.5 ± 0.7% |
| **ffpp_c23_baseline_soft_lstm (Project Run)** | **30-Video Physical Subset** | **0.8000** | **0.8000** | **1.0000** | **0.8889** | **0.3958** | **66.67%** |

*Note on FF++ Test Metrics: On this 30-sample test subset (6 real, 24 fake), all samples received positive logits ($p \ge 0.5$) under the 12-epoch subset model, yielding $\text{TP}=24, \text{FP}=6, \text{TN}=0, \text{FN}=0$. The classification metrics ($\text{Acc}=0.8000, \text{Prec}=0.8000, \text{Rec}=1.0000, \text{F1}=0.8889$) follow directly from this matrix. The ROC-AUC of 0.3958 reflects the continuous probability ranking across the 6 real and 24 fake samples (Mann-Whitney $U = 57.0 / 144$).*

## 2. Systematic Ablation Study (FF++ C23 Literature Reference)

| Architecture Variant | Accuracy | ROC-AUC | EER | Source |
|---|---|---|---|---|
| Full Model [Ref] | 0.9288 ± 0.012 | 0.962 ± 0.008 | 4.2 ± 0.5% | Published (Lal et al., 2025) |
| w/o Dual Attention [Ref] | 0.8900 ± 0.015 | 0.917 ± 0.010 | 8.7 ± 1.1% | Published (Lal et al., 2025) |
| w/o BiLSTM (CNN only) [Ref] | 0.9050 ± 0.014 | 0.935 ± 0.009 | 6.5 ± 0.8% | Published (Lal et al., 2025) |
| w/o Self-Attention Gate [Ref] | 0.9150 ± 0.013 | 0.945 ± 0.008 | 5.3 ± 0.7% | Published (Lal et al., 2025) |

## 3. Cross-Dataset Generalization (Trained on FF++ C23)

| Target Benchmark | Scale | Accuracy | ROC-AUC | EER | $\Delta$ AUC vs Val | $\Delta$ AUC vs Test | Paper Ref AUC [Ref] |
|---|---|---|---|---|---|---|---|
| **Celeb-DF v2** | 8 Physical Videos (4 Real, 4 Fake) | 0.5000 | 0.4375 | 50.00% | -0.2014 | +0.0417 | 0.950 ± 0.009 |
| **DFDC** | Unvetted / Inaccessible | PENDING | PENDING | PENDING | N/A | N/A | 0.882 ± 0.011 |

*Note on Delta Metrics: $\Delta\text{AUC vs Val} = 0.4375 - 0.6389 = -0.2014$ compares the cross-dataset score to the in-domain validation performance stored in the checkpoint. $\Delta\text{AUC vs Test} = 0.4375 - 0.3958 = +0.0417$ compares against the in-domain test subset score.*