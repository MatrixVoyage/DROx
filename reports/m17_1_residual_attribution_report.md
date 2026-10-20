# Milestone 17.1 (M17.1) — Forensic Attribution, Scale Disentanglement & Codec Matching
## Empirical Attribution of M17 Track A Multi-Scale Residuals

**Date**: 2026-10-05  
**Model Under Test**: M17 Track A Candidate (`checkpoints/m17_track_a_seed42/best.pth`, Frozen)  
**Control Baseline**: M13.5 Champion Control (`checkpoints/m13_3_f2f_seed42/best.pth`, Frozen)  
**Calibrated Temperature**: $T = 3.2254$  

---

## 1. Core Answers to the 5 Research Questions

### Q1: Which residual scales provide the Celeb-DF improvement?
Under the frozen model, comparative scale ablations reveal:
- **Full Pyramidal Residuals ($S1+S2+S3$)**: **0.8304** ROC-AUC ($\Delta = +0.0490$ vs M13.5, $p = 0.0210$)
- **Scale 1 Only ($112\times 112$ SRM+Bayar)**: **0.8196** ROC-AUC ($\Delta = -0.0107$ vs Full, Gate Mean = 0.1009)
- **Scale 2 Only ($56\times 56$ Bayar Resampling)**: **0.8187** ROC-AUC ($\Delta = -0.0119$ vs Full, Gate Mean = 0.1020)
- **Scale 3 Only ($28\times 28$ Macro Block)**: **0.7958** ROC-AUC ($\Delta = -0.0347$ vs Full, Gate Mean = 0.1017)

### Q2: Which scales survive JPEG Q=50 compression?
- **Full Model under Q=50**: **0.6250** (Control = 0.5793, $\Delta = +0.0447$)
- **Scale 1 Only under Q=50**: **0.5863**
- **Scale 2 Only under Q=50**: **0.6315**
- **Scale 3 Only under Q=50**: **0.5803**

### Q3: Which scales contribute to Face2Face detection?
- **Full Model on F2F**: **0.7351** (Control = 0.7946)
- **Scale 1 Only on F2F**: **0.7394**
- **Scale 2 Only on F2F**: **0.7362**
- **Scale 3 Only on F2F**: **0.7267**

### Q4 & Q5: Does Track A survive matched codec, resolution, and bitrate controls?
1. **Resolution-Matched Stratum ($W \ge 900, H = 500$)**:
   - Cand AUC: **0.7705** vs Control: **0.7582** ($\Delta = +0.0125$, 95% CI: [-0.0941, +0.1104], $p = 0.4040$)
2. **Bitrate-Matched Stratum ($900 \le \text{Bitrate} \le 1400$ kbps)**:
   - Cand AUC: **0.8238** vs Control: **0.8050** ($\Delta = +0.0203$, 95% CI: [-0.0459, +0.0882], $p = 0.2780$)
3. **Uniform H.264 CRF 23 Re-Encoding (All 511 Clips)**:
   - Cand AUC: **0.5803** vs Control: **0.5984** ($\Delta = -0.0179$, 95% CI: [-0.0814, +0.0435], $p = 0.7160$)
   - S1 Re-encoded: 0.5238 | S2 Re-encoded: 0.5446 | S3 Re-encoded: 0.4904

---

## 2. Complete Frozen-Weight Inference Scale Ablation Matrix

| Scale Configuration | Celeb-DF AUC (PR-AUC) | $\Delta$ vs Full | $\Delta$ vs M13.5 | JPEG Q=50 AUC | F2F AUC | FF++ Mixed AUC | Gate Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full_S123** | 0.8304 (0.8930) | +0.0000 | +0.0490 | 0.6250 | 0.7351 | 0.8473 | 0.100 $\pm$ 0.005 |
| **S1_Only** | 0.8196 (0.8891) | -0.0107 | +0.0384 | 0.5863 | 0.7394 | 0.8394 | 0.101 $\pm$ 0.005 |
| **S2_Only** | 0.8187 (0.8852) | -0.0119 | +0.0372 | 0.6315 | 0.7362 | 0.8418 | 0.102 $\pm$ 0.005 |
| **S3_Only** | 0.7958 (0.8793) | -0.0347 | +0.0143 | 0.5803 | 0.7267 | 0.8285 | 0.102 $\pm$ 0.005 |
| **S1_S2** | 0.8388 (0.8961) | +0.0085 | +0.0575 | 0.6258 | 0.7409 | 0.8512 | 0.101 $\pm$ 0.005 |
| **S1_S3** | 0.8194 (0.8884) | -0.0109 | +0.0381 | 0.6005 | 0.7310 | 0.8396 | 0.101 $\pm$ 0.005 |
| **S2_S3** | 0.8143 (0.8835) | -0.0164 | +0.0326 | 0.6375 | 0.7394 | 0.8408 | 0.102 $\pm$ 0.005 |
| **M13.5 Control (`v5`)** | 0.7806 (0.8746) | -0.0498 | 0.0000 | 0.5793 | 0.7946 | 0.8344 | 0.000 $\pm$ 0.000 |

---

## 3. Scientific Conclusions for M18 Foundation

1. **Leakage vs. Forensic Evidence Finding**:
   - Evaluates whether Track A's advantage survives when resolution, bitrate, and codec are strictly matched or re-encoded.
2. **Scale Hierarchy**:
   - Identifies whether the cross-domain signal is concentrated in high frequencies (Scale 1), resampling boundaries (Scale 2), or macro blocks (Scale 3).
3. **Implications for M18 Architecture**:
   - Confirms which residual scales provide genuine invariant forensic grounding before attaching targeted reenactment supervision.
