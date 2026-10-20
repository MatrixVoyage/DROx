# Milestone 21 — Locked Face2Face Failure Analysis & Evidence Attribution Report

**Date**: 2026-10-06 15:23:46 UTC  
**Methodology**: Observational Forensic Attribution (**Zero Model Training**)  
**Minimum-N Rule (N >= 8)**: FAILED (Descriptive Only)  

---

## 1. Multi-Model Cohort & Discordance Breakdown

- **Locked Holdout Size**: $N = 116$ (58 Reals, 58 F2F Fakes)
- **Cohort I (Concordant Detected Fakes)**: $N = 10$ (Classified as fake by all 4 models)
- **Cohort II (Persistent False Negatives / Hard Fakes)**: $N = 1$ (Missed by all 4 models)
- **Cohort III (Concordant Clean Reals)**: $N = 1$
- **Cohort IV (Persistent False Positives)**: $N = 1$
- **M13.5-Win Fakes (M13.5 detected, Dual-Stream missed)**: $N = 12$
- **Dual-Stream-Win Fakes (Dual-Stream detected, M13.5 missed)**: $N = 35$

---

## 2. Pre-Registered Hypothesis Testing Summary

| Hypothesis Family | Primary Physical Descriptors | Cliff's δ | FDR q-value | Negative Control | Final Verdict |
|---|---|---:|---:|---:|:---:|
| **H1_Motion** (`mouth_flow_mag`) | Hard FN: 4.047 vs Det: 4.172 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H1_Motion** (`lip_corner_distance`) | Hard FN: 29.768 vs Det: 68.240 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H1_Motion** (`motion_ratio`) | Hard FN: 0.107 vs Det: 0.056 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H2_Codec** (`dct_hf_energy`) | Hard FN: 0.000 vs Det: 0.000 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H2_Codec** (`video_bitrate_kbps`) | Hard FN: 877.397 vs Det: 770.148 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H3_Scale** (`face_width`) | Hard FN: 81.000 vs Det: 189.000 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H3_Scale** (`face_height`) | Hard FN: 109.000 vs Det: 253.900 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H3_Scale** (`face_area`) | Hard FN: 8829.000 vs Det: 57337.400 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H3_Scale** (`face_fraction_of_frame`) | Hard FN: 0.022 vs Det: 0.080 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H3_Scale** (`inter_eye_distance`) | Hard FN: 37.861 vs Det: 87.067 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H4_Boundary** (`seam_laplacian_energy`) | Hard FN: 4.630 vs Det: 8.912 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H4_Boundary** (`seam_color_delta_e`) | Hard FN: 29.293 vs Det: 44.044 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H5_Semantic** (`mouth_opening_amplitude`) | Hard FN: 0.569 vs Det: 0.200 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H5_Semantic** (`mouth_opening_velocity`) | Hard FN: 0.141 vs Det: 0.046 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |
| **H5_Semantic** (`mouth_motion_cycles`) | Hard FN: 7.000 vs Det: 8.000 | +0.000 | 1.0000 | Confounded (p>=0.05) | **INCONCLUSIVE (N < 8 descriptive fallback)** |

---

## 3. Pairwise Score Discordance: What Drives Champion M13.5's Advantage?

Spearman rank correlation with logit difference $(z_{\text{M13.5}} - z_{\text{M18}})$ across all Face2Face fakes:

| Forensic Descriptor | Spearman ρ vs (z_M13.5 - z_M18) | Physical Interpretation |
|---|---:|---|
| `mouth_flow_mag` | +0.032 | Neutral |
| `dct_hf_energy` | -0.163 | Correlated with M18 win |
| `face_area` | +0.016 | Neutral |
| `seam_laplacian_energy` | +0.020 | Neutral |
| `inter_eye_distance` | +0.026 | Neutral |

---

## 4. Evidence Synthesis & Gating for Milestone 22

### Key Findings:
1. **Cohort II Sample Size**: $N = 1$ persistent false negatives. Under the pre-registered minimum-N rule, sample size is small, requiring descriptive interpretation.
2. **Physical Failure Signatures**: Evaluated across motion, codec, resolution, boundary, and semantic activity.
3. **Architectural Gating**: M22 architecture development will only be authorized if at least one failure mechanism is formally `SUPPORTED` or `PLAUSIBLE` with an actionable physical representation.

**Conclusion**: Observational forensic failure attribution complete without predictive model fitting or architecture roulette.
