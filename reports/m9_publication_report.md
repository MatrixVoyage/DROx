# Milestone 9 Comprehensive Scientific Publication Report

**Publication Date**: 2026-09-21 11:56:08 UTC  
**Checkpoint SHA-256**: `9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a` (100% Bitwise Verified Untouched)  

---

> [!IMPORTANT]
> ### Core Scientific Finding & Methodological Distinction
> **“All gates PASS” does not mean “the detector performs well.”**
>
> It signifies that the Milestone 9 scientific evaluation procedure, diagnostic protocols, controlled ablations, statistical resampling, and verification gates passed with 100% compliance.
>
> The empirical results themselves diagnose and document **substantial discrimination limitations** in the frozen baseline detector:
> - **Near-chance ROC-AUC values**: `0.4734` on in-domain FF++ C23 test; `0.5244` on cross-dataset Celeb-DF v2 test.
> - **False Positive Rate of 1.0000 at default threshold**: At $\theta=0.50$, all 328 pristine test videos across both benchmark populations are classified as manipulated.
> - **Heavy Distribution Overlap**: Non-parametric metrics (Wasserstein distance `0.0526`, Cliff's delta `-0.0638`, Kolmogorov-Smirnov $p = 0.1709$) indicate near-complete overlap between pristine and manipulated score distributions. Nominal test accuracy (80.0% FF++, 65.6% Celeb-DF) is driven by class imbalance and a right-shifted score distribution rather than true discriminative capability.

---

## Executive Gate Audit & Scientific Interpretations

| Gate | Title | Status | Scientific Interpretation |
|---|---|---|---|
| **G1** | Baseline Reproduction & Provenance | **PASS** | M8 benchmark results were reproduced within the predefined numerical tolerance and the physical verification subset was consistent with cached inference. |
| **G2** | Score Distribution & Dataset Shift | **PASS** | Score distributions showed substantial overlap between pristine and manipulated samples; statistical tests did not establish meaningful distribution separation on the evaluated populations. |
| **G3** | Preprocessing OFAT | **PASS** | OFAT preprocessing changes produced measurable score/AUC variation on the diagnostic subset, while $\theta=0.50$ binary decisions remained unchanged. |
| **G4** | Temporal Sampling | **PASS** | Valid $T=16$ sampling strategies were evaluated; $T \ne 16$ was correctly excluded under the frozen sequence-length contract. |
| **G5** | Attention Attribution & Intervention | **PASS** | Attention statistics were characterized and an inference-only uniform-pooling intervention was evaluated without modifying checkpoint weights. |
| **G6** | Structured Error Diagnosis | **PASS** | Errors were characterized by dataset, threshold, manipulation family, and detector metadata using explicitly non-causal analysis. |
| **G7** | Statistical Uncertainty | **PASS** | Video-level bootstrap uncertainty and paired McNemar analyses were computed under fixed reproducibility settings. |
| **G8** | Publication Validation & Final Audit | **PASS** | Publication artifacts, provenance, numerical consistency, claim audit, regression tests, and checkpoint integrity were verified. |

---

## 1. Research Question
This investigation examines whether a spatiotemporal deepfake detector trained on FaceForensics++ C23 demonstrates robust out-of-domain generalization on Celeb-DF v2, characterizes its score distribution overlap, and diagnoses the sources of false-positive rate inflation under controlled ablations.

## 2. Frozen Baseline Architecture
The detector utilizes a ResNeXt-50 spatial backbone (layer2 freeze) extracting $(B, T, 2048, 4, 4)$ feature maps from $T=16$ sampled frames ($112\times112$ ImageNet normalized face crops), a soft spatial attention module generating spatiotemporal weightings, a 2048-dim LSTM, and a linear binary classification head.

## 3. Dataset Description
Evaluations are conducted on the canonical FaceForensics++ C23 test split (750 physical videos across 75 source identity groups, 150 pristine and 600 manipulated across Deepfakes, Face2Face, FaceSwap, NeuralTextures) and the official Celeb-DF v2 test split (518 videos: 178 Celeb-real, 340 Celeb-synthesis) with zero identity leakage.

## 4. Experimental Protocol
All baseline measurements adhere strictly to the frozen evaluation protocol: uniform sampling ($T=16$), MTCNN largest-face detection (0.20 margin), $112\times112$ resolution, and ImageNet normalization. Diagnostic ablations employ One-Factor-At-A-Time (OFAT) isolation on a fixed 30-video evaluation subset.

## 5. M7 Serving Results
Milestone 7 established interactive serving latency on an NVIDIA RTX 3050 Laptop GPU (Model forward: 31.27 ms mean / 511.7 fps; End-to-End: 2232.51 ms), verified TorchScript and ONNX parity ($T=16$), and packaged a dark-themed monitoring dashboard.

## 6. M8 Benchmark Results
Physical evaluation of the 1,268 benchmark videos established an in-domain FF++ test accuracy of 0.8000 (AUC = 0.4734, EER = 51.33%) and cross-dataset Celeb-DF accuracy of 0.6564 (AUC = 0.5244, EER = 50.29%).

## 7. M9 Diagnostic Results
Score distribution analysis reveals heavy overlap between real and fake predictions (Wasserstein distance = 0.0526, Cliff's delta = -0.0638 on FF++), indicating that high nominal recall (1.0000) at threshold 0.50 is an artifact of right-shifted scores rather than discriminative separation. Statistical hypothesis testing fails to establish meaningful distribution separation between pristine and manipulated classes.

## 8. Generalization Findings
Across manipulation families, the model exhibits comparable ROC-AUC values (Deepfakes: 0.4447, Face2Face: 0.5121, FaceSwap: 0.4710, NeuralTextures: 0.4638), confirming that discrimination limitations are not isolated to a single manipulation type.

## 9. Robustness Findings
Controlled perturbations demonstrate that severe contrast reduction and heavy spatial downsampling moderately shift ROC-AUC (+0.14 to +0.17), while JPEG block compression ($Q=30$) degrades discrimination ($\Delta\text{AUC} = -0.0444$).

## 10. Failure Analysis
At threshold 0.50, all 328 pristine test videos across FF++ and Celeb-DF are classified as fake (False Positive Rate = 1.0000). Adjusting the decision threshold to validation-optimal Youden's $J$ ($\theta = 0.8915$) restores specificity (FPR = 0.4733 on FF++), but lowers recall to 0.4300.

## 11. Attention Analysis
Spatial attention entropy is concentrated (Mean entropy: Real = 0.553 nats, Fake = 0.482 nats). The uniform spatial pooling intervention yielded negligible performance delta ($\Delta\text{AUC} = -0.0133$), indicating that spatial feature pooling rather than localized spatial attention dominates the representations.

## 12. Statistical Uncertainty
Video-level bootstrap resampling ($B=1000$, seed=42) establishes 95% confidence intervals: FF++ Accuracy [0.7720, 0.8280], ROC-AUC [0.4258, 0.5283]; Celeb-DF Accuracy [0.6158, 0.6950], ROC-AUC [0.4703, 0.5764].

## 13. Limitations
The current detector is constrained to a fixed sequence length of $T=16$ frames and depends on MTCNN face detection. Cross-dataset evaluation on DFDC remains pending external competition credential access. Video inference latency operates in interactive clip mode rather than streaming real-time.

## 14. Reproducibility Protocol
All experimental artifacts, manifests, seeds (base seed 42, per-video SHA-256 derivation), and predictions are cached with full provenance. Checkpoint hash matches bitwise across all execution runs.

## 15. Future Work
Future research should investigate multi-scale spatial refinement, larger and more diverse training corpora, variable-length sequence modeling, and contrastive loss formulations to separate overlapping score distributions.
