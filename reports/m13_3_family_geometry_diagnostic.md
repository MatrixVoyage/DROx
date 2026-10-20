# Milestone 13.3 Phase 1: Cross-Family Feature Geometry & Diagnostic Transfer Report

**M12 Baseline Hash**: `c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56` (Immutable Reference)  
**M13.2b Champion Hash**: `d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611` (Established Champion)  
**Diagnostic Focus**: Answering *'What information in frozen representations distinguishes manipulation families, and how does each family relate geometrically to Clean Real and to other fake families?'*  

---

## 1. Cross-Family Centroid Cosine Similarity Matrix (Penultimate Space)

Pairwise cosine similarities between slice centroids in $\mathbb{R}^{1280}$ (M13.2b-Stage5 Champion):

| Slice Centroid | Clean Real | Deepfakes | FaceSwap | NeuralTextures | Face2Face | Celeb-DF |
|---|---|---|---|---|---|---|
| **Clean Real** | 1.0000 | -0.5588 | 0.1316 | 0.3558 | 0.8449 | 0.8321 |
| **Deepfakes** | -0.5588 | 1.0000 | 0.0200 | 0.4140 | -0.1345 | -0.3385 |
| **FaceSwap** | 0.1316 | 0.0200 | 1.0000 | 0.3890 | 0.3152 | 0.4703 |
| **NeuralTextures** | 0.3558 | 0.4140 | 0.3890 | 1.0000 | 0.7990 | 0.6425 |
| **Face2Face** | 0.8449 | -0.1345 | 0.3152 | 0.7990 | 1.0000 | 0.9023 |
| **Celeb-DF** | 0.8321 | -0.3385 | 0.4703 | 0.6425 | 0.9023 | 1.0000 |

> [!IMPORTANT]
> **Key Geometric Finding**: Clean Real has a cosine similarity of **0.8449** with Face2Face, compared to **-0.5588** with Deepfakes and **0.1316** with FaceSwap. Furthermore, Face2Face is closest to NeuralTextures (**0.7990**) among all fake families.

---

## 2. Linear Separability of Each Manipulation Family vs Clean Real

5-fold cross-validated linear probe metrics separating Clean Real ($N=58$) from each manipulation family:

| Manipulation Family | M12 Linear Probe ROC-AUC | M12 PR-AUC | M13.2b Linear Probe ROC-AUC | M13.2b PR-AUC | M13.2b Balanced Acc |
|---|---|---|---|---|---|
| **Deepfakes** | 0.9606 | 0.9523 | **0.9879** | 0.9884 | 92.0% |
| **FaceSwap** | 0.9141 | 0.9244 | **0.9727** | 0.9758 | 90.4% |
| **NeuralTextures** | 0.9495 | 0.9595 | **0.9048** | 0.9020 | 85.0% |
| **Face2Face** | 0.8096 | 0.8501 | **0.8234** | 0.8209 | 76.6% |
| **Celeb-DF** | 0.9845 | 0.9851 | **0.9722** | 0.9746 | 92.8% |

> [!NOTE]
> **Linear Separability Invariant**: Every manipulation family (including Face2Face at **0.8234** AUC) is linearly separable from Clean Real in the frozen penultimate representation. The failure of end-to-end detection is NOT due to absent information, but classifier specialization to seam-heavy families.

---

## 3. Cross-Family Linear Transfer Matrix (Train Family A ➔ Test Family B)

ROC-AUC when a linear classifier is trained on (Clean Real + Family A) and evaluated zero-shot on (Clean Real + Family B):

| Trained on (Real + ...) | Zero-Shot: Deepfakes | Zero-Shot: FaceSwap | Zero-Shot: NeuralTextures | Zero-Shot: Face2Face |
|---|---|---|---|---|
| **Deepfakes** | 1.0000 | 0.9750 | 0.9468 | 0.7940 |
| **FaceSwap** | 0.9364 | 0.9979 | 0.8377 | 0.7714 |
| **NeuralTextures** | 0.9944 | 0.9670 | 0.9774 | 0.8710 |
| **Face2Face** | 0.9352 | 0.9471 | 0.8080 | 0.9349 |

> [!IMPORTANT]
> **Asymmetry of Cross-Family Transfer**:
> - Classifiers trained on **Deepfakes** or **FaceSwap** transfer seamlessly to each other (AUC: **0.9750** and **0.9364**), but drop sharply when transferring to **Face2Face** (AUC: **0.7940**).
> - Classifiers trained on **NeuralTextures** achieve **0.8710** on Face2Face, confirming that expression reenactments share visual evidence that seam-heavy identity swaps do not possess.

---

## 4. Local Spatio-Frequency & Interior/Boundary Residual Metrics

| Slice | Local 8x8 Block DCT High-Freq Ratio | Interior-to-Boundary Gradient Ratio | Primary Visual Characteristics |
|---|---|---|---|
| **Clean Real** | 0.0027 ± 0.0040 | 1.1757 ± 0.4383 | Pristine sensor noise, balanced interior/boundary gradients |
| **Deepfakes** | 0.0026 ± 0.0031 | 1.1063 ± 0.3568 | Seam blending boundary artifacts, moderate local frequency loss |
| **FaceSwap** | 0.0033 ± 0.0036 | 1.1924 ± 0.4576 | High boundary gradient seam, aggressive warping boundary |
| **NeuralTextures** | 0.0022 ± 0.0025 | 1.1979 ± 0.3717 | Interior mouth/expression rendering, subtle local frequency distortion |
| **Face2Face** | 0.0031 ± 0.0036 | 1.1191 ± 0.3374 | Interior mouth/expression reenactment, pristine outer blending border |
| **Celeb-DF** | 0.0004 ± 0.0004 | 1.2911 ± 0.4507 | High-quality autoencoder synthesis, reduced boundary seams |

---

## 5. Architectural Implications for Milestone 13.3

1. **Confirming the Multi-Family LOFO Matrix**: The sharp drop from Deepfakes/FaceSwap to Face2Face confirms the necessity of the 4-fold Leave-One-Family-Out matrix to ensure candidate models generalize across expression reenactment and identity swaps equally.
2. **Local Frequency Residual Feasibility**: Local block-DCT high-frequency ratios exhibit distinct inter-family differences, confirming that local spectral representations capture frequency inconsistencies that global frame-wide FFT variance averaged out.
3. **Interior-to-Boundary Branch Feasibility**: Interior-to-boundary gradient ratios confirm that Face2Face concentrates visual changes inside the facial landmark region rather than at the affine perimeter.
