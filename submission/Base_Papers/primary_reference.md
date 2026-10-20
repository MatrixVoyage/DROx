# Foundational Reference Paper Analysis: Lal et al. (2025)

## Bibliographic Metadata

- **Title**: Deepfake video deception detection using visual attention-based method
- **Authors**: K. Lal, S. Shiwani, G. C. Gandhi
- **Journal**: *Scientific Reports* (Nature Publishing Group)
- **Volume & Article**: Vol. 15, Article no. 40089 (2025)
- **DOI**: [`10.1038/s41598-025-23920-0`](https://doi.org/10.1038/s41598-025-23920-0)
- **Local PDF Copy**: `submission/Base_Papers/Lal_2025_Deepfake_Visual_Attention_Base_Paper.pdf`
- **Role in Project**: Primary foundational research baseline for DeepGuard

---

## 1. Paper Overview & Proposed Reproduction Rationale

The selected reference paper addresses the problem of detecting localized deepfake manipulations in compressed video sequences. The authors identify that conventional spatial CNNs relying on global average pooling dilute fine-grained blending and warping artifacts across facial regions. To solve this, they introduce a visual soft-attention mechanism applied over penultimate convolutional feature maps prior to bidirectional recurrent modeling.

### Mathematical Formulation of the Base Method
1. **Frame Extraction**: Consecutive 16-frame sequence extraction.
2. **Face Detection**: MTCNN face detection localizing facial ROI; cropped and resized to $112 \times 112 \times 3$ pixels.
3. **Spatial Feature Extraction**: ImageNet-pretrained ResNeXt-50 ($32 \times 4\text{d}$) producing feature tensor $F \in \mathbb{R}^{B \cdot T \times 2048 \times 4 \times 4}$.
4. **Soft Visual Attention**:
   $$\alpha = \text{Softmax}\left(\text{Conv2D}_{1 \times 1}(F)\right) \in \mathbb{R}^{B \cdot T \times 1 \times 4 \times 4}$$
   $$\tilde{F} = \sum_{h=1}^4 \sum_{w=1}^4 \alpha_{h,w} \odot F_{h,w} \in \mathbb{R}^{B \cdot T \times 2048}$$
5. **Bidirectional Recurrent Modeling**: 2-layer BiLSTM ($H=1024$ per direction $\to 2048$ output dimensions) over sequence length $T=16$.
6. **Temporal Pooling & Classification**: Temporal mean pooling + post-pool dropout ($p=0.50$) + linear classification head (2048 $\to$ 1 binary logit).

---

## 2. Published Benchmark Results (Reference Baselines)

> [!IMPORTANT]
> The performance metrics below represent published results achieved by the original paper authors under their high-performance compute environment. These numbers serve as empirical targets and reference benchmarks for the DeepGuard project, NOT as metrics achieved by DeepGuard.

| Dataset / Benchmark | Published Paper Result [Ref] | Metric Definition & Protocol Notes |
|---|:---:|---|
| **FaceForensics++ C23 (HQ)** | **AUC = 0.962 ± 0.008** | In-domain training across 720 train / 140 val / 140 test subjects; reported Accuracy = 91.50%. |
| **Celeb-DF v2** | **AUC = 0.950 ± 0.009** | Zero-shot cross-dataset evaluation on 518 official test videos. |
| **DFDC (Challenge)** | **AUC = 0.920 ± 0.010** | Cross-dataset evaluation on Deepfake Detection Challenge preview subset. |
| **Attention Ablation Impact** | **Degradation of 3.2% to 8.5%** | Removing spatial soft attention dropped in-domain AUC from 0.962 to 0.928 - 0.877. |

---

## 3. Identified Research Gaps & Proposed DeepGuard Extensions

The DeepGuard team conducted an in-depth critical analysis of the base paper and identified key scientific and practical limitations that motivate our proposed extensions:

1. **High Compute & Memory Barrier**:
   - *Limitation*: The authors trained their architecture using large batch sizes (batch 32) across high-end GPU clusters for $\sim 101$ epochs.
   - *DeepGuard Proposal*: DeepGuard will re-engineer the training pipeline using mixed precision (FP16/AMP), micro-batching (batch size 4), and gradient accumulation to ensure reproducibility within 6.0 GB VRAM on an NVIDIA RTX 3050 Laptop GPU.

2. **The 'Within-Manifold' Face2Face Vulnerability**:
   - *Limitation*: The base paper evaluates performance across aggregate datasets without isolating per-family performance drops. Facial reenactment methods like Face2Face deform expressions without creating boundary blending seams, evading spatial boundary detectors.
   - *DeepGuard Proposal*: DeepGuard proposes a multi-branch feature decomposition integrating mid-spatial Stage 5 features with high-frequency residual filters and texture anomaly streams to detect within-manifold expression manipulations.

3. **Recurrent Bottleneck in Temporal Modeling**:
   - *Limitation*: The paper's BiLSTM enforces sequential inter-frame dependencies, which restricts parallelization and can suffer vanishing gradients on longer sequences.
   - *DeepGuard Proposal*: DeepGuard proposes to investigate replacing the BiLSTM with a multi-head Temporal Transformer encoder with self-attention across the 16-frame sequence.

4. **Uncalibrated Model Confidence**:
   - *Limitation*: The base paper relies on standard sigmoid classification thresholds ($\tau = 0.50$), which frequently suffer severe logit drift and high false positive rates under lossy compression.
   - *DeepGuard Proposal*: DeepGuard will integrate post-hoc temperature scaling calibration and dual-threshold operating policies to guarantee $\ge 90\%$ specificity on clean authentic media.
