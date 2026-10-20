# DeepGuard Base Papers & Research Foundation

This directory houses the foundational reference literature selected for the DeepGuard visual deepfake video detection project proposal.

## Primary Benchmark Paper

- **Title**: Deepfake video deception detection using visual attention-based method
- **Authors**: K. Lal, S. Shiwani, and G. C. Gandhi
- **Journal**: *Scientific Reports* (Nature Publishing Group), 15(1):40089, 2025.
- **DOI**: [`10.1038/s41598-025-23920-0`](https://doi.org/10.1038/s41598-025-23920-0)
- **Local PDF File**: `Lal_2025_Deepfake_Visual_Attention_Base_Paper.pdf`
- **Methodological Analysis**: See [`primary_reference.md`](primary_reference.md) for full mathematical formulation, published baseline metrics, and proposed DeepGuard extensions.

## Academic Governance & Proposal Principles

1. **Strict Attribution**: All published metrics reported in the base paper are designated with the `[Ref]` tag and represent reference baseline targets, not metrics achieved by DeepGuard.
2. **Reproduction Scope**: The first phase of DeepGuard proposes to implement an exact structural reproduction of this paper's ResNeXt-50 + Soft Attention + BiLSTM pipeline.
3. **Proposed Extensions**: Subsequent project phases propose to extend this baseline using multi-branch frequency/texture representations, Temporal Transformers, and post-hoc temperature calibration.
