# Milestone 16 — Step 5 Incremental Modality Value Report

Evaluates the empirical incremental value of candidate forensic modalities over the M13.5 baseline.

| Modality | Primary Feature | Direction | Base AUC | Candidate AUC | Delta AUC | Specificity | Status |
|---|---|---|---|---|---|---|---|
| `motion` | `motion_divergence` | `-1` | 0.9563 | 0.9675 | **+0.0112** | 0.9500 | **EARNED** |
| `temporal` | `forehead_diff` | `-1` | 0.9563 | 0.9688 | **+0.0125** | 0.9500 | **EARNED** |
| `landmarks` | `ear_variance` | `-1` | 0.9563 | 0.9612 | **+0.0050** | 0.9500 | EXCLUDED |
| `spatial` | `noise_autocorr_G` | `+1` | 0.9563 | 0.9563 | **+0.0000** | 0.9500 | EXCLUDED |
| `frequency` | `hf_lf_ratio` | `+1` | 0.9563 | 0.9587 | **+0.0025** | 0.9500 | EXCLUDED |
| `identity` | `min_cosine_similarity` | `-1` | 0.9563 | 0.9537 | **-0.0025** | 0.9500 | EXCLUDED |
| `compression` | `blocking_variance` | `+1` | 0.9563 | 0.9538 | **-0.0025** | 0.9500 | EXCLUDED |
| `lipsync` | `sync_correlation` | `-1` | 0.9563 | 0.9550 | **-0.0013** | 0.9500 | EXCLUDED |

## Combined M16 Evidence-Gated Model

- **Baseline M13.5 AUC**: `0.9563`
- **M16 Combined AUC**: `0.9675` (**+0.0112**)
- **Clean-Real Specificity**: `0.9500` (`+0.0000`)
- **Detection Rate**: `0.8250`
