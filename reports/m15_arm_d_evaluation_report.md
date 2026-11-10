# Milestone 15 Arm D Evaluation Report: Ablation Analysis (M15-Ablation-NoGate)

**Evaluation Date**: `2026-09-27T14:01:00.930464+00:00`  
**Governing Standard**: `context/m15_preregistration.md`  
**Arm Designation**: **Arm D (`M15-Ablation-NoGate`)**  
**Promotion Verdict**: **`RETAIN_M13_5_FROZEN`**  

---

## 1. Executive Summary & Promotion Verdict

Milestone 15 executed the preregistered evaluation of **Arm D (`M15-Ablation-NoGate`)** (matched-capacity linear projection + addition, zero cross-attention, zero dynamic gating) against the **frozen `v5-family-invariant` control (Arm A)**.

| Promotion Gate | Metric / Target | Candidate Score | Reference Score | Delta / Margin | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Gate 1: Face2Face Gain** | $\Delta \text{AUC}_{\text{F2F}} \ge +0.0500, p < 0.05$ | `0.7284` | `0.8104` | `-0.0821` ($p=1.0000$) | **FAIL** |
| **Gate 2: External Generalization** | $\Delta \text{AUC}_{\text{CelebDF}} \ge -0.0150$ | `0.7902` | `0.7808` | `+0.0094` | **PASS** |
| **Gate 3: Clean FF++ Specificity** | $\text{Spec}_{\text{clean}} \ge 93.00\%$ ($\tau=0.50$) | `94.85\%` (`12/233` FA) | `93.88\%` | Floor: $93.00\%$ | **PASS** |
| **Gate 4: Clean Celeb Specificity** | $\text{Spec}_{\text{celeb}} \ge 96.50\%$ ($\tau=0.50$) | `94.86\%` (`9/175` FA) | `97.75\%` | Floor: $96.50\%$ | **FAIL** |
| **Gate 5: Compression Robustness** | $\text{ROC-AUC}(\text{JPEG } Q=50) \ge 0.6500$ | `0.5625` | `0.6360` | Floor: $0.6500$ | **FAIL** |
| **Gate 6: Runtime & VRAM** | $P50 \le 35.0\text{ms}, \text{VRAM} \le 5.20\text{GB}$ | `25.39\text{ ms}, 0.82\text{ GB}` | `24.95\text{ ms}, 4.85\text{ GB}` | Within Budgets | **PASS** |

**Final Decision**: `RETAIN_M13_5_FROZEN`  
> **Governing Action**: In accordance with Rule 2, 6, 10, and the preregistered multi-gate promotion protocol, `v5-family-invariant` is RETAINED as the active production champion. The candidate is preserved as an empirical research checkpoint.

---

## 2. Four-Arm Milestone 15 Comparative Synthesis

| Arm ID | Architectural Formulation | Face2Face AUC | Celeb-DF v2 AUC | Clean FF++ Spec | Clean Celeb Spec | JPEG Q=50 AUC | P50 Latency | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm A (Frozen M13.5)** | Base + Temporal Transformer | **0.8104** | 0.7808 | 93.88% | **97.75%** | **0.6360** | 24.95 ms | 4.85 GB |
| **Arm B (TS-CrossAttn)** | Cross-Attention + Gating | 0.6980 | 0.8275 | **93.99%** | 94.86% | 0.4994 | 20.48 ms | **0.51 GB** |
| **Arm C (RobustGen)** | Cross-Attn + Frequency Loss | 0.7053 | **0.8723** | 81.55% | 82.29% | 0.5900 | 21.18 ms | 0.74 GB |
| **Arm D (Ablation-NoGate)** | Linear Projection + Addition | **0.7284** | **0.7902** | **94.85%** | **94.86%** | **0.5625** | **25.39 ms** | **0.82 GB** |

### Mechanism Isolation Insights:
1. **Dynamic Gating vs Capacity on Celeb-DF**: $\Delta \text{AUC}(\text{Arm B} - \text{Arm D}) = \mathbf{+0.0373}$
2. **Dynamic Gating vs Capacity on Face2Face**: $\Delta \text{AUC}(\text{Arm B} - \text{Arm D}) = \mathbf{-0.0303}$

---

## 3. Cryptographic Provenance & Audit Trail
- **Arm D Checkpoint**: `checkpoints/m15_arm_d_seed42/best.pth`
- **Arm D Checkpoint SHA-256**: `850d9b2dce4eca9e04312a3c75cec5dd178744436ceb139584d2ccb105595ab1`
- **v5 Champion Checkpoint**: `checkpoints/m13_3_f2f_seed42/best.pth`
- **v5 Champion SHA-256**: `4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7`
- **Derived Temperature $T_{\text{M15-D}}$**: `3.0350`
- **Derived Spec90 Threshold $\tau^*_{\text{M15-D}}$**: `0.0756`
