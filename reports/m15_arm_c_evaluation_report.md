# Milestone 15 Arm C Evaluation Report: Robustness & Domain Generalization

**Evaluation Date**: `2026-09-27T13:12:41.937343+00:00`  
**Governing Standard**: `context/m15_preregistration.md`  
**Arm Designation**: **Arm C (`M15-RobustGen`)**  
**Promotion Verdict**: **`RETAIN_M13_5_FROZEN`**  

---

## 1. Executive Summary & Promotion Verdict

Milestone 15 executed the preregistered evaluation of **Arm C (`M15-RobustGen`)** (cross-attention + multi-scale frequency consistency loss + compression jitter) against the **frozen `v5-family-invariant` control (Arm A)**.

| Promotion Gate | Metric / Target | Candidate Score | Reference Score | Delta / Margin | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Gate 1: Face2Face Gain** | $\Delta \text{AUC}_{\text{F2F}} \ge +0.0500, p < 0.05$ | `0.7053` | `0.8104` | `-0.1052` ($p=1.0000$) | **FAIL** |
| **Gate 2: External Generalization** | $\Delta \text{AUC}_{\text{CelebDF}} \ge -0.0150$ | `0.8723` | `0.7808` | `+0.0916` | **PASS** |
| **Gate 3: Clean FF++ Specificity** | $\text{Spec}_{\text{clean}} \ge 93.00\%$ ($\tau=0.50$) | `81.55\%$` (`43/233` FA) | `93.88\%` | Floor: $93.00\%$ | **FAIL** |
| **Gate 4: Clean Celeb Specificity** | $\text{Spec}_{\text{celeb}} \ge 96.50\%$ ($\tau=0.50$) | `82.29\%$` (`31/175` FA) | `97.75\%` | Floor: $96.50\%$ | **FAIL** |
| **Gate 5: Compression Robustness** | $\text{ROC-AUC}(\text{JPEG } Q=50) \ge 0.6500$ | `0.5900` | `0.6360` | Floor: $0.6500$ | **FAIL** |
| **Gate 6: Runtime & VRAM** | $P50 \le 35.0\text{ms}, \text{VRAM} \le 5.20\text{GB}$ | `21.18\text{ ms}, 0.74\text{ GB}` | `24.95\text{ ms}, 4.85\text{ GB}` | Within Budgets | **PASS** |

**Final Decision**: `RETAIN_M13_5_FROZEN`  
> **Governing Action**: Because one or more promotion gates did not meet the preregistered non-inferiority margins, `v5-family-invariant` is RETAINED as the active production champion. The candidate is preserved as an empirical research checkpoint.

---

## 2. Hypothesis H-M15-3 (Compression & Frequency Regularization) Findings

- **Hypothesis**: Penalizing high-frequency representation variance under simulated DCT compression during training prevents false alarms on compression blocking artifacts.
- **Arm C Real $\to$ Fake Flips under JPEG Q=50**: `0/190` (0.00%)
- **Arm B Real $\to$ Fake Flips under JPEG Q=50**: `0/219` (0.00%)
- **Relative Flip Rate Reduction**: **0.00%** (Preregistered Target: $> 20.0\%$)
- **Hypothesis Status**: **UNSUPPORTED**

---

## 3. Cryptographic Provenance & Audit Trail
- **Arm C Checkpoint**: `G:\MyProjects\DeepFake Video Detection\checkpoints\m15_arm_c_seed42\best.pth`
- **Arm C Checkpoint SHA-256**: `eeef507ab9526a475fecca9d515a33283a2b23ab1663303d70c527bb83aeb24a`
- **v5 Champion Checkpoint**: `G:\MyProjects\DeepFake Video Detection\checkpoints\m13_3_f2f_seed42\best.pth`
- **v5 Champion SHA-256**: `4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7`
- **Derived Temperature $T_{\text{M15-C}}$**: `2.5316`
- **Derived Spec90 Threshold $\tau^*_{\text{M15-C}}$**: `0.0840`
