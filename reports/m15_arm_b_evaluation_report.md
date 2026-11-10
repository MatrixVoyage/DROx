# Milestone 15 Arm B Evaluation Report: Temporal–Spatial Integration

**Evaluation Date**: `2026-09-27T09:26:53.839243+00:00`  
**Governing Standard**: `context/m15_preregistration.md`  
**Arm Designation**: **Arm B (`M15-TS-CrossAttn`)**  
**Promotion Verdict**: **`RETAIN_M13_5_FROZEN`**  

---

## 1. Executive Summary & Promotion Verdict

Milestone 15 executed the preregistered evaluation of **Arm B (`M15-TS-CrossAttn`)** against the **frozen `v5-family-invariant` control (Arm A)**.
Under the preregistered multi-gate promotion protocol, a candidate architecture qualifies to replace `v5-family-invariant` **only if it passes all six promotion gates simultaneously**.

| Promotion Gate | Metric / Target | Candidate Score | Reference Score | Delta / Margin | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Gate 1: Face2Face Gain** | $\Delta \text{AUC}_{\text{F2F}} \ge +0.0500, p < 0.05$ | `0.6980` | `0.8104` | `-0.1124` ($p=1.0000$) | **FAIL** |
| **Gate 2: External Generalization** | $\Delta \text{AUC}_{\text{CelebDF}} \ge -0.0150$ | `0.8275` | `0.7808` | `+0.0467` | **PASS** |
| **Gate 3: Clean FF++ Specificity** | $\text{Spec}_{\text{clean}} \ge 93.00\%$ ($\tau=0.50$) | `93.99\%$` (`14/233` FA) | `93.88\%` | Floor: $93.00\%$ | **PASS** |
| **Gate 4: Clean Celeb Specificity** | $\text{Spec}_{\text{celeb}} \ge 96.50\%$ ($\tau=0.50$) | `94.86\%$` (`9/175` FA) | `97.75\%` | Floor: $96.50\%$ | **FAIL** |
| **Gate 5: Compression Robustness** | $\text{ROC-AUC}(\text{JPEG } Q=50) \ge 0.6500$ | `0.4994` | `0.6360` | Floor: $0.6500$ | **FAIL** |
| **Gate 6: Runtime & VRAM** | $P50 \le 35.0\text{ms}, \text{VRAM} \le 5.20\text{GB}$ | `20.48\text{ ms}, 0.51\text{ GB}` | `24.95\text{ ms}, 4.85\text{ GB}` | Within Budgets | **PASS** |

**Final Decision**: `RETAIN_M13_5_FROZEN`  
> **Governing Action**: Because one or more promotion gates did not meet the preregistered non-inferiority margins, `v5-family-invariant` is RETAINED as the active production champion. The candidate is preserved as an empirical research checkpoint.

---

## 2. Statistical Analysis & Bootstrap Results

### Gate 1: Face2Face Unseen-Manipulation Transfer
- **Candidate ROC-AUC**: `0.6980`
- **Reference ROC-AUC**: `0.8104`
- **Delta AUC**: `-0.1124`
- **1,000-Resample Paired Bootstrap 95% CI**: `[-0.1724, -0.0537]`
- **Empirical Paired p-value**: `1.0000`
- **Detection Rate (Standard $\tau=0.50$)**: `32.76\%$ (19/58) vs `37.93\%$ (22/58)
- **Detection Rate (Spec90 Policy $\tau^*=0.1024$)**: `96.55\%$ (56/58)

### Gate 2: Celeb-DF v2 Cross-Dataset Generalization
- **Candidate External AUC**: `0.8275`
- **Reference External AUC**: `0.7808`
- **Generalization Delta**: `+0.0467` (Margin Floor: $-0.0150$)

---

## 3. Cryptographic Provenance & Audit Trail
- **Arm B Checkpoint**: `checkpoints/m15_arm_b_seed42/best.pth`
- **Arm B Checkpoint SHA-256**: `6371ae340795b2067ce7c5abf4d88b0dc15fb7aba682f11d2986e9cfa680fde9`
- **v5 Champion Checkpoint**: `checkpoints/m13_3_f2f_seed42/best.pth`
- **v5 Champion SHA-256**: `4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7`
- **Derived Temperature $T_{\text{M15}}$**: `2.8719` (Fitted strictly on M15-Val NLL)
- **Derived Spec90 Threshold $\tau^*_{\text{M15}}$**: `0.1024` (Fitted strictly on M15-Val clean reals)
