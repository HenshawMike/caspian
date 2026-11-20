# Phase 16 Plan and Pre-Registration (Written Before Any Phase 16 Confirmatory Run)

**Experiment ID:** `CSP-P16-E001`  
**Date:** 2026-11-20  
**Branches:** `feature/multiple-worlds` and `research/ontology`  
**Central Question:** Can Caspian separate reusable structural knowledge from world-specific parameters to achieve zero-shot and few-shot transfer across unfamiliar worlds—without catastrophic forgetting of the source world and without suffering negative transfer when rules invert?

## 1. Source and Target Worlds (Experiments 16.1–16.6)

All worlds have $K = 4$ underlying binary properties $p \in \{-1, +1\}^4$ ($2^4 = 16$ combinations) embedded into an $8$-dimensional neutral sensory space $X = p M_w \in \mathbb{R}^8$ (`Observation.entities[0].properties`, verified by `Observation.assert_semantic_purity()`), where $M_w \in \mathbb{R}^{4 \times 8}$ is a world-specific row-orthonormal mixing matrix ($M_w M_w^\top = I_4$).

- **Source World (`W16_source`):**
  $$Y_{\text{src}} = 6 p_0 - 4 p_1 + 3 p_2 + 5 p_0 p_1 - 1.0 \qquad (\text{value\_range} = 36.0)$$
- **Target Worlds:**
  1. **`W16_obs_shift` (Exp 16.4 — Changing Observation Distribution, Shared Structure):** Identical structural equation $Y = 6 p_0 - 4 p_1 + 3 p_2 + 5 p_0 p_1 - 1.0$, but new orthogonal sensor mixing matrix $M_{\text{tgt}} \neq M_{\text{src}}$ (tests zero-shot vs. few-shot sensor alignment).
  2. **`W16_shared_plus_new` (Exp 16.2 & 16.3 — Shared Core + Unseen World-Specific Rule):** Retains $6 p_0 - 4 p_1 + 3 p_2 + 5 p_0 p_1$ and adds a new world-specific rule $+4 p_3 - 4 p_2 p_3 - 1.0$ (`value_range = 44.0`).
  3. **`W16_rule_shift` (Exp 16.3 — Unseen Rule Shift on Shared Properties):** Same sensor space $M_{\text{src}}$, shifted rule $Y = 6 p_0 + 4 p_1 - 3 p_2 + 5 p_1 p_2 - 1.0$ (`value_range = 36.0`).
  4. **`W16_negative_control` (Exp 16.1 & Negative Transfer Control):** Inverted structural rule $Y = -6 p_0 + 4 p_1 - 3 p_2 - 5 p_0 p_1 - 1.0$ (`value_range = 36.0`). Tests whether models detect incompatibility and avoid negative transfer worse than training from scratch.

## 2. Transfer Models & Baselines (Section 37 Specification)

1. **`modular_transfer` (`ModularTransferModel` in `models/transfer.py`):** Pretrained on `W16_source` ($N_{\text{src}} = 320$), separating sensor adapter $W_{\text{adapter}}$, reusable structural core $(w_{\text{core}}, b_{\text{core}})$, and world-specific residual $(w_{\text{world}}, b_{\text{world}})$ with an automatic **negative-transfer gate** (`core_gate` $\in \{0, 1\}$). Adapted on $N_{\text{shot}} \in \{20, 48\}$ target samples.
2. **`direct_transfer`:** Frozen `ModularTransferModel` evaluated zero-shot on the target world without any target adaptation.
3. **`finetune_mlp`:** `PredictiveMLP` `(32, 16)` pretrained on `W16_source` and fine-tuned on $N_{\text{shot}}$ target samples.
4. **`scratch_mlp`:** `PredictiveMLP` `(32, 16)` trained from scratch on $N_{\text{shot}}$ target samples.
5. **`scratch_structural`:** `ModularTransferModel` trained from scratch on $N_{\text{shot}}$ target samples (controls for architecture vs. transferred knowledge).
6. **`oracle_target`:** Exact analytical target structural equation (upper bound).

## 3. Seeds, Budgets, and Evaluation Metrics

- **Confirmatory Seeds:** `51, 52, 53, 54, 55, 56, 57, 58, 59, 60` ($n = 10$, strictly disjoint from Phases 11–15).
- **Exploratory Smoke Seeds:** `9001, 9002, 9003`.
- **Few-Shot Budgets:** `lo_shot` ($N_{\text{shot}} = 20$) and `hi_shot` ($N_{\text{shot}} = 48$). Primary evaluation level: `lo_shot = 20`.
- **Metrics:**
  - `target_nrmse`: Held-out test nRMSE across all 16 combinations of the target world.
  - `source_retention_nrmse`: Error on `W16_source` *after* target adaptation (measures catastrophic forgetting).
  - `core_gate`: Selected gate value (`1.0` = source core retained, `0.0` = gated off to prevent negative transfer).

## 4. Pre-Registered Hypotheses and Statistical Test $T$

Test $T(a \text{ vs } b, m)$: mean paired improvement $d \ge m$, 95% bootstrap CI lower bound $> 0$ ($10,000$ resamples), and $d \ge m$ in $\ge 8/10$ seeds.

- **P16-H1 (Few-Shot Sensor-Shift Transfer — `W16_obs_shift`, `lo_shot`):** `modular_transfer` beats `direct_transfer`, `scratch_mlp`, and `finetune_mlp` on `target_nrmse` by margin $m = 0.03$.
- **P16-H2 (Shared Core + New Rule Transfer — `W16_shared_plus_new`, `lo_shot`):** `modular_transfer` beats `scratch_mlp`, `finetune_mlp`, and `scratch_structural` on `target_nrmse` by margin $m = 0.03$.
- **P16-H3 (Rule Shift Adaptation — `W16_rule_shift`, `lo_shot`):** `modular_transfer` beats `direct_transfer`, `scratch_mlp`, and `finetune_mlp` on `target_nrmse` by margin $m = 0.03$.
- **P16-H4 (Negative Transfer Avoidance — `W16_negative_control`, `lo_shot`):** `direct_transfer` suffers severe negative transfer (`target_nrmse >= 0.40` in $\ge 9/10$ seeds), whereas `modular_transfer` gates off the incompatible core (`core_gate == 0` in $\ge 9/10$ seeds) and beats `finetune_mlp` on `target_nrmse` by margin $m = 0.05$.
- **P16-H5 (Source Knowledge Retention — `W16_shared_plus_new` & `W16_rule_shift`, `lo_shot`):** After target adaptation, `modular_transfer` maintains `source_retention_nrmse < 0.02` in $\ge 9/10$ seeds and beats `finetune_mlp` on `source_retention_nrmse` by margin $m = 0.05$.
