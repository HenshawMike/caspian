# Phase 17 Experiment Pre-Registration (`CSP-P17-E001`)

- **Phase:** Phase 17 — Continual Learning and Persistent Environmental Memory
- **Branches:** `feature/memory`, `research/learning-mechanisms` (merge `--no-ff` to `main` after Stage C exit review)
- **Seeds:**
  - Exploratory smoke: `[9001, 9002, 9003]`
  - Locked confirmatory (disjoint from all prior phases): `SEEDS17 = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]`
- **Design Invariants (Carried Forward from Phase 16B/16C):**
  1. **Zero Harness Privilege:** Neither `persistent_memory` nor any baseline receives `world_id` or `obs_shifted` flags at training or revisit time. Context recognition and outdated-memory detection are 100% autonomous from raw observations/interactions.
  2. **Fair Equally-Structured Baseline:** `memoryless_structural` uses the exact same `v3` 66-parameter $O(4)$-invariant basis and Empirical Bayes ridge as `persistent_memory`, isolating the causal effect of persistent memory.
  3. **Non-Overdetermined Forward-Transfer Regime:** Stage 3 (`W17_gamma_shared`) uses $N = 10 < 14 = \dim(\Phi_{\text{full}})$ shots so memoryless structural fitting is strictly underdetermined.
  4. **Semantic Firewall:** Every world realization is verified via GridWorld `Observation.assert_semantic_purity()`.

---

## 1. Sequential 5-Stage Multi-World Stream

| Stage | World | Sensor Matrix | Structural Rule | Budget ($N$) | Continual-Learning Challenge |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **1** | `W17_alpha` | $M_1$ | $6p_0 - 4p_1 + 3p_2 + 5p_0 p_1 - 1$ | `160` | Initial world acquisition |
| **2** | `W17_beta` | $M_2$ | $-5p_0 + 6p_1 - 3p_3 + 4p_1 p_3 - 1$ | `160` | Distinct sensor & rule (tests retention of `W17_alpha`) |
| **3** | `W17_gamma_shared` | $M_3$ | $6p_0 - 4p_1 + 3p_2 + 5p_0 p_1 - 1$ | `10` | Underdetermined forward transfer from `W17_alpha` across shifted sensors |
| **4** | `W17_delta_inverted` | $M_1$ | $-6p_0 + 4p_1 - 3p_2 - 5p_0 p_1 - 1$ | `160` | Same sensors as `W17_alpha`, inverted rule (interference / negative-transfer test) |
| **5** | `W17_beta_mutated` | $M_2$ | $+5p_0 - 6p_1 + 3p_3 - 4p_1 p_3 - 1$ | `40` | Experiment 17.4: rule flip in previously encountered `W17_beta` (outdated memory detection) |

---

## 2. Architectures Evaluated (`6`)
1. `persistent_memory`: `PersistentWorldMemory` (pure NumPy, compact `66`-param `ModularTransferModelV3` engrams, autonomous `8`-probe retrieval, Empirical Bayes forward transfer & novelty/outdated-memory detection).
2. `naive_sequential_mlp`: `PredictiveMLP(8, (32, 16), 1)` (`833` params) trained sequentially with no forgetting mitigation.
3. `replay_mlp`: `PredictiveMLP(8, (32, 16), 1)` (`833` params) with bounded episodic replay (`40` samples per prior world).
4. `ewc_mlp`: `PredictiveMLP(8, (32, 16), 1)` (`833` params) with diagonal empirical Fisher Elastic Weight Consolidation ($\lambda_{\text{EWC}} = 50.0$).
5. `memoryless_structural`: `ModularTransferModelV3` (`66` params) reset and fitted from scratch on each stage (fair equally-structured memoryless baseline).
6. `oracle_multi_world`: Upper-bound diagnostic (`0.0000` nRMSE, explicitly non-deployable).

---

## 3. Pre-Registered Hypotheses & Pass Criteria

| Hypothesis | Criterion | Pass Threshold |
| :--- | :--- | :--- |
| **`P17-H1`** | **Retention & Anti-Forgetting (Stages 1–4):** After learning Stages 1–4, evaluate on all 4 worlds using only an `8`-sample probe (no `world_id`). | Mean 4-world retention nRMSE `< 0.025` in `>= 9/10` seeds; beats `naive_sequential_mlp`, `replay_mlp`, `ewc_mlp`, and `memoryless_structural` with bootstrap 95% CI strictly `> 0`; mean forgetting on `W17_alpha` & `W17_beta` `< 0.01`. |
| **`P17-H2`** | **Autonomous Memory Retrieval (Exp 17.3):** Retrieve the correct world engram from an `8`-sample probe with zero `world_id` access across all 4 worlds $\times$ 10 seeds (`40` probes). | Retrieval accuracy `>= 0.95` (`>= 38/40` exact matches). |
| **`P17-H3`** | **Underdetermined Forward Transfer (Exp 17.2 & 17.5):** On Stage 3 (`W17_gamma_shared`, `10`-shot `< 14` features), transfer stored structure from memory. | Beats fair `memoryless_structural`, `naive_sequential_mlp`, `replay_mlp`, and `ewc_mlp` with bootstrap 95% CI strictly `> 0` and `>= 8/10` wins vs `memoryless_structural`. |
| **`P17-H4`** | **Outdated Memory Detection & Rule-Change Adaptation (Exp 17.4):** On Stage 5 (`W17_beta_mutated`), detect outdated `W17_beta` memory, adapt, and preserve `W17_alpha`. | Outdated memory detected in `10/10` seeds; `W17_beta_mutated` nRMSE `< 0.025` in `>= 9/10` seeds; beats `ewc_mlp` and `replay_mlp` (95% CI `> 0`); `W17_alpha` post-Stage-5 retention `< 0.02` in `>= 9/10` seeds. |
| **`P17-H5`** | **Memory Efficiency:** Compact parametric memory without storing raw transition replay buffers. | Total stored parameters across all 5 stages `<= 330` (`< 40%` of one `833`-param MLP), `0` raw replay samples retained, while beating `replay_mlp` and `ewc_mlp` on retention. |
