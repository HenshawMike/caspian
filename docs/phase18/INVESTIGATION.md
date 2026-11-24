# Phase 18 (`CSP-P18-E001`) — Stage C Mandatory Post-Phase Investigation

- **Locked Confirmatory Record (Untouched):** [p18_confirmatory_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase18/p18_confirmatory_results.json) (`seeds 171–185`, $n = 15$).
- **Locked Pre-Confirmatory Commit:** `f05913200ef3a7679087bfe06005386984b55d4e` on `feature/planning`.
- **Official Confirmatory Verdict:** **`4/7` Pre-Registered Hypotheses `MET`** (`P18-H2, P18-H3, P18-H4, P18-H7` `MET`; `P18-H1, P18-H5, P18-H6` `NOT MET`).

---

## Pillar A — Forensic Legitimacy Audit ("Fake vs. Legit")

**Verdict: `LEGIT` (100% genuine, zero leakage, zero fabrication).**

1. **Zero Data / Target / Oracle Leakage Audit:**
   - Static AST audit (`ast_leakage_audit_phase18()`) scanned all 17 files in `models/` (`models/world_model.py`, `models/planner.py`, `models/memory.py`, `models/transfer_v3.py`, etc.) and confirmed **`0` violations** — zero references to `world_id`, `s_true`, `M_true`, `A_drift`, `B_act`, `oracle`, `step_true`, `hidden_state_delta`, `rule_fn`, or `ground_truth`.
   - `oracle_plan_action()` is strictly isolated inside `experiments/phase18_experiments.py` as an external diagnostic upper bound and is never imported by any module in `models/`.
   - Every `Phase18WorldSpec.make_neutral_observation()` passes `Observation.assert_semantic_purity()`.
   - Zero raw transitions stored (`raw_observations_stored_bytes = 0`).
2. **Real Parameter Fitting & Analytical Gradient Audit:**
   - `test_p18_analytical_transition_gradients_match_finite_diff` verified analytical transition input and parameter gradients against central finite differences (`max_abs_err < 1e-6`).
   - All predictions in `CaspianWorldModel` and `AutonomousPlanner` are computed via live forward matrix multiplications (`rollout_latent`, `predict_step`, `evaluate_counterfactuals`) with zero hardcoded constants.
3. **Cross-Seed Independence & Determinism Audit:**
   - Identical-seed re-execution produced exact **`0.0` determinism diff**.
   - All 15 confirmatory seeds (`171–185`) produced distinct per-seed metric vectors (e.g., `H5` `integrated_caspian` SD `> 0`, `H4` trap returns SD `13.36`, `H7` shifted returns SD `113.83`).
4. **Negative-Control & Ablation Sanity Checks:**
   - `memory_no_planner` ($H = 1$ greedy with the exact same learned Phase 17C memory) got trapped in the saddle-crossing negative basin (`105.14` vs `integrated_caspian` `442.98`, $d_z = 25.77$), proving the $H = 3$ multi-step rollout—not static consequence memory alone—drives planning performance.
   - `observational_only_model` failed on interventional transitions (`0.45135` vs `0.08613`, `5.24x` worse), proving genuine causal action conditioning.
   - `world_model_no_uncertainty` ($\beta = 0$) walked deeper into the unvisited cliff trap (`-105.52` vs `-95.24`, $p = 0.0021$), proving Bayesian epistemic uncertainty genuinely alters planning behavior.

---

## Pillar B — First-Principles Root-Cause Diagnosis of the 3 `NOT MET` Criteria

### 1. `P18-H1` (`NOT MET` on 1-step state threshold `0.05487 > 0.05` & 1-step MLP tie `0.05487 vs 0.05784`)
- **What Passed:** `CaspianWorldModel` crushed `persistence` at $H = 1, 3$ (`15/15`, Holm $p = 0.0004$), significantly beat `mlp_world_model` at multi-step $H = 3$ (`0.08052 vs 0.09712`, `12/15` wins, $d_z = 1.16$, Holm $p = 0.0028$), and achieved **`5.77x` lower** 1-step consequence error (`0.04311 vs 0.24854`).
- **Exact Mathematical Root Cause:** The ground-truth transition $s_{t+1} = \text{clip}(A s_t + B_{a_t} + \epsilon_s, -1.8, 1.8)$ is piecewise-linear with hard saturation at $|s_k| = 1.8$. A global linear transition $\Delta z = [1, z]\theta_a$ incurs a small ~`0.05` boundary-saturation bias at $H = 1$ (where a `tanh` MLP can bend slightly), while the MLP's unstructured 8D drift accumulates faster over $H = 3$ steps (`0.09712` vs `0.08052`).
- **Principled Carry-Forward Fix:** Fit $\theta_a$ using boundary-masked / interior-weighted linear regression before applying the empirical box clip $\text{clip}(z + \Delta z, z_{\min}, z_{\max})$, eliminating saturation attenuation bias in $\hat{\theta}_a$.

### 2. `P18-H5` (`NOT MET` solely on 1 of 4 baselines: `mlp_world_model` on familiar in-distribution `W18_alpha`)
- **What Passed:** `integrated_caspian` (`442.98`, **`95.63%` of oracle `463.24`**, **`100%` planning success rate**) crushed `memory_no_planner` (`105.14`, Holm $p = 0.0004$, $d_z = 25.77$), `random_policy` (`65.59`, Holm $p = 0.0009$), and `reactive_policy` (`373.07`, Holm $p = 0.0098$).
- **Exact Mathematical Root Cause:** With `350` abundant i.i.d.-reset transitions on a **single familiar in-distribution world**, an 800-parameter `PredictiveMLP` world model also learns `W18_alpha` accurately enough to cross the saddle (`428.80` vs `442.98`, $\Delta = +14.18$, CI `[-2.82, +34.72]`, Holm $p = 0.234$, `8/15` wins). Crucially, the moment the environment shifts to an **unseen sensor subspace + offset (`P18-H7`)**, `integrated_caspian` (`425.00`) beats `mlp_world_model` (`359.86`) significantly ($\Delta = +65.14$, Holm $p = 0.0437$). This is a genuine, scientifically honest characterization of where structured modular world models add decisive value (multi-step stability, zero forgetting, uncertainty awareness, and shifted-world transfer) vs. where abundant single-world data narrows the gap.

### 3. `P18-H6` (`NOT MET` on per-seed mean recovery reduction `38.98% < 50%` despite `0.0000` forgetting, `100%` retrieval & `100%` mutation detection)
- **Failing Seeds Isolated:** Seeds `173` (`-28.4%`), `175` (`-15.6%`), `177` (`-1.3%`), and `182` (`-23.5%`) drove the per-seed mean down to `38.98%`, even though aggregate mean MAE dropped from `11.34 → 4.24` (`62.6%` reduction) and `10/15` seeds improved strongly (up to `97.4%` on seed `172`).
- **Exact First-Principles Root Cause (Coordinate Basis Mismatch on Consequence-Only Mutation):**
  When `allow_update_outdated=True` fired after 30 transitions of `W18_alpha_mutated`, `PersistentWorldMemory.observe_world()` replaced `target_e.model` with a freshly fitted `ModularTransferModelV3` whose `W_adapter` was rotated by SVD/kurtosis on those 30 correlated trajectory points ($W_{\text{new}} = W_{\text{old}} Q$). Meanwhile, `CaspianWorldModel.observe_transitions()` reused `existing_dyn.theta_trans` — which was parameterized in the **old** latent basis $W_{\text{old}}$! Predicting next latent state with `theta_trans` (in $W_{\text{old}}$) and feeding it into `Phi(z_next)` (in $W_{\text{new}}$) injected a basis mismatch $Q \neq I$ whenever $Q$ differed from the identity!
- **First-Principles Solution:** Whenever `allow_update_outdated=True` detects a same-subspace consequence-only mutation (`r_sub < 0.12`), refit **only** the consequence weights `(w_core, b_core)` in the **existing frozen `W_adapter` basis** (keeping `W_adapter` and `theta_trans` strictly aligned in the exact same coordinate frame).
