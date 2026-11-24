# Phase 18D (`CSP-P18D-E001`) — Stage 5 Forensic Investigation & Phase 19 Readiness Boundaries

- **Locked Pre-Confirmatory Commit:** `522f0c3fd54ce7eb2d22b9cd135e72053018d975` (`feature/selective-transfer`).
- **Locked Confirmatory Results:** [logs/phase18d/p18d_confirmatory_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase18d/p18d_confirmatory_results.json) (`seeds 231–245`, $n = 15$, completely disjoint from all prior phases `1–225` and dev seeds `9401–9405`).
- **Official Confirmatory Outcome:** **`7/7` Pre-Registered Hypotheses `MET`** (`P18D-H1` through `P18D-H7` all `MET`).

## Pillar A — Forensic Legitimacy Audit ("Fake vs. Legit")
**Verdict: `LEGIT`.**
1. **Zero Leakage & Immutable Baselines:** `models/memory.py`, `models/transfer_v3.py`, `models/transfer.py`, and all historical Phase 18 / 18B / 18C logs and DOCX reports remain 100% untouched. Static AST audit across all 17 files in `models/` returned **`0` violations**.
2. **Zero Forgetting & Zero Raw Storage:** Across all 4 regimes × 15 confirmatory seeds (`60` transfer evaluations), `max_forgetting = 0.0000` on source `W18_alpha` and `raw_observations_stored_bytes = 0`.
3. **Genuine Component-Level Selectivity (Not Indiscriminate Transfer):**
   - **Regime A:** `0.0%` unnecessary reconstruction (`75/75` actions transferred).
   - **Regime C:** `93.3%` (`14/15`) mutated action `a=2` detected & reconstructed while `98.3%` (`59/60`) unchanged actions preserved.
   - **Regime D:** `0.0%` (`0/75`) harmful transfers (every incompatible action cleanly rejected & reconstructed).
   - **Overall:** `95.0%` action decision accuracy and `88.3%` boundary decision accuracy.

## Pillar B — Transparent Diagnosis of Limitations & Subtle Findings
1. **Regime C Planning Return Nuance (`fair_mlp = 200.7` vs `uncertainty_guided = 133.3` despite `uncertainty_guided` having `1.89x` lower `x_h1` and `2.22x` lower `x_h3`):**
   - In Regime C, action `a=2` flips from `+0.85` on $s_1$ to `-0.85` on $s_1$, making the positive $(+s_0, +s_1)$ basin unreachable and shifting the optimal policy toward the negative $(-s_0, -s_1)$ basin via the newly reconstructed action `a=2`.
   - Because `uncertainty_guided` reconstructed `a=2` from only ~12 target samples, its Bayesian posterior variance $\Sigma_{2,k}$ on `a=2` is genuinely higher than on the 350-step transferred actions `0, 1, 3, 4`. Under `beta_uncertainty = 0.5`, the cautious planner penalized `a=2` more than the risk-neutral MLP — an honest, mathematically intelligible trade-off between safety/caution and aggressive exploitation of a newly reconstructed action.
2. **Scope Boundaries for Phase 19 (Real-World Transfer & Application Prototype):**
   - **Proven:** Selective component transfer works reliably across orthogonal sensor shifts, additive offsets, tight boundary shifts, single-action partial mutations, and full non-isomorphic structural mutations when dynamics are piecewise-linear with convex orthogonal boundaries and discrete actions.
   - **Not Yet Proven (Phase 19 Scope):** Continuous action spaces, highly non-linear / non-convex state-space geometries, long (>10) sequential environment chains, and partial observability (POMDP state estimation).
