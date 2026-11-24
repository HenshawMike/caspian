# Phase 18 (`CSP-P18-E001`) — Stage 1 Architectural Audit & Gap Analysis

- **Baseline Commit (`main`):** `7e0b2e7`
- **Verified Foundation:** Phase 17C (`CSP-P17C-E001`, `Outcome B`, `4/5` hypotheses `MET`, `100.0%` zero-`world_id` retrieval, `0.0000` forgetting, `40/40` shift/offset/rotation invariance, `3,676 B` persistent memory).
- **Regression Baseline:** `.venv/bin/pytest -q tests/test_phase16c.py tests/test_phase17.py tests/test_phase17b.py` → `16 passed`.

## 1. Audited Components & Reuse Matrix

| Component | Module | Status in Phase 18 |
| :--- | :--- | :--- |
| Semantic Firewall & Actions | `environment/observations.py`, `environment/actions.py` | Reused unmodified; every Phase 18 observation audited with `assert_semantic_purity()` |
| Persistent World Memory | `models/memory.py` (`PersistentWorldMemory`) | Reused via **composition** (zero edits to Phase 17C internals) |
| Structural Adapter & Core | `models/transfer_v3.py` (`ModularTransferModelV3`) | Reused directly for canonical subspace encoding $z_t$ and degree-2 $O(K)$-invariant readout $\Phi(z)$ |
| Causal Discovery & Counterfactuals | `models/causal.py` (`CausalStructuralModel`) | Reused causal sparsity & Pearl Abduction-Action-Prediction pattern |
| Pure-NumPy MLP Stack | `models/mlp.py` (`PredictiveMLP`) | Reused as parameter- and data-budget-matched unstructured world-model baseline |
| Phase 4 Rollout Prototype | `evaluation/world_model_rollout.py` | **Excluded from reuse:** no persistent memory, no uncertainty, no causal intervention distinction, and direct `world._entities` privilege |

## 2. Five Mathematical & Architectural Gaps Identified

1. **Gap 1 — Static Outcome Mapping vs. Action-Conditioned State Transitions:** `ModularTransferModelV3` and `PersistentWorldMemory` model static $x \mapsto y$, not dynamical $x_{t+1} = T(x_t, a_t)$.
2. **Gap 2 — Observation vs. Intervention Separation:** `ModularTransferModelV3` has no action conditioning and cannot distinguish passive observational drift ($a = 0$) from active causal interventions ($a \in \{1..4\}$).
3. **Gap 3 — Missing Predictive Uncertainty:** `ModularTransferModelV3._empirical_bayes_ridge` computes $(\sigma^2, \lambda)$ and $\Phi_w^\top \Phi_w + \lambda I$ but discards the Bayesian posterior covariance $\Sigma = \sigma^2 (\Phi_w^\top \Phi_w + \lambda I)^{-1}$.
4. **Gap 4 — No Internal Multi-Step Rollout or Counterfactual Action Tree:** No operator chains $(\hat{z}_{t+1}, \hat{\sigma}_{t+1}^2, \Delta\hat{e}_t)$ internally over horizon $H \ge 2$ without querying the real simulator.
5. **Gap 5 — No Uncertainty-Aware Autonomous Planner or Surprise-Recovery Loop:** No risk-averse receding-horizon MPC objective and no standardized surprise monitor to trigger autonomous engram re-retrieval or online adaptation after mid-episode rule mutations.
