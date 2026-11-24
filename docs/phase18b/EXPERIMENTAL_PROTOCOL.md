# Phase 18B (`CSP-P18B-E001`) — Pre-Registered Experimental Protocol

- **Experiment ID:** `CSP-P18B-E001`
- **Status:** Locked **before** running confirmatory seeds `191–205`.
- **Development / Ablation Seeds (Stage 3–4):** `9301, 9302, 9303, 9304, 9305` ($n = 5$)
- **Locked Confirmatory Seeds (Stage 6):** `191, 192, ..., 205` ($n = 15$, completely disjoint from all prior phases `1–185` and dev seeds `9301–9305`).
- **Statistical Methodology:** Two-sided paired permutation test ($B = 10{,}000$), paired 95% bootstrap CI ($B = 5{,}000$), Cohen's $d_z$, Holm-Bonferroni multiplicity correction ($\alpha = 0.05$).

## Pre-Registered Hypotheses & Acceptance Criteria (`P18B-H1`–`P18B-H7`)

| Hypothesis | Operational Metric(s) | Pre-Registered Acceptance Criterion ($n = 15$, seeds `191–205`) |
| :--- | :--- | :--- |
| **P18B-H1 (Boundary-Aware Prediction)** | 1-step & 3-step open-loop $x$ nRMSE + boundary vs interior breakdown | 1-step $x$ nRMSE $< 0.05$; beats `mlp_world_model` at **both** $H = 1$ and $H = 3$ (95% CI $> 0$, Holm $p < 0.05$); beats Phase 18 baseline on boundary 1-step error (95% CI $> 0$, $p < 0.05$) |
| **P18B-H2 (Representation Synchronization)** | Post-rotation predictive mean equivalence & variance equivalence under $Q \in O(4)$, plus serialization round-trip diff | Max predictive mean equivalence error $< 10^{-11}$; max predictive variance equivalence error $< 10^{-11}$; serialization round-trip diff $= 0.0$; `assert_basis_synchronized()` passes `15/15` |
| **P18B-H3 (Mutation Recovery)** | Mid-episode mutation recovery error reduction (`W18_alpha_mutated`) | Mean per-seed recovery error reduction $\ge 50\%$ AND beats unsynchronized Phase 18 baseline on post-recovery MAE (95% CI $> 0$, $p < 0.05$) |
| **P18B-H4 (Persistent Memory Integrity)** | Prior-world forgetting, zero-`world_id` retrieval accuracy, `W18_beta` retention post-mutation | `0.0000` forgetting on `W18_alpha`, retrieval accuracy $\ge 95\%$, `W18_beta` post-mutation nRMSE $< 0.10$, `0 B` raw observations stored |
| **P18B-H5 (Planning Robustness)** | `W18_alpha` ($N=350$) oracle ratio, beat reactive/memory-only/random, plus trap-world risk-aware vs risk-neutral | Achieves $\ge 90\%$ of oracle; beats `memory_no_planner`, `reactive_policy`, and `random_policy` (all Holm $p < 0.05$); risk-aware ($\beta > 0$) beats risk-neutral ($\beta = 0$) in `W18_trap` ($p < 0.05$) |
| **P18B-H6 (Planning Generalization & Low-Data)** | Low-data (`N=80`) planning & unseen shifted world (`W18_gamma_shifted`) planning vs `mlp_world_model` & `reactive_policy` | Beats `mlp_world_model` on **both** low-data (`N=80`) and shifted (`W18_gamma_shifted`) planning returns (both 95% CIs $> 0$, Holm $p < 0.05$); honestly reports `N=160` and `N=350` regimes |
| **P18B-H7 (Integrated Model Consistency)** | Causal action F1, internal counterfactual $\rho$ & Top-1, OOD AUROC, determinism (`0.0`), AST leakage (`0`) | Causal F1 $\ge 0.90$, CF $\rho \ge 0.85$ & Top-1 $\ge 85\%$, OOD AUROC $\ge 0.90$, `determinism_diff = 0.0`, `0` AST violations |
