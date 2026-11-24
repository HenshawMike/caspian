# Phase 18 (`CSP-P18-E001`) — Pre-Registered Experimental Protocol

- **Experiment ID:** `CSP-P18-E001`
- **Status:** Locked **before** running confirmatory seeds `171–185`.
- **Exploratory Smoke Seeds:** `9201, 9202, 9203` ($n = 3$)
- **Locked Confirmatory Seeds:** `171, 172, ..., 185` ($n = 15$, completely disjoint from all prior phases `1–165`).
- **Statistical Test:** Paired two-sided exact/Monte-Carlo permutation test ($B = 10{,}000$) + 95% paired bootstrap CI ($B = 5{,}000$) + Cohen's $d_z$, with Holm-Bonferroni correction across baselines within each hypothesis family ($\alpha = 0.05$).

## Pre-Registered Hypotheses & Acceptance Criteria (`P18-H1`–`P18-H7`)

| Hypothesis | Operational Metric(s) | Pre-Registered Acceptance Criterion ($n = 15$, seeds `171–185`) |
| :--- | :--- | :--- |
| **P18-H1 (Predictive World Model)** | Open-loop 1-step & 3-step state nRMSE ($x_{t+h}$) and $\Delta e$ nRMSE | 1-step state nRMSE $< 0.05$; beats `persistence` and `mlp_world_model` at $H = 1$ and $H = 3$ (95% CI $> 0$, Holm $p < 0.05$) |
| **P18-H2 (Causal Action Prediction)** | Interventional $\text{do}(a)$ transition nRMSE & causal action-effect F1 | Causal action-effect F1 $\ge 0.90$; beats `observational_only_model` on interventional transitions (95% CI $> 0$, Holm $p < 0.05$) |
| **P18-H3 (Counterfactual Evaluation)** | Spearman $\rho$ across all 5 actions & Top-1 counterfactual action accuracy (zero environment steps) | Mean Spearman $\rho \ge 0.85$ and Top-1 counterfactual accuracy $\ge 85\%$ |
| **P18-H4 (Uncertainty Awareness)** | OOD AUROC, error-uncertainty correlation $r(\hat{\sigma}, |e|)$, and trap-world return ($\beta > 0$ vs $\beta = 0$) | OOD AUROC $\ge 0.90$, $r(\hat{\sigma}, |e|) > 0.40$, and risk-aware ($\beta > 0$) beats risk-neutral ($\beta = 0$) in high-uncertainty trap world (95% CI $> 0$, $p < 0.05$) |
| **P18-H5 (Autonomous Multi-Step Planning)** | Mean cumulative return over $T = 20$ steps ($H = 3$), planning success rate, oracle ratio | Beats `random_policy`, `reactive_policy`, `memory_no_planner`, and `mlp_world_model_planner` (all 95% CIs $> 0$, Holm $p < 0.05$) AND achieves $\ge 75\%$ of `oracle_planner` return |
| **P18-H6 (Persistent Knowledge & Recovery)** | Prior-world forgetting, zero-`world_id` retrieval accuracy, mid-episode mutation recovery | `0.0000` forgetting on stored engrams; $\ge 95\%$ autonomous retrieval; post-mutation surprise recovery reduces prediction error by $\ge 50\%$ within 10 steps |
| **P18-H7 (Unseen Configuration Generalization)** | Planning return & 1-step nRMSE on unseen initial states + novel sensor-shifted worlds | Beats `mlp_world_model_planner` and `reactive_policy` on unseen shifted configurations (95% CI $> 0$, Holm $p < 0.05$) |

## Mandatory Integrity & Audit Criteria
- Zero `world_id` privilege in `models/world_model.py` and `models/planner.py`.
- Zero AST leakage violations across all files in `models/`.
- Exact deterministic reproducibility (`0.0` diff across repeated identical-seed runs).
- Analytical vs. numerical finite-difference gradient check (`max_rel_err < 1e-5`) in `tests/test_phase18.py`.
- Overwrite protection: zero modifications to historical `logs/`, `docs/CSP-P*.docx`, `models/transfer.py`, `models/transfer_v3.py`, or `models/memory.py`.
