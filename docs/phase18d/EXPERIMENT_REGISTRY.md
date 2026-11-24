# Phase 18D (`CSP-P18D-E001`) — Pre-Registered Confirmatory Protocol

- **Experiment ID:** `CSP-P18D-E001`
- **Status:** Locked **before** running confirmatory seeds `231–245`.
- **Dev Ablation Seeds:** `9401–9405` ($n = 5$)
- **Locked Confirmatory Seeds:** `231–245` ($n = 15$, completely disjoint from all prior Phases `0–18C` `1–225`).
- **Statistical Tests:** Paired two-sided permutation test ($B = 10{,}000$), paired 95% bootstrap CI ($B = 5{,}000$), Cohen's $d_z$, Holm-Bonferroni correction ($\alpha = 0.05$).

## Pre-Registered Hypotheses (`P18D-H1`–`P18D-H7`)
- **`P18D-H1` (Selective Dynamics Transfer):** In Regime A (`N=60`), `uncertainty_guided` beats both `consequence_only` (P18C) and `full_reconstruction` on `x_h1` (both 95% CIs `> 0`, Holm $p < 0.05$).
- **`P18D-H2` (Boundary Knowledge Continuity):** Beats `consequence_only` on Regime A `boundary_nrmse` AND beats `dynamics_and_boundary` on Regime B `boundary_nrmse` (both 95% CIs `> 0`, Holm $p < 0.05$).
- **`P18D-H3` (Selective Adaptation):** In Regime C (partial mutation), beats both `dynamics_and_boundary` and `full_reconstruction` on `x_h1` (both 95% CIs `> 0`, Holm $p < 0.05$).
- **`P18D-H4` (Structural Change Detection):** In Regime D (full mutation), beats both `unconditional` and `dynamics_and_boundary` on `x_h1` ($p < 0.05$) with `harmful_transfer_rate < 0.15`.
- **`P18D-H5` (Knowledge-Efficient Reconstruction):** Action decision accuracy $\ge 85\%$ and boundary decision accuracy $\ge 75\%$ across all 4 regimes.
- **`P18D-H6` (Cross-Environment Planning Generalization):** In Regime A (`N=60`), beats `fair_mlp` and `full_reconstruction` on episodic planning return (both 95% CIs `> 0`, Holm $p < 0.05$).
- **`P18D-H7` (Uncertainty-Guided Transfer & Integrity):** `uncertainty_guided` achieves mean 4-regime `x_h1 <= selective + 0.005`, `0.0` forgetting, `0` AST violations, and action accuracy $\ge 85\%$.
