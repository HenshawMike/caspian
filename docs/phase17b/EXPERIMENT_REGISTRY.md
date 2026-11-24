# CSP-P17B-E001 — Pre-Registration Protocol

- **Parent Experiment (Preserved Untouched):** `CSP-P17-E001` (`seeds 71–80`, `3/5` hypotheses met).
- **Pre-Implementation Forensic Audit:** [P17B_FORENSIC_AUDIT.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase17b/P17B_FORENSIC_AUDIT.md).
- **Stage 1 Development Seeds:** `[9101, 9102, 9103, 9104, 9105]`.
- **Stage 2 Confirmatory Seeds (Disjoint, Frozen Before Execution):** `SEEDS17B = [131..145]` ($n = 15$, providing ~84% power at $\alpha = 0.05$ for $d_z = 0.80$).

---

## 1. Pre-Registered Hypotheses (`P17B-H1` through `P17B-H5`)

All primary error comparisons use paired differences $\Delta_i = E_{\text{baseline}, i} - E_{\text{occam}, i}$ (positive favors `occam_hierarchy_memory`), bootstrap 95% CIs (`4000` resamples), exact sign-flip permutation $p$-values, and **Holm-Bonferroni** correction across primary comparisons.

| Hypothesis | Primary Endpoint (10-Shot Gamma, Seeds `131–145`) | Pre-Registered Pass Criterion |
| :--- | :--- | :--- |
| **`P17B-H1`** | **Few-Shot Adaptation Improvement vs Original V3:** `forward_transfer_gamma` (`original_v3_memory - occam_hierarchy_memory`) | Mean paired improvement $\ge 0.03$ (minimum practically meaningful improvement), bootstrap 95% CI strictly `> 0`, Holm-adjusted $p < 0.05$. |
| **`P17B-H2`** | **Comparative Forward Transfer vs Baselines:** `forward_transfer_gamma` vs `memoryless_structural`, `naive_sequential_mlp`, `replay_mlp`, `ewc_mlp` | Bootstrap 95% CI strictly `> 0` and Holm-adjusted $p < 0.05$ against `memoryless_structural` AND all three MLP baselines. |
| **`P17B-H3`** | **Retention Preservation:** `ret_alpha_post4`, `ret_beta_post4`, `ret_delta_post4`, `mean_ret_post4`, `mean_forgetting_alpha_beta`, `ret_alpha_post5`, `retrieval_acc_post4` | `mean_forgetting_alpha_beta < 0.01`, `ret_alpha_post5 < 0.02` in `>= 14/15` seeds, `retrieval_acc_post4 >= 0.95`, and `mean_ret_post4` beats all 4 non-memory/MLP baselines (95% CI `> 0`). |
| **`P17B-H4`** | **Phase 16C Offset & Rotation Invariance Preservation:** `W16_obs_shift`, `W16_offset_0.5`, `W16_in_subspace_rot`, `W16_negative_control` (20-shot, seeds `131–140`) | `10/10` shift detection & `10/10` gate ON on `W16_obs_shift` and `W16_offset_0.5`; `0/10` false shift & `10/10` gate OFF on `W16_negative_control`; `>= 7/10` in-subspace rotation detection. |
| **`P17B-H5`** | **Deterministic Reproducibility & Zero Leakage:** Repeated runs, row-order permutations, AST & leakage audit | Exact `0.0` diff on repeated runs; row-order permutation spread `< 1e-12`; `0` AST violations and `0` leakage rows. |
