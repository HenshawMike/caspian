# Phase 18C (`CSP-P18C-E001`) — First-Principles Resolution of `P18B-H1` via Orthogonal Box-Projection Dynamics

- **Permanent DOCX Report:** [docs/CSP-P18C-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P18C-E001.docx)
- **Immutable Historical Records:** [CSP-P18-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P18-E001.docx) (`f059132`, `4/7` `MET`) & [CSP-P18B-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P18B-E001.docx) (`b187add`, `6/7` `MET`).
- **Locked Pre-Confirmatory Commit:** `c603ed8f` (`feature/planning-sync`).
- **Brand-New Unseen Confirmatory Seeds:** `211–225` ($n = 15$) → [p18c_confirmatory_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase18c/p18c_confirmatory_results.json).

## Summary of Results (`Seeds 211–225`, $n = 15$)
- **`P18B-H1` (`MET` — Breakthrough):** `x_h1 = 0.01496` vs MLP `0.06096` (**`4.08x` lower**, Holm $p = 0.0002$); `x_h3 = 0.02734` vs MLP `0.11162` (**`4.08x` lower**, Holm $p = 0.0002$); boundary `0.03636` vs P18 `0.12125` (**`3.33x` lower**); 95% PI `93.2% / 90.3%`.
- **`P18B-H2` (`MET`):** `all_synced = 15/15`, `max_mean_equiv = 5.08e-15`, `max_var_equiv = 9.97e-18`, `ser_diff = 0.0`.
- **`P18B-H3` (`MET`):** Mutation recovery **`88.21%`** (vs P18 `49.00%`), post-rec MAE **`0.940 vs 3.482`** (`p = 0.0003`), post-rec `x_rmse` **`0.0230 vs 0.3374` (`14.68x` lower)**.
- **`P18B-H4` (`MET`):** `0.0000` forgetting, `100.0%` retrieval, `W18_beta` retention `0.0408`, **`390` params / `20,046 B` pickle (`4.34x` smaller than P18B)**, `0 B` raw observations.
- **`P18B-H5` (`MET`):** **`461.62` (`99.55%` of oracle `463.70`)**, beat MLP (`431.07`, $p = 0.0001$), `memory_no_planner` (`106.54`), `reactive` (`366.01`), `random` (`74.91`), and trap safe `+17.00 vs +6.54` ($p = 0.004$).
- **`P18B-H6` (`NOT MET` marginally on shifted `Holm p = 0.0521`):** Beat MLP at `N=80` (`446.16 vs 412.14`, Holm $p = 0.0028$), `N=160` (`459.51 vs 427.52`), `N=350` (`461.62 vs 431.07`, $p = 0.0001$), and directionally on shifted (`430.87 vs 397.39`, Holm $p = 0.0521$).
- **`P18B-H7` (`MET`):** Causal `F1 = 0.92524`, CF $\rho = 0.98467$ / Top-1 `98.67%`, OOD AUROC `1.0000`, `det = 0.0`, `0` AST violations.
