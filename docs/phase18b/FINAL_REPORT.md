# Phase 18B (`CSP-P18B-E001`) — Final Scientific Report

- **Permanent DOCX Report:** [docs/CSP-P18B-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P18B-E001.docx)
- **Immutable Parent Report:** [docs/CSP-P18-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P18-E001.docx) (`f059132`, `4/7` `MET`)
- **Locked Pre-Confirmatory Commit:** `b187add6` (`feature/planning-sync`)
- **Confirmatory Seeds:** `191–205` ($n = 15$, completely unseen during development/ablations `9301–9305`).
- **Machine-Readable Logs:** [p18b_ablation_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase18b/p18b_ablation_results.json), [p18b_confirmatory_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase18b/p18b_confirmatory_results.json)

## 1. Confirmatory Verdicts (`6/7` Pre-Registered Hypotheses `MET`)

| Hypothesis | Criterion | Observed on Fresh Seeds `191–205` ($n = 15$) | Verdict |
| :--- | :--- | :--- | :---: |
| **`P18B-H1`** | `x_h1 < 0.05`; beat MLP at $H=1,3$ (Holm $p<0.05$); beat P18 boundary | `x_h1 = 0.05047` vs MLP `0.06033` (`14/15`, Holm $p=0.0018$); `x_h3 = 0.06896` vs MLP `0.10497` (`15/15`, Holm $p=0.0002$); boundary `0.06538` vs P18 `0.10776` (`15/15`, $p=0.0001$); 95% PI `94.7% / 93.4%` | **NOT MET** *(`0.05047 > 0.05`)* |
| **`P18B-H2`** | Mean & var equiv `< 1e-11` under $Q \in O(4)$, ser diff `0.0`, `15/15` synced | `all_synced = 15/15`, `max_mean_equiv = 1.78e-15`, `max_var_equiv = 2.17e-17`, `ser_diff = 0.0` | **MET** |
| **`P18B-H3`** | Mutation recovery reduction $\ge 50\%$ & beat P18 post-rec MAE ($p<0.05$) | Recovery reduction **`74.62%`** (vs P18 `46.39%`); post-rec MAE **`1.579 vs 3.544`** (`12/15`, $p=0.0041$); post-rec `x_rmse` **`0.0318 vs 0.3413` (`10.74x` lower)** | **MET** |
| **`P18B-H4`** | `0.0` forgetting, retrieval $\ge 95\%$, Beta ret `< 0.10`, `0 B` raw obs | `forgetting = 0.0000`, `retrieval = 100.0%`, `beta_ret = 0.06443`, `0 B` raw obs (`742` params, `86,950 B` pickle) | **MET** |
| **`P18B-H5`** | $\ge 90\%$ oracle; beat mem-only/reactive/random; trap risk-aware $>$ neutral | **`451.96` (`97.60%` of oracle `463.06`)**; crushed mem-only (`105.16`), reactive (`276.94`), random (`59.31`, all Holm $p=0.0003$), & beat MLP (`434.53`, $p=0.0146$); trap **`+28.69 vs -26.09`** (`14/15`, $p=0.0005$) | **MET** |
| **`P18B-H6`** | Beat MLP at `N=80` & shifted world (Holm $p<0.05$); report `N=160` honestly | `N=80`: **`440.53 vs 402.44`** (`12/15`, Holm $p=0.0102$); `N=160`: `427.35 vs 431.36` ($p=0.939$, tie); shifted: **`427.03 vs 389.54`** (`11/15`, Holm $p=0.0102$) | **MET** |
| **`P18B-H7`** | Causal F1 $\ge 0.90$, CF $\rho \ge 0.85$ / Top-1 $\ge 85\%$, OOD $\ge 0.90$, `0` leaks | `F1 = 0.91692`, `CF rho = 0.96978`, `Top-1 = 93.33%`, `OOD AUROC = 1.0000`, `det = 0.0`, `0` AST violations | **MET** |

## 2. Answer to the Phase 18B Central Research Question
**Yes — proven mathematically and experimentally.** When `PersistentWorldMemory` updates its learned coordinate representation by an orthogonal transformation $Q \in O(4)$, Caspian's `BasisSync` transformation law $M(Q, c)^{-1} \Theta_a Q$ and rotated full-covariance propagation preserve exact predictive mean (`1.78e-15`) and full Bayesian uncertainty (`2.17e-17`) coherence across representation updates, cutting post-mutation dynamics error **`10.74x`** and raising mutation recovery from `46.4%` to **`74.6%`** without sacrificing zero forgetting (`0.0000`) or `100%` autonomous retrieval.
