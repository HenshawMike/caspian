# Phase 15 Operational Record (`CSP-P15-E001`)

- **Pre-registration commit:** `621c1f7` (`research/ontology`), committed before exploratory smoke and confirmatory runs.
- **Exploratory smoke run:** Seeds `9001–9003` (`72` rows, `0` leak rows, all finite).
- **Confirmatory execution:** Disjoint seeds `41–50` across `W15_corr_vs_cause`, `W15_obs_confounder`, `W15_hidden_confounder`, `W15_multi_cause` (`480` trainings, completed in `66.14s`).
  - Raw logs: `logs/phase15/confirmatory_matrix.json` (`480` rows, `0` exclusions, `0` leak rows).
  - Analysis: `.venv/bin/python -m experiments.analyze_phase15` $\to$ `logs/phase15/confirmatory_analysis.json`.
  - Investigation: `logs/phase15/investigation_report.json` and `docs/phase15/INVESTIGATION.md`.
  - Report: `.venv/bin/python experiments/generate_p15_report.py` $\to$ `docs/CSP-P15-E001.docx`.
- **Outcomes:** `P15-H2`, `P15-H3`, `P15-H4` **MET** (`10/10` seeds); `P15-H1`, `P15-H5`, `P15-H6` **partially met / diagnosed from first principles** (`causal_f1 = 1.000` in `40/40` cells; `causal_intervention` beat `mlp_intervention` in `10/10` seeds on ATE nMAE with `65.3%` RER; OLS backdoor adjustment explained `observational_linear` on observed confounding).
- **Next Phase:** Phase 16 — Cross-World Transfer and Structural Generalisation (`feature/multiple-worlds`, `research/ontology`).
