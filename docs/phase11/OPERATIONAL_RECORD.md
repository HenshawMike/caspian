# Phase 11 operational record
- 2026-11-15 Stage 11.0: repo inspected, 113 tests pass, plan/hypotheses/design/known-issues/registry written and committed (94a9efc) before confirmatory runs.
- Exploratory smoke run (seeds 9001-9003): `logs/phase11/exploratory_rows.json`; pipeline works; no tuning.
- Confirmatory: `logs/phase11/confirmatory_matrix.json` (2 worlds x 2 data levels x 5 archs x 10 seeds, per-seed rows, CKA), `confirmatory_a1.json`, `confirmatory_adaptation.json` (2560 fine-tune curves), `raw/` (200 npz: predictions, train data, weights).
- Analysis: `python -m experiments.analyze_phase11` -> `confirmatory_analysis.json`. Report: `python experiments/generate_p11_report.py` -> `docs/CSP-P11-E001.docx`.
- Registered outcomes: H1 met, H2 not met (inconclusive), H3 met at hi only, H4 met, H5 not met (seed count; floor), H6 mlp_small/additive_mlp met but additive prediction refuted, H7 met, H8 descriptive.
- Deviations from registration: none. Registry text for the W2 split was corrected before the confirmatory run (committed together with the registration).
- Open: literature review; non-additive identified world; optimiser sensitivity; third world.
