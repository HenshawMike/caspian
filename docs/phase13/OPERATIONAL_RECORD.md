# Phase 13 Operational Record (`CSP-P13-E001`)

- **Pre-registration commits:** `7a14b67` (initial protocol, models, and unit tests) and `b8779c5` (pre-confirmatory 4-part computation speedup: single-pass `hi` collection with `lo` prefix slicing, shared unmixing cache, batch size 64 / 120 epochs, and 4-core parallel seed execution). Both committed on `research/ontology` before any confirmatory data collection.
- **Exploratory smoke run:** Seeds `9001–9003` (`W1_raw`, `hi` level), executed to verify pipeline finiteness and zero combination leakage (`27` rows, `0` leak rows, all finite).
- **Confirmatory execution:** Disjoint seeds `21–30` across `W1_raw`, `W2_raw`, and `W3_raw` (`lo` and `hi` levels, 9 architectures = 540 model trainings, completed in `169.75s`).
  - Raw logs: `logs/phase13/confirmatory_matrix.json` (`540` rows, `0` exclusions, `0` leak rows) and `logs/phase13/confirmatory_cka.json` (`2,160` inter-seed CKA pairs).
  - Analysis: `.venv/bin/python -m experiments.analyze_phase13` $\to$ `logs/phase13/confirmatory_analysis.json`.
  - Report: `.venv/bin/python experiments/generate_p13_report.py` $\to$ `docs/CSP-P13-E001.docx`.
- **Outcomes:**
  - **Met:** `P13-H1` (compositional extrapolation from raw observations in `W1_raw` and `W2_raw`), `P13-H2` (superiority over `pca_statistical`, `untrained_encoder`, and `random_repr` in all 6 cells, `10/10` seeds), `P13-H3` (necessity of sensor unmixing vs. `raw_naive_additive` on held-out nRMSE and distractor sensitivity in `W2_raw`), `P13-H6` (counterfactual single-property intervention flip accuracy and intervention selectivity in `W1_raw` and `W2_raw`).
  - **Partially met (strictly reported as `NOT MET`):**
    - `P13-H4`: All 4 paired tests vs. `pca_statistical` and `raw_mlp_small` passed (`slot_match_r2` mean `1.000` in `W1_raw` and `0.948` in `W2_raw`), and `10/10` seeds met `>= 0.90` in `W1_raw`, but `8/10` seeds met `>= 0.90` in `W2_raw` (1 seed short of `9/10` due to 2 local-minimum Stiefel restarts).
    - `P13-H5`: Inter-seed Linear CKA was `1.000` in `W1_raw` (`MET`), and `0.745` in `W2_raw` (exceeding `untrained_encoder` `0.438`, but below `0.85` due to the same 2 seeds).
    - `P13-H7`: In `W3_raw`, `latent_residual` beat both `latent_additive` (`mean_diff = 0.1172`, `8/10` seeds: `MET`) and `raw_mlp_small` (`mean_diff = 0.0478`, `9/10` seeds: `MET`), while `latent_additive` had in-distribution nRMSE `>= 0.05` in `8/10` seeds (1 seed short of `9/10`).
- **Deviations / Amendments:** None post-confirmatory.
- **Next Phase:** Phase 14 — Interaction Discovery and Structured Relationships (`research/ontology` and `research/architecture`).
