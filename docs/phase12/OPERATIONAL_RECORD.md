# Phase 12 operational record
- Registration commit 294303b (before any confirmatory run). Exploratory smoke: seeds 9001-9003, W3 only, run after the registration text, before confirmatory runs.
- Confirmatory: logs/phase12/confirmatory_w3.json (10 seeds x 2 data levels x 6 archs), confirmatory_sensitivity.json (720 trainings). Analysis: `python -m experiments.analyze_phase12` -> confirmatory_analysis.json. Report: `python experiments/generate_p12_report.py` -> docs/CSP-P12-E001.docx.
- Outcomes: P12-H1 not met; H2a, H2b, H3, H4, H5 met; H6 not met (7/10 seeds). Deviations: none. Amendments: none.
- Open: sparse-interaction world, property discovery, Phase 9 matched-start re-measure, literature review.
