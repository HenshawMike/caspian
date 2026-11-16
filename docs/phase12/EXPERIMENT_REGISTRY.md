# Phase 12 plan and pre-registration (written before any Phase 12 confirmatory run)
Date: 2026-11-16. Scope chosen from the open items listed at the end of Phase 11 (no new directive was supplied): (12.1) a world whose non-additive rule IS identifiable from training data; (12.2) optimiser sensitivity of the Phase 11 architecture comparison (closes known issue 8). Deferred: matched-start Phase 9 re-measure (issue 2), literature review (issue 15), property-discovery test.
Reuse: everything in experiments/phase11_experiments.py (worlds, collectors, metrics, train_model, paired_summary). New: world W3, architecture `additive_residual` (additive model + concat-factorised residual model, output summed; 1434 params at 3 channels), experiments/phase12_experiments.py, tests/test_phase12.py.
Seeds: confirmatory seeds 11-20 (disjoint from Phase 11 seeds, so W1/W2 data here are new draws); exploratory seeds 9001-9003 = smoke test only. Primary data level = hi.

## W3 (12.1)
Binary A,B,C; channels 0.2/0.8; Y = 6A - 4B + 2C + 5AB (always). Train: 000,100,010,110,001,111. Held out: 101, 011. The AB term is identifiable from training (110 and 111 contain it; 000,100,010 do not); held-out combinations contain C together with exactly one of A,B, so correct extrapolation needs additive C effect plus no AB. value_range 13 (range of true Y over all 8 combinations: -4 to 9). Everything else as Phase 11 (Adam lr 0.01, wd 1e-4, 200 epochs, batch 32, MSE, balanced passive collection, steps per combo 667/2667).
Hypotheses:
- P12-H1 (additive bias hurts when the world is not additive): mean in-distribution nRMSE of `additive` on W3 >= 0.05 in >= 9/10 seeds, AND T(mlp_small vs additive, margin 0.02) on held-out nRMSE, hi.
- P12-H2 (a model that can represent both recovers): T(additive_residual vs additive, 0.02) on W3 held-out hi; AND in W1 (additive world), T(additive_residual vs mlp_small, 0.02) held-out hi (it keeps the additive advantage).
- P12-H3 (memorisation reference, descriptive + test): is `mlp_small` better than the memorisation baseline in W3? Report per-seed; support if mean diff >= 0.02 with CI lower > 0 (paired).
## Optimiser sensitivity (12.2)
Grid: lr in {0.003, 0.01, 0.03} x epochs in {200, 400}, worlds W1 and W2, hi level, 5 architectures (mlp_small, mlp_large, factorised, additive_mlp, additive) + additive_residual, seeds 11-20.
- P12-H4: in every one of the 6 configurations and both worlds, mean held-out nRMSE of `additive` is lower than that of `mlp_small` by >= 0.02 (W1) / > 0 (W2; floor effect found in Phase 11) .
- P12-H5 (fair-tuning): for each architecture choose the configuration with the lowest mean IN-DISTRIBUTION nRMSE (independent in-dist test set; never held-out data); then T(additive vs mlp_small, 0.02) on held-out hi in W1 using each arch's selected config.
- P12-H6 (capacity under tuning, revisits Phase 11 H2): same selection, T(mlp_large vs mlp_small, 0.02) in W1.
## Test T and criteria
T(a vs b, m): mean of (nRMSE_b - nRMSE_a) >= m, bootstrap 95% CI lower > 0, >= 8/10 seeds above m. Failure = criterion not met; reversed sign with CI excluding 0 = refuted. No change without an amendment below.
Exclusions: non-finite predictions or < 50 training events. 
Amendments: none.
Exploratory smoke run (seeds 9001-9003, W3 only) was executed AFTER this registration text was written and before any confirmatory run: additive in-dist nRMSE about 0.08, held-out about 0.19; additive_residual and factorised lowest held-out. No criterion was changed after seeing it.
