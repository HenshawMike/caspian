# Known-Issues Register (Stage 10.0)

| # | Issue | Affects | Status |
|---|---|---|---|
| 1 | Phase 9 report said 3 of 6 pre-registered hypotheses met; correct count is 4 (H2, H3, H6, H8; H1 and H7 failed). | P9 report | Fixed, original archived |
| 2 | Phase 9 "unseen-position" gap compares test worlds that differ in BOTH entity positions and agent start position (origin vs near the held-out block). The gap is therefore confounded with agent start. | P9 H2, unseen-position findings | OPEN: re-measure with matched agent starts in Phase 10 before using the gap |
| 3 | No Phase 0 / Phase 1 raw results logs exist (only reports). | P0-P1 | OPEN: cannot be audited from logs |
| 4 | Only aggregated metrics and per-seed scalar scores are stored; no model checkpoints or raw interaction logs. | All | MITIGATED by determinism (verified); Phase 10 experiments must store checkpoints and raw logs |
| 5 | Phase 7 E005 first run: mix arms trained on contradictory labels (design bug). | P7 E005 | Disclosed in report; corrected rerun (E005-R2) kept alongside |
| 6 | Phase 8 shuffled-pretrain control is itself worse than scratch, so "beats control" is a weak bar; the "identical" world trivially supports transfer. | P8 | Documented; use scratch as the informative comparison |
| 7 | Phase 8 H2 novelty criterion was not met (most novel worlds were not very novel). | P8 | Documented in report |
| 8 | Phase 9 used one fixed, untuned optimizer setting for all model sizes. | P9 | OPEN |
| 9 | `run_all_phase7_experiments` previously omitted E005-R2/E006/E007 (results merged by hand). | P7 reproducibility | Fixed |
| 10 | Earlier `run_all_*` functions could overwrite existing logs. | P6-P9 | Fixed via safe_io |
| 11 | Phase 10 E001 first run failed the reproduction gate (noise3 passive anchor corner mismatch); kept as INVALID, config fixed, rerun passed 10/10. | P10 E001 | Disclosed; both logs kept |
| 12 | Phase 10 E002 negative control (C3) is flawed: training combo 111 carries the A*B interaction, so the "interaction" world is partly identifiable from training data and C3 cannot test what it was meant to. C3's "not met" is therefore uninformative, not evidence of composition. | P10 E002 | OPEN: redesign control so interaction is unidentifiable from T |
| 13 | E002 C1 named one_factor as the pre-chosen intervention arm; coverage_count, which did better, was not registered as primary. Reported descriptively only. | P10 E002 | Disclosed |
| 14 | E002 C2 as registered (|uniform - count| < 0.05 at every budget) fails at budget 1000 (-0.11); it holds at 2000-8000. | P10 E002 | Disclosed |
| 15 | Only a preliminary literature check exists; no novelty claim is made. | P10 Stage 10.5 | OPEN |
| 16 | Not done in Phase 10: probes on compositional models, adaptation after new combinations, ablations of inductive bias, replication in a second world. | P10 | OPEN |
