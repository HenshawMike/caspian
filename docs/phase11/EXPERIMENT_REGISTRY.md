# Experiment registry and pre-registration (written before any confirmatory run)

Registration date: 2026-11-15. Exploratory data seen: seeds 9001-9003 smoke run only (additive arch near 0 error in W1/W2 as expected from its structure; MLPs 0.02-0.13 nRMSE; everything finite; zero leakage). No criterion below was tuned on them beyond choosing thresholds in view of the nRMSE scale; this is disclosed.

## Worlds
- W1: properties A,B,C binary, 3 channels (0.2/0.8). Y = 6A-4B+2C (+5AB in interaction variant). Train 000,100,010,001,101,011; held out 110,111. value_range 12.
- W2 (independent construction): channel order (w, distractor z, u, v); u and w have 3 levels (0.10/0.50/0.90 and 0.15/0.55/0.95), v has 2 (0.25/0.75), z is a 2-level nuisance that never matters. Y = 3u - 4v + w^2 (+3 if u=2 and w>=1 in the interaction variant). Combinations (u,v,w): 18. Held out: u=2 and w>=1 (4 combinations: all v, w in {1,2}); the other 14 are training. value_range 14. Quadratic effect of w is within-property, so the world stays additive across properties.
- Both: interaction effect is zero on every training combination; training data identical across variants (tested).
Outcome observed = rule + constant step cost -1.0 (verified against simulator).

## Design
Architectures: mlp_small, mlp_large, factorised, additive, additive_mlp; references: memorisation, additive-dummy regression, analytical oracle; A1 = Phase 10 pipeline on W1.
Data levels: lo = 667 (W1)/643 (W2) env steps per training combination; hi = 2667/2571. Balanced passive collection over training combinations (Phase 10: balance explains gains). Seeds: 42,101,7,2026,555,1,2,3,4,5 (n=10), world seed = model seed. Epochs 200, lr 0.01.
Dependent: held-out nRMSE (RMSE / value_range) on interaction events, per world variant; in-distribution nRMSE; per-combination MAE/bias; pairwise ordering accuracy; counterfactual flip error (seen vs unseen pairs); latent marginal eta2; cross-combination ridge probe R2 with shuffled-label control; distractor sensitivity (W2); CKA between architectures.
Statistics: paired over seeds, difference d = nRMSE(baseline) - nRMSE(candidate); 95% bootstrap CI (10000 resamples), d_z, seed count.
Primary data level = hi. A result is called robust only if it also meets its criterion at lo.

## Success criteria (a hypothesis is "supported" only if all conditions hold)
Test T(a vs b, margin m) = mean d >= m AND bootstrap CI lower bound > 0 AND d >= m in >= 8/10 seeds.
- H1: T(additive vs mlp_small, 0.02) on W1 held-out additive, hi.
- H2: NOT supported-as-capacity-explanation if T(mlp_large vs mlp_small, 0.02) fails. (If it passes, capacity contributes and H1's attribution to structure is weakened.)
- H3: T(factorised vs mlp_small, 0.02), W1 additive hi.
- H4: additive arch held-out nRMSE in W1 interaction >= 0.25 in >= 9/10 seeds, and 0 leak rows everywhere.
- H5: same as H1 on W2 hi (nRMSE additive variant).
- H6: for mlp_small and additive_mlp: mean over seeds of [unrevealed nRMSE before reveal - after 20 observations, 30 epochs, replay mode] >= 0.05 with CI lower > 0 in W1 interaction; for additive: improvement < 0.05 (predicted failure).
- H7: T(additive vs mlp_small, 0.02) on unseen flip MAE (W1 additive, hi); and on distractor sensitivity (W2, hi) with margin 0.005.
## Failure criteria
A hypothesis whose criterion is not met is reported as unsupported; a reversed sign with CI excluding 0 is reported as refuted. No criterion is changed post hoc; any change is a formal amendment below.
## Exclusion criteria
Cells with non-finite predictions or fewer than 50 training interaction events are excluded and listed; none expected.
## Adaptation protocol (Stage 11.3)
Base model trained on train combos (hi data). Two reveal orders (seeded permutation of held-out combos). Reveal n in {1,5,20,80} observations of one held-out combination (independent data, not evaluation rows); modes replay (+100 stored training rows) and no_replay; lr 0.003; curve at epochs 0,1,3,10,30,100. Measures: revealed-combo error, unrevealed held-out error, training-combo error (forgetting). Memorising vs rule: improvement on the unrevealed combo counts as rule-like; improvement only on the revealed combo counts as memorisation.
## Amendments
(none)
