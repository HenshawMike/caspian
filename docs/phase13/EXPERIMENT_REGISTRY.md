# Phase 13 Plan and Pre-Registration (Written Before Any Phase 13 Confirmatory Run)

**Experiment ID:** `CSP-P13-E001`  
**Date:** 2026-11-17  
**Branch:** `research/ontology` (and synced to `research/learning-mechanisms`)  
**Central Question:** Can Caspian discover useful latent properties of its environment from raw, entangled sensory observations without receiving explicitly separated property channels or property labels, and use those discovered representations compositionally?

## 1. Raw Observation Environments (Experiment 13.1)

In Phases 11–12, models received pre-separated property channels $p = (p_1, \dots, p_K)$. In Phase 13, pre-separated property channels are removed from the agent-facing model input. Instead, each entity exposes a $D = 12$-dimensional raw entangled sensory vector $x \in \mathbb{R}^{12}$ in `Observation.entities[0].properties`:

$$x = z_{\text{scaled}} M, \qquad z_{\text{scaled}} = [p_1, p_2, p_3, \gamma_{\text{nuis}} u_1, \gamma_{\text{nuis}} u_2, \gamma_{\text{nuis}} u_3] \in \mathbb{R}^6$$

where:
- $p_1, p_2, p_3 \in [-1, +1]$ are the 3 task-relevant latent properties (binary levels $\{-1, +1\}$ in `W1_raw` and `W3_raw`; 3/2/3 levels $\{-1, 0, +1\} \times \{-1, +1\} \times \{-1, 0, +1\}$ in `W2_raw`).
- $u_1, u_2, u_3 \in \{-1, +1\}$ are 3 independent environmental nuisance/distractor factors scaled by $\gamma_{\text{nuis}} = 2.0$ (so nuisance variance is $4\times$ property variance in raw sensor space, meaning $80\%$ of raw sensor variance is task-irrelevant nuisance variation).
- $M \in \mathbb{R}^{6 \times 12}$ is a dense row-orthonormal sensor mixing matrix ($M M^\top = I_6$) generated deterministically from a world-level QR decomposition so that every single raw sensor $x_j$ is an entangled linear combination of all underlying properties and nuisance variables.
- All observations pass `Observation.assert_semantic_purity()` (Semantic Firewall enforced).

### Worlds Tested
1. **`W1_raw` (Additive Binary World, 12D Raw Sensors):**
   - Underlying binary properties $(A, B, C) \in \{0, 1\}^3$; outcome $Y = 6A - 4B + 2C$ (`value_range = 12.0`).
   - Training combinations: `000, 100, 010, 001, 101, 011` (6 combinations, each combined with all $2^3 = 8$ nuisance configurations $\to 48$ distinct entity configurations).
   - Held-out combinations: `110, 111` (2 combinations $\times 8$ nuisance configurations $= 16$ held-out entity configurations).
2. **`W2_raw` (Nonlinear Multi-Level Additive World, 12D Raw Sensors):**
   - Underlying properties $(u, v, w)$ with levels $(3, 2, 3)$; outcome $Y = 3u - 4v + w^2$ (`value_range = 14.0`).
   - Held-out combinations: $u = 2$ and $w \ge 1$ (4 combinations $\times 8$ nuisance configurations); remaining 14 combinations are training.
3. **`W3_raw` (Identifiable Non-Additive Interaction World, 12D Raw Sensors):**
   - Underlying binary properties $(A, B, C) \in \{0, 1\}^3$; outcome $Y = 6A - 4B + 2C + 5AB$ (`value_range = 13.0`).
   - Training combinations: `000, 100, 010, 110, 001, 111`; held-out combinations: `101, 011`.

## 2. Architectures and Baselines (Experiment 13.2)

All models are implemented in pure NumPy (`models/discovery.py`) with analytical forward and backward passes:
1. **`latent_additive` (`LatentDiscoveryModel`, mode=`"additive"`):** Projects $x \in \mathbb{R}^{12}$ onto the 6D active sensor subspace and learns an orthogonal unmixing matrix $W \in O(6)$ via scale-invariant fourth-moment kurtosis contrast $\mathcal{L}_{\text{kurt}}(Z) = \frac{1}{4K} \sum_{m=1}^K \frac{\mathbb{E}[Z_{:, m}^4]}{(\mathbb{E}[Z_{:, m}^2] + \varepsilon)^2}$ on unlabeled raw observations $X_{\text{train}}$, followed by strictly additive per-slot univariate MLPs $\hat{y} = \sum_{m=1}^6 f_m(Z_{:, m})$.
2. **`latent_residual` (`LatentDiscoveryModel`, mode=`"residual"`):** Uses the same orthogonal sub-Gaussian unmixing encoder $Z \in \mathbb{R}^6$ followed by `AdditivePlusResidual` (strictly additive per-slot trunk + concatenated factorised interaction residual branch).
3. **`raw_naive_additive`:** `FactorisedPropertyModel("sum")` applied directly to the 12 raw sensor channels $x_1, \dots, x_{12}$ without unmixing.
4. **`pca_statistical` (`PCAPropertyModel`):** Unsupervised top-3 principal components of $X_{\text{train}}$ followed by `FactorisedPropertyModel("sum")`.
5. **`raw_mlp_small`:** Standard `PredictiveMLP` `(32, 16)` on $x \in \mathbb{R}^{12}$.
6. **`raw_mlp_large`:** Higher-capacity `PredictiveMLP` `(64, 32)` on $x \in \mathbb{R}^{12}$.
7. **`untrained_encoder`:** Frozen random `PredictiveMLP` `(32, 16)` hidden representation on $x \in \mathbb{R}^{12}$ with a trained linear readout head.
8. **`random_repr`:** Frozen random linear projection $\mathbb{R}^{12} \to \mathbb{R}^3$ with a trained linear readout head.
9. **`oracle_channels_additive` (Upper-Bound Reference):** Phase 11 `additive` model receiving the 3 clean, manually separated ground-truth property channels $p \in \mathbb{R}^3$.

## 3. Seeds, Training Protocol, and Evaluation Metrics

- **Confirmatory Seeds:** `21, 22, 23, 24, 25, 26, 27, 28, 29, 30` ($n = 10$, strictly disjoint from Phase 11 seeds and Phase 12 seeds `11–20`).
- **Exploratory Smoke Seeds:** `9001, 9002, 9003` (pipeline verification only).
- **Data Budgets:** `lo` and `hi` (`(250, 750)` steps per training combination across `W1_raw`, `W2_raw`, `W3_raw`, yielding ~120 interaction events at `lo` and ~360–900 interaction events at `hi`; `lo` sliced as a deterministic prefix of the `hi` trajectory). Primary evaluation level: `hi`.
- **Optimiser:** Adam (`lr = 0.01`, `weight_decay = 1e-4`, `batch_size = 64`, `epochs = 120`), identical across all architectures.
- **Metrics:**
  - `heldout_nrmse` and `indist_nrmse` ($\text{RMSE} / \text{value\_range}$).
  - `slot_match_r2`: Mean univariate $R^2$ between the 3 ground-truth properties $(p_1, p_2, p_3)$ and the best Hungarian-matched individual latent coordinates on held-out combinations (measures true single-slot property disentanglement rather than multi-slot probe mixing).
  - `probe_cross_combo_r2` vs. `probe_shuffled_r2`: Cross-combination ridge probe $R^2$ from latent state to true outcome vs. shuffled-label control.
  - `inter_seed_cka`: Linear CKA between functional latent representations across all $\binom{10}{2} = 45$ pairs of independent seeds on a common evaluation set.
  - `flip_unseen_mae` & `flip_seen_mae`: Counterfactual single-property intervention prediction error when one underlying property $p_k$ is incremented by 1 level while holding $p_{-k}$ and nuisance $u$ fixed.
  - `intervention_selectivity`: Fraction of total latent change $\|\Delta Z\|_1$ concentrated in a single slot when a single underlying property $p_k$ is intervened on.
  - `distractor_sensitivity`: Mean $|\Delta \hat{y}| / \text{value\_range}$ when nuisance factors $u$ are negated while holding $(p_1, p_2, p_3)$ fixed (true effect $= 0$).
  - `pairwise_acc`: Behavioural ordering accuracy across pairs of combinations touching held-out combinations.

## 4. Pre-Registered Hypotheses and Statistical Test $T$

Statistical test $T(a \text{ vs } b, m)$: For lower-is-better metrics, $d = \text{metric}_b - \text{metric}_a$; for higher-is-better metrics, $d = \text{metric}_a - \text{metric}_b$. $T$ is **MET** iff mean $d \ge m$, 95% bootstrap CI lower bound $> 0$ ($10,000$ resamples), and $d \ge m$ in $\ge 8/10$ seeds.

- **P13-H1 (Compositional Extrapolation from Raw Sensors in Additive Worlds):** `latent_additive` beats `raw_mlp_small` on held-out nRMSE (`hi` data) in both `W1_raw` and `W2_raw`: $T(\text{latent\_additive vs raw\_mlp\_small}, 0.02)$ holds in both worlds.
- **P13-H2 (Superiority over Variance-Based PCA, Untrained, and Random Representations):** In both `W1_raw` and `W2_raw` (`hi` data), `latent_additive` beats `pca_statistical`, `untrained_encoder`, and `random_repr` on held-out nRMSE with margin $m = 0.05$.
- **P13-H3 (Necessity of Sensor Unmixing Before Additive Readout):** In `W2_raw` (`hi` data, nonlinear $w^2$ rule), `latent_additive` beats `raw_naive_additive` on held-out nRMSE ($T(\text{latent\_additive vs raw\_naive\_additive}, 0.02)$) and achieves lower nuisance distractor sensitivity ($T(\text{latent\_additive vs raw\_naive\_additive}, 0.005)$ on `distractor_sensitivity`).
- **P13-H4 (Autonomous Single-Slot Property Recovery):** In both `W1_raw` and `W2_raw` (`hi` data), `latent_additive` achieves Hungarian-matched single-slot property recovery `slot_match_r2` $\ge 0.90$ in $\ge 9/10$ seeds, and beats both `pca_statistical` and `raw_mlp_small` on `slot_match_r2` with margin $m = 0.20$.
- **P13-H5 (Representation Consistency Across Independent Seeds):** In both `W1_raw` and `W2_raw` (`hi` data), mean pairwise inter-seed Linear CKA of `latent_additive` across all 45 seed pairs is $\ge 0.85$ and exceeds that of `untrained_encoder` and `raw_mlp_small` by $\ge 0.15$.
- **P13-H6 (Intervention Sensitivity & Behavioural Utility):** In both `W1_raw` and `W2_raw` (`hi` data), `latent_additive` achieves lower counterfactual unseen single-property intervention error (`flip_unseen_mae` normalized by `value_range`) than `raw_mlp_small` ($T(\text{latent\_additive vs raw\_mlp\_small}, 0.02)$) and higher `intervention_selectivity` than `raw_mlp_small` (margin $0.15$).
- **P13-H7 (Identifiable Non-Additive Interaction World `W3_raw` — Negative Control & Residual Recovery):** In `W3_raw` (`hi` data), `latent_additive` has in-distribution nRMSE $\ge 0.05$ in $\ge 9/10$ seeds (confirming its discovered-slot readout cannot fabricate cross-slot interactions), while `latent_residual` beats both `latent_additive` and `raw_mlp_small` on held-out nRMSE: $T(\text{latent\_residual vs latent\_additive}, 0.02)$ and $T(\text{latent\_residual vs raw\_mlp\_small}, 0.02)$.

## 5. Exclusion Criteria and Amendments
- **Exclusions:** Any cell with non-finite predictions or $< 50$ training interaction events is excluded and reported.
- **Amendments:** None.
