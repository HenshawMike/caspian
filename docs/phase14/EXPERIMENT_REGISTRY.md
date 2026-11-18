# Phase 14 Plan and Pre-Registration (Written Before Any Phase 14 Confirmatory Run)

**Experiment ID:** `CSP-P14-E001`  
**Date:** 2026-11-18  
**Branch:** `research/ontology` (and synced to `research/architecture`)  
**Central Question:** Can Caspian discover how multiple environmental properties interact—including independent main effects, sparse pairwise interactions, multiple simultaneous interactions, and higher-order (3-way) interactions under clean and noisy observations—without receiving interaction labels?

## 1. Five Interaction-Discovery Worlds (`K = 5` Properties, Experiments 14.1–14.6)

Each world has $K = 5$ binary environmental properties $(p_0, p_1, p_2, p_3, p_4) \in \{0, 1\}^5$ encoded as neutral physical channels $(0.2, 0.8)^5$ ($2^5 = 32$ total combinations).
- **Compositional Split (Experiment 14.6):**
  - Training combinations (`22` combinations): all $c \in \{0, 1\}^5$ except the held-out set below (ensuring every pairwise and 3-way interaction among $(p_0, p_1, p_2)$ is observed when $p_3 = 0$).
  - Held-out combinations (`10` combinations): all $c \in \{0, 1\}^5$ where $(p_0 = 1 \land p_1 = 1 \land p_3 = 1)$ or $(p_3 = 1 \land p_4 = 1)$ (testing whether an interaction $(p_0, p_1)$ learned at $p_3 = 0$ transfers compositionally to the unseen context $p_3 = 1$, and whether non-interacting $(p_3, p_4)$ extrapolates additively).
- **Base Main-Effect Structure:**
  $$f_{\text{main}}(p) = 5 p_0 - 4 p_1 + 3 p_2 - 2 p_3 + 1.5 p_4$$

### Worlds Tested
1. **`W14_indep` (Experiment 14.1 — Purely Independent Properties):**
   - Rule: $Y = f_{\text{main}}(p)$ (`value_range = 15.5`).
   - True active interactions: $\mathcal{I}^* = \emptyset$ ($0$ out of $10$ pairs, $0$ out of $10$ triplets).
2. **`W14_sparse_pair` (Experiment 14.2 — Sparse Single Pairwise Interaction):**
   - Rule: $Y = f_{\text{main}}(p) + 6 p_0 p_1$ (`value_range = 21.5`).
   - True active interactions: $\mathcal{I}^* = \{(0, 1)\}$ ($1$ out of $20$ candidate terms).
3. **`W14_multi_pair` (Experiment 14.3 — Multiple Simultaneous Pairwise Interactions):**
   - Rule: $Y = f_{\text{main}}(p) + 6 p_0 p_1 - 5 p_1 p_2$ (`value_range = 21.5`).
   - True active interactions: $\mathcal{I}^* = \{(0, 1), (1, 2)\}$ ($2$ out of $20$ candidate terms).
4. **`W14_higher_order` (Experiment 14.4 — Higher-Order 3-Way Interaction):**
   - Rule: $Y = f_{\text{main}}(p) + 4 p_0 p_1 + 6 p_0 p_1 p_2$ (`value_range = 25.5`).
   - True active interactions: $\mathcal{I}^* = \{(0, 1), (0, 1, 2)\}$ ($1$ pairwise + $1$ three-way interaction).
5. **`W14_noisy_pair` (Experiment 14.5 — Noisy Interaction Discovery):**
   - Rule: $Y = f_{\text{main}}(p) + 6 p_0 p_1 + \varepsilon_y$ with Gaussian observation noise $\varepsilon_x \sim \mathcal{N}(0, 0.03^2)$ on property channels and transition outcome noise $\varepsilon_y \sim \mathcal{N}(0, 0.45^2)$ (`value_range = 21.5`).
   - True active interactions: $\mathcal{I}^* = \{(0, 1)\}$.

## 2. Architectures Compared (Section 35 Specification)

All 6 architectures are implemented in pure NumPy with analytical forward and backward passes:
1. **`sparse_interaction` (`SparseInteractionDiscoveryModel` in `models/interactions.py`):** Univariate per-property neural additive trunk (`FactorisedPropertyModel("sum")`) + sparse $\binom{5}{2} + \binom{5}{3} = 20$-term interaction dictionary learned via Frisch-Waugh-Lovell main-effect projection, Iterative Soft-Thresholding (ISTA), and Gauss-Lasso debiasing (`533` params).
2. **`additive_residual` (`AdditivePlusResidual`):** Phase 12 additive trunk + dense concatenated factorised residual network (`2,218` params at $K=5$).
3. **`additive` (`FactorisedPropertyModel("sum")`):** Strictly additive per-property model (`849` params at $K=5$).
4. **`factorised` (`FactorisedPropertyModel("concat")`):** Per-property encoders concatenated into a shared MLP head (`1,369` params at $K=5$).
5. **`mlp_small` (`PredictiveMLP` `(32, 16)`):** Standard joint MLP (`737` params).
6. **`mlp_large` (`PredictiveMLP` `(64, 32)`):** Larger joint MLP (`2,497` params).

## 3. Seeds, Protocol, and Evaluation Metrics

- **Confirmatory Seeds:** `31, 32, 33, 34, 35, 36, 37, 38, 39, 40` ($n = 10$, strictly disjoint from Phases 11, 12, and 13).
- **Exploratory Smoke Seeds:** `9001, 9002, 9003` (pipeline verification only).
- **Data Budgets:** `lo` and `hi` (`(120, 360)` steps per training combination across the 24 training combinations). Primary evaluation level: `hi`.
- **Model-Agnostic Interaction Discovery Probing (Möbius / Discrete Mixed Partial Derivatives):**
  For every trained model $\hat{f}$, we evaluate its discrete 2nd-order mixed partial derivative $\Delta_{ij}\hat{f}(\mathbf{0})$ across all $\binom{5}{2}=10$ pairs and 3rd-order mixed partial derivative $\Delta_{ijk}\hat{f}(\mathbf{0})$ across all $\binom{5}{3}=10$ triplets. Candidate interaction $\alpha$ is detected ($\alpha \in \hat{\mathcal{I}}$) iff $|\Delta_\alpha \hat{f}(\mathbf{0})| \ge \tau_{\text{detect}} = 1.0$.
- **Metrics:**
  - `heldout_nrmse` and `indist_nrmse` (evaluated against noiseless ground-truth rule outcomes on held-out and in-dist combinations).
  - `interaction_precision`, `interaction_recall`, `interaction_f1`, and `false_interaction_count` ($|\hat{\mathcal{I}} \setminus \mathcal{I}^*|$).
  - `main_effect_mae`: Mean absolute error of single-property baseline main effects $\hat{f}(e_k) - \hat{f}(\mathbf{0})$ against true main effects $(5, -4, 3, -2, 1.5)$, normalized by `value_range`.

## 4. Pre-Registered Hypotheses and Statistical Test $T$

Statistical test $T(a \text{ vs } b, m)$: mean paired improvement $d \ge m$, 95% bootstrap CI lower bound $> 0$ ($10,000$ resamples), and $d \ge m$ in $\ge 8/10$ seeds.

- **P14-H1 (Independent Properties & Zero False Interactions — `W14_indep`):** In `W14_indep` (`hi`), `sparse_interaction` achieves `false_interaction_count == 0` in $\ge 9/10$ seeds and beats `mlp_small` and `additive_residual` on held-out nRMSE: $T(\text{sparse\_interaction vs mlp\_small}, 0.02)$ and $T(\text{sparse\_interaction vs additive\_residual}, 0.01)$.
- **P14-H2 (Sparse Pairwise Interaction Discovery — `W14_sparse_pair`):** In `W14_sparse_pair` (`hi`), `sparse_interaction` achieves `interaction_f1 >= 0.95` in $\ge 9/10$ seeds and beats `additive`, `additive_residual`, and `mlp_small` on held-out nRMSE by margin $m = 0.02$.
- **P14-H3 (Multiple Simultaneous Pairwise Interactions — `W14_multi_pair`):** In `W14_multi_pair` (`hi`), `sparse_interaction` achieves `interaction_f1 >= 0.90` in $\ge 9/10$ seeds and beats `additive`, `additive_residual`, and `mlp_small` on held-out nRMSE by margin $m = 0.02$.
- **P14-H4 (Higher-Order 3-Way Interaction Discovery — `W14_higher_order`):** In `W14_higher_order` (`hi`), `sparse_interaction` recovers the 3-way interaction (`interaction_f1 >= 0.90` in $\ge 9/10$ seeds) and beats `additive`, `additive_residual`, and `mlp_small` on held-out nRMSE by margin $m = 0.02$.
- **P14-H5 (Noisy Interaction Discovery — `W14_noisy_pair`):** Under observation and outcome noise in `W14_noisy_pair` (`hi`), `sparse_interaction` achieves `interaction_f1 >= 0.85` in $\ge 8/10$ seeds and beats `additive_residual` and `mlp_small` on held-out nRMSE by margin $m = 0.015$.
- **P14-H6 (False Interaction Suppression vs. Dense Residual & MLP):** Averaged across all 5 worlds (`hi`), `sparse_interaction` produces strictly fewer false interactions (`false_interaction_count`) than `additive_residual`, `factorised`, `mlp_small`, and `mlp_large` by margin $m = 1.0$ false interaction ($T$ in all 4 comparisons).
- **P14-H7 (Independent Main-Effect Recovery):** In both `W14_sparse_pair` and `W14_higher_order` (`hi`), `sparse_interaction` achieves lower `main_effect_mae` than both `additive` (which is biased by unmodeled interactions) and `mlp_small` by margin $m = 0.02$.

## 5. Exclusion Criteria and Amendments
- **Exclusions:** Cells with non-finite predictions or $< 50$ training interaction events.
- **Amendments:** None.
