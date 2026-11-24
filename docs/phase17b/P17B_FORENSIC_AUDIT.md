# CSP-P17B-E001 — Section 3 Pre-Implementation Forensic Audit

**Audit Status:** Completed *before* any Phase 17B implementation or confirmatory execution.
**Historical Phase 17 Record (`logs/phase17/confirmatory_matrix.json`, seeds `71–80`):** Preserved completely untouched.

---

## 1. Independent Recalculation of Archived Phase 17 Results

Every published aggregate in `logs/phase17/confirmatory_analysis.json` was independently recalculated from raw seed-level records in `logs/phase17/confirmatory_matrix.json`:

| Metric (`persistent_memory`, Seeds `71–80`) | Independent Recalc | Published Report | Abs Diff | Status |
| :--- | :---: | :---: | :---: | :---: |
| `mean_ret_post4` | `0.052228` | `0.052230` | `2.0e-6` | Exact (5-dec rounding) |
| `ret_alpha_post4` | `0.001908` | `0.001910` | `2.0e-6` | Exact |
| `ret_beta_post4` | `0.001940` | `0.001940` | `0.0e+0` | Exact |
| `ret_gamma_post4` | `0.202922` | `0.202920` | `2.0e-6` | Exact |
| `ret_delta_post4` | `0.002140` | `0.002140` | `0.0e+0` | Exact |
| `forward_transfer_gamma` | `0.176397` | `0.176400` | `3.0e-6` | Exact |
| `stage5_beta_mutated_nrmse` | `0.003975` | `0.003980` | `5.0e-6` | Exact |
| `ret_alpha_post5` | `0.001908` | `0.001910` | `2.0e-6` | Exact |
| **H3 Paired Diff (`scratch - memory`)** | `mean=0.01465, SD=0.08543, 7/10 wins` | `mean=0.01465, SD=0.08543, 7/10 wins` | `0.0e+0` | Exact |

### Why `ret_gamma_post4 (0.20292)` Differs from `forward_transfer_gamma (0.17640)`
- On `8/10` seeds, `ret_gamma_post4 == forward_transfer_gamma` identically.
- On **seeds 74 and 75**, `forward_transfer_gamma` was poor (`0.23023` and `0.20614`), causing the post-Stage-4 `8`-sample probe in `PersistentWorldMemory.retrieve()` to misclassify Gamma's engram (`retrieval_correct_post4 = 3/4`), raising `ret_gamma_post4` to `0.32598` (seed 74) and `0.37564` (seed 75). Thus the two metrics genuinely measure distinct quantities (immediate Stage 3 transfer vs post-Stage-4 probe-retrieved retention) and are computed separately.

---

## 2. Exact Mathematical Diagnostics of Stage 3 (`W17_gamma_shared`, `N = 10` Shots)

| Seed | $N$ | Unique States $G$ | Replicate DOF ($N - G$) | $\text{rank}(X_c)$ | $\kappa(X_{c,4})$ | $\text{rank}(D_\Phi)$ / 15 | $\kappa(D_\Phi)$ | $\hat{\sigma}^2_{\text{rep}}$ (True `0.0625`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `71` | 10 | `7` | 3 | `4` | `2.38` | `7/15` | `3.92` | `0.0112` |
| `72` | 10 | `8` | 2 | `4` | `1.68` | `8/15` | `3.12` | `0.0218` |
| `73` | 10 | `7` | 3 | `4` | `2.65` | `7/15` | `4.27` | `0.0646` |
| `74` | 10 | `7` | 3 | `4` | `2.58` | `7/15` | `3.30` | `0.0985` |
| `75` | 10 | `8` | 2 | `4` | `2.35` | `8/15` | `4.46` | `0.0710` |
| `76` | 10 | `8` | 2 | `4` | `2.18` | `8/15` | `3.76` | `0.0190` |
| **`77`** | 10 | **`5`** | 5 | **`3`** | **`2.83e12` (SINGULAR)** | **`5/15`** | `1.99` | `0.1584` |
| `78` | 10 | `8` | 2 | `4` | `1.81` | `8/15` | `3.76` | `0.0049` |
| `79` | 10 | `8` | 2 | `4` | `2.34` | `8/15` | `3.88` | `0.0368` |
| `80` | 10 | `7` | 3 | `4` | `2.38` | `7/15` | `3.92` | `0.0820` |

---

## 3. Forensic Verdict on the Underdetermination Diagnosis

1. **Confirmed — Severe 10-DOF Underdetermination:** Across all 10 seeds, $G \in [5, 8]$ (mean `7.3`), so Original V3's 10-DOF alignment ($R \in O(4)$ [6 DOF] + $b_{\text{adapter}} \in \mathbb{R}^4$ [4 DOF]) has **more parameters (`10`) than unique states (`5–8`) on `10/10` seeds**.
2. **Critical Nuance — Even 6-DOF Alignment Is Near the Identifiability Limit at `N = 10`:**
   - With `6` rotational DOF and $G \in [5, 8]$, residual degrees of freedom are only $G - 6 \in [-1, 2]$.
   - On **seed 77**, `rank(X_c) = 3 < 4` ($\kappa = 2.83\times 10^{12}$), making the 4th latent axis mathematically **unidentifiable** from the 10-shot sample regardless of alignment complexity.
   - Furthermore, when choosing between two stored cores (`W17_alpha` and `W17_beta`) on $G = 7$ points, only `1` residual DOF separates them, so **multi-core identifiability** is an additional bottleneck beyond 10-DOF overfitting alone.
3. **Replicate Noise Estimator Instability:** With $df_{\text{rep}} = N - G \in [2, 3]$ on `9/10` seeds, $\hat{\sigma}^2_{\text{rep}}$ varies `32x` (`0.0049` to `0.1584` around true `0.0625`). Consequently, any noise-floor test using `4 * sig2_est` must include a variance-relative guard (`0.015 * var_y`) and cannot assume $\hat{\sigma}^2_{\text{rep}}$ is exact.
4. **Leakage & Restart Selection Audit:**
   - `_deduplicate_canonical_with_noise` removes duplicate rows into weighted sufficient statistics `(Xu, yu, wu)` using training data only.
   - Restart selection minimizes weighted training MSE on `(Xu, yu)` deterministically given `self.seed + 71`; zero evaluation data enters alignment, restart selection, or gating.
   - **Important Implementation Note:** Multiple restarts selecting minimum training loss on $G = 7$ points can introduce **optimistic training-loss selection bias** across competing stored cores (`W17_alpha` vs `W17_beta`) when residual DOF $\le 1$. This must be tested via controlled ablations (`20` vs `30` restarts) on fresh confirmatory seeds.

**Conclusion of Forensic Audit:** The underdetermination diagnosis is **mathematically supported** (`10 DOF > G = 5..8` on `10/10` seeds), **but** the audit also predicts that at `N = 10` shots, even 6-DOF Occam-Hierarchy alignment may face a hard identifiability ceiling on low-$G$ / rank-deficient seeds (`G <= 6` or `rank < 4`). Therefore, `CSP-P17B-E001` must test Occam-Hierarchy rigorously on fresh disjoint seeds without assuming it will fully solve `N = 10`.
