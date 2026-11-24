# Phase 17C (`CSP-P17C-E001`) — First-Principles Root-Cause Resolution of Phase 17B (`Seeds 151–165`)

- **Permanent DOCX Report:** [docs/CSP-P17C-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P17C-E001.docx)
- **Historical Records Preserved Intact:** [docs/CSP-P17-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P17-E001.docx) (`seeds 71–80`) and [docs/CSP-P17B-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P17B-E001.docx) (`seeds 131–145`, `Outcome C`).
- **Fresh Unseen Independent Confirmation Seeds:** `151–165` ($n = 15$, never touched during diagnosis or development).
- **Machine-Readable Results:** [logs/phase17c/p17c_kurtosis_occam_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase17c/p17c_kurtosis_occam_results.json)
- **Official Outcome Classification:** **`Outcome B — Partial Improvement` (`4/5` hypotheses `MET`; `P17B-H1` 95% CI strictly `> 0` with `delta = +0.0598`, `exact_perm_p = 0.061`)**.

---

## 1. Three First-Principles Mathematical Root Causes Diagnosed & Fixed in `models/transfer_v3.py`

1. **Root Cause 1 — Rotated Source-Mean Injection (`P17B-H4`: `4/10` $\rightarrow$ `10/10`):**
   - Subtracting the unrotated finite-sample source mean `_source_x_mean` (`||xm_src|| ~ 0.15`) *before* rotating by $R$ in the in-subspace pure rotation test injected a $2\|\bar{x}_{\text{src}}\|$ offset error, inflating `mse_in` by `5x–10x`.
   - **Fix:** Use `xm_fixed = 0` for the in-subspace rotation test and zero-offset canonical centering in `fit_source` when `||mean(Xu)(I - V V^T)||_2 < 0.10`.
   - **Result on Fresh Seeds `151–160`:** `W16_in_subspace_rot` detection jumped from **`4/10` $\rightarrow$ `10/10` (`mean = 0.00656`)**, while `W16_obs_shift` (`10/10`), `W16_offset_0.5` (`10/10`, `mean = 0.01070`), and `W16_negative_control` (`0/10` false shift, `10/10` gate OFF) all passed **`40/40`** (`P17B-H4` $\rightarrow$ **`MET`**).

2. **Root Cause 2 — Raw Sample-Mean Norm Triggering False 10-DOF Expansion (Seed 131 `0.3386`):**
   - For $G = 7$ zero-mean draws from $\{-1,+1\}^4$, $\mathbb{E}[\|\bar{x}_u\|] \approx 0.75 > 0.35$, falsely tripping the old `||mean(Xu)|| > 0.35` offset guard.
   - **Fix:** Replace with the exact **orthogonal-subspace mean residual** $\|\bar{x}_u (I - V_{\text{tgt}} V_{\text{tgt}}^\top)\|_2 > 0.10$ (`0.0000` for any zero-offset finite-sample fluctuation; `~1.0` for genuine additive offsets $c = 0.5$).

3. **Root Cause 3 — Continuous Off-Manifold Overfitting in 6-DOF $O(4)$ Alignment (Phase 13 + Phase 17 Unification):**
   - On `G = 7–8` points, unregularized continuous $O(4)$ optimization achieved `loss_found = 0.0075 < loss_true = 0.1234` by warping $Z$ off the source latent hypercube manifold (`err_found = 0.3175`).
   - **Fix:** Align `W_adapter` in `fit_source` via Phase 13 unsupervised sub-Gaussian kurtosis unmixing ($\min_Q \mathbb{E}[(U Q)_k^4]$) and apply the unsupervised kurtosis regularizer $\lambda_{\text{kurt}} \mathbb{E}[(U_t R)_k^4]$ ($\lambda_{\text{kurt}} = 3.0$, automatically active only when `_source_kurtosis < 2.0`) on the `G <= 10` 6-DOF branch.

---

## 2. Independent Confirmation Results on Fresh Unseen Seeds `151–165` ($n = 15$)

| Hypothesis | Criterion | Observed on Fresh Seeds `151–165` | Verdict |
| :--- | :--- | :--- | :---: |
| **`P17B-H1`** | 10-shot Gamma vs Original V3 (`delta >= 0.03`, CI `> 0`, Holm `p < 0.05`) | **`0.06306` vs `0.12285`** (`delta = +0.05978`, 95% CI **`[+0.0069, +0.1166]` > 0**, `d_z = 0.522`, `Holm p = 0.061`) | **NOT MET** *(p = 0.061)* |
| **`P17B-H2`** | 10-shot Gamma vs `memoryless_structural` & all 3 MLPs (CI `> 0`, Holm `p < 0.05`) | Beat `memoryless_structural` (`0.14228`, `12/15` wins, **`Holm p = 0.0155`**, `d_z = 0.85`), `naive_mlp` (`p = 0.0150`), `ewc_mlp` (`p = 0.0070`), `replay_mlp` (`p = 0.0005`, `15/15`) | **MET** |
| **`P17B-H3`** | Retention preservation (`forgetting < 0.01`, `alpha5 < 0.02`, `retr >= 0.95`, beat all 4 baselines) | `mean_ret_post4 = 0.01717`, `forgetting = 0.0000`, `alpha5 < 0.02 = 15/15`, **`retrieval = 60/60 (100.0%)`** | **MET** |
| **`P17B-H4`** | Phase 16C offset & rotation invariance (`obs 10/10`, `offset 10/10`, `neg 0/10`, `in_rot >= 7/10`) | `obs = 10/10`, `offset_0.5 = 10/10 (0.0107)`, `in_subspace_rot = 10/10 (0.0066)`, `neg = 0/10 false & 10/10 gate OFF` | **MET** |
| **`P17B-H5`** | Determinism (`0.0`), row-order permutation (`< 1e-12`), `0` AST/leakage violations | `det = 0.0`, `perm = 0.0`, `ast_passed = True`, `0` leaks | **MET** |
