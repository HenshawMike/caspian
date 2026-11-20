# Phase 16C (`CSP-P16C-E001`) — First-Principles Resolution of Phase 16B Issues (`ModularTransferModelV3`)

- **Frozen Historical `v2` Record (Preserved Intact):** [models/transfer.py](file:///home/henshawmike/Projects/Project_Caspian/models/transfer.py) (`SHA-256: c6ec7df4...`), [CSP-P16B-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P16B-E001.docx)
- **Upgraded Pure-NumPy 66-Parameter Candidate:** [models/transfer_v3.py](file:///home/henshawmike/Projects/Project_Caspian/models/transfer_v3.py) (`ModularTransferModelV3`)
- **Verification Script & Logs:** [phase16c_v3_verification.py](file:///home/henshawmike/Projects/Project_Caspian/experiments/phase16c_v3_verification.py), [p16c_v3_verification.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase16c/p16c_v3_verification.json)
- **Unit & Gradient Verification:** [tests/test_phase16c.py](file:///home/henshawmike/Projects/Project_Caspian/tests/test_phase16c.py) (`15 passed`, `0` AST violations, `66` trainable params)

---

## 1. Four First-Principles Root Causes & Fixes (`v2` $\rightarrow$ `v3`)

| Phase 16B Failure / Sensitivity | Exact Mathematical Root Cause | First-Principles Fix in `ModularTransferModelV3` |
| :--- | :--- | :--- |
| **1. Gate C & Gate D `FAIL` (Seed 107 `0.2013` spike on `W16_negative_control`)** | When $G = 11 < p = 15$, high-leverage corners had $H_{ii} = 0.9994$, multiplying pointwise LOOCV squared residuals by $(1 - H_{ii})^{-2} \approx 2.7\times 10^6$ and forcing LOOCV to pick $\lambda = 15$ (`0.2013`) instead of the true optimum $\lambda = 0.01$ (`0.0056`). | **Empirical Bayes (Type-II ML) Ridge with Replicate-Noise Variance:** Estimate unbiased noise variance $\hat{\sigma}^2_{\text{rep}}$ from within-state replicate draws ($N - G$ degrees of freedom) and minimize exact closed-form Gaussian marginal likelihood $\frac{1}{2}\sum_k [\text{proj}_k / (\hat{\sigma}^2(1 + \mu_k/\lambda)) + \log(\hat{\sigma}^2(1 + \mu_k/\lambda))]$ — zero $(1 - H_{ii})^{-1}$ singularity. |
| **2. Additive Sensor Offset Failure (`c = 0.5`: `0.0509`, `gate ON 3/5`)** | `v2` hardcoded `self.x_mean = 0` during sensor alignment, so any nonzero offset $c$ injected a rank-1 constant vector that corrupted SVD and shifted $Z$. | **Center-Before-SVD + Joint $(R \in O(K), b_{\text{adapter}})$ Adam Optimization:** Subtracting $\bar{x}_u$ removes any $c \in \mathbb{R}^D$ identically before SVD, while jointly optimizing $b_{\text{adapter}} \in \mathbb{R}^K$ absorbs the finite-sample mean shift inside the active subspace. |
| **3. Sample-Ordering Spread (`0.0405` on `W16_obs_shift`)** | Row-order-dependent SVD sign flips changed the starting frame for finite-step Adam on $O(4)$. | **Canonical Lexicographic Deduplication + Deterministic SVD Sign Canonicalization:** Deduplicate and sort unique states lexicographically before SVD and canonicalize the largest-magnitude entry of each singular vector to be positive. |
| **4. In-Subspace Pure Rotation Miss (`0/5` detected)** | When $M_{\text{in}} = Q M_{\text{src}}$, the unsupervised subspace residual is identically `0`. | **Dual-Trigger Shift Detection with Strict $\chi^2$ Noise-Floor Gate:** When $r_{\text{subspace}} \le 0.10$ and the unaligned core fails the Bayes-factor test, test a fixed-$b=0$ $O(K)$ rotation and accept it only if aligned residual MSE reaches the replicate noise floor ($\le \max(4\hat{\sigma}^2, 0.015\,\text{Var}(y))$). |

---

## 2. Verified Before (`v2`) vs After (`v3`) & Fresh Unseen Confirmation (`121–130`)

### A. Phase 16B Seeds (`101–110`, 20-Shot) — Direct Fix Verification
| World | `v2` Mean (Max) | **`v3` Mean (Med / Max)** | Gate ON | vs `scratch_mlp` (95% CI `> 0`) | vs `finetune_mlp` (95% CI `> 0`) | Source Ret. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`W16_obs_shift`** | `0.0208` (`0.1081`) | **`0.0093`** (`0.0060` / `0.0386`) | `10/10` | **`+0.0468` (True)** | **`+0.0398` (True)** | `0.0012` |
| **`W16_shared_plus_new`** | `0.0232` (`0.0923`) | **`0.0229`** (`0.0069` / `0.0923`) | `10/10` | **`+0.0592` (True)** | **`+0.0385` (True)** | `0.0012` |
| **`W16_rule_shift`** | `0.0206` (`0.0893`) | **`0.0169`** (`0.0108` / `0.0779`) | `2/10` | **`+0.0561` (True)** | **`+0.0806` (True)** | `0.0012` |
| **`W16_negative_control`** | `0.0355` (`0.2013`, **Gate D FAIL**) | **`0.0161`** (`0.0092` / `0.0817`) | **`0/10`** | **`+0.0450` (True)** | **`+0.0732` (True)** | `0.0012` |

### B. Fresh Unseen Independent Seeds (`121–130`, 20-Shot) — Out-of-Sample Confirmation
| World | **`v3` Mean (Med / Max)** | Gate ON | vs `scratch_mlp` (95% CI `> 0`) | vs `finetune_mlp` (95% CI `> 0`) | Source Ret. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`W16_obs_shift`** | **`0.0288`** (`0.0077` / `0.2197`) | `10/10` | **`+0.0355` (True)** | **`+0.0542` (True)** | `0.0012` |
| **`W16_shared_plus_new`** | **`0.0162`** (`0.0114` / `0.0493`) | `10/10` | **`+0.0750` (True)** | **`+0.0507` (True)** | `0.0012` |
| **`W16_rule_shift`** | **`0.0467`** (`0.0136` / `0.1945`) | `1/10` | **`+0.0413` (True)** | **`+0.0626` (True)** | `0.0012` |
| **`W16_negative_control`** | **`0.0340`** (`0.0126` / `0.1086`) | **`0/10`** | **`+0.0412` (True)** | **`+0.0843` (True)** | `0.0012` |

- **Gate D (`W16_negative_control`):** **`PASS`** on both `101–110` AND fresh unseen `121–130`.
- **Beats BOTH `scratch_mlp` and `finetune_mlp` (95% CI `> 0`) on `4/4` worlds** on both `101–110` AND fresh unseen `121–130`.

### C. Boundary Stress-Test Fixes (Fresh Unseen Seeds `121–130`)
| Boundary Stress Test | Phase 16B `v2` | **`v3` (Fresh Seeds `121–130`)** |
| :--- | :---: | :---: |
| **Sample Ordering (`W16_obs_shift`, 5 perms)** | `max spread = 0.0405` | **`max spread = 2.75e-16` (exact invariance)** |
| **Additive Sensor Offset (`c = 0.5`)** | `mean = 0.0509`, `gate ON 3/5` | **`med = 0.0077`, `gate ON 10/10` (exact offset invariance)** |
| **In-Subspace Pure Rotation (`M = Q M_src`)** | `shift = 0/5`, `gate ON = 0/5` | **`shift = 8/10`, `gate ON = 8/10`, `med = 0.0116`** |
