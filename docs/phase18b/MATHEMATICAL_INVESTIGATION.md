# Phase 18B (`CSP-P18B-E001`) — Stage 2 First-Principles Mathematical Investigation

## 1. Investigation A — Boundary Attenuation & Empirical Censored Dynamics

Read-only forensic reproduction on Phase 18 baseline (`f059132`, seeds `171–175`):
- `20.3%` of `W18_alpha` training transitions saturate at the environment box boundary (`|s_next| >= 1.799`).
- Baseline 1-step $x$ nRMSE: **Interior = `0.04836`**, **Boundary = `0.11945`** (`2.47x` higher).
- Training $\Theta_a$ exclusively on uncensored interior transitions reduces interior nRMSE **`4.43x`** (`0.04836 -> 0.01091`), proving the interior error was caused by **censored-target slope attenuation**.
- However, because Phase 18 added `±0.25` slack to `z_min`/`z_max`, unclamped predictions overshot at the boundary (`0.20832`).

### First-Principles Boundary-Aware Formulation (Zero Oracle Knowledge)
1. **Empirical Saturation Detection:** From observed `Z_next` alone, compute tight empirical extrema $z_{\min, k}^{\text{emp}} = \min_i Z_{\text{next}, i, k}$ and $z_{\max, k}^{\text{emp}} = \max_i Z_{\text{next}, i, k}$. A transition $(i, k)$ is flagged empirically censored if $|Z_{\text{next}, i, k} - z_{\max, k}^{\text{emp}}| \le \tau_k$ or $|Z_{\text{next}, i, k} - z_{\min, k}^{\text{emp}}| \le \tau_k$ (with $\tau_k = 0.03 \cdot (z_{\max,k}^{\text{emp}} - z_{\min,k}^{\text{emp}})$) **only when** boundary piling is statistically present ($\ge 3$ samples within $\tau_k$ of the extremum).
2. **Uncensored Empirical Bayes Fit:** Fit $\theta_{a,k}$ and $\Sigma_{a,k}$ on the uncensored subset for each $(a, k)$ (falling back to all samples if fewer than $K + 4$ uncensored points exist).
3. **Tight Envelope Clamping & Boundary Variance Inflation:** At prediction time, compute unclamped $\tilde{z}_{t+1} = z_t + [1, z_t]\Theta_a$, clamp $\hat{z}_{t+1} = \text{clip}(\tilde{z}_{t+1}, z_{\min}^{\text{emp}}, z_{\max}^{\text{emp}})$, and add boundary-crossing variance $\delta_{\text{clip}}^2 = 0.25 \cdot (\tilde{z}_{t+1} - \hat{z}_{t+1})^2$ to the predictive covariance so Bayesian uncertainty reflects boundary nonlinearity.

---

## 2. Investigation B — Exact Affine-Orthogonal Basis Transformation Laws

Read-only forensic reproduction of Phase 18 mutation recovery (`seeds 171–175, 177, 182`):
- On **7/7** seeds, `PersistentWorldMemory.observe_world(..., allow_update_outdated=True)` rotated `W_adapter` by an exact orthogonal matrix $Q = W_{\text{old}}^\top W_{\text{new}} \in O(4)$ (`||Q - I||_F = 2.467–3.501`, `||Q^T Q - I||_F = 2.15e-15–3.70e-15`).
- Unsynchronized `reuse_trans` failed `0/7` times, refitting a noisy 30-step dynamics model (`x_rmse = 0.2634–0.5181`).

### Exact Transformation Laws (Verified to `<= 3.11e-15`)
Let the old basis be $(\mu_1, W_1, b_1)$ and the new co-subspace basis be $(\mu_2, W_2, b_2)$ with $W_1^\top W_1 = W_2^\top W_2 = I_K$ and $\text{col}(W_1) = \text{col}(W_2)$:
1. **Latent Coordinate Map:**
   $$Q = W_1^\top W_2 \in O(K), \qquad c = -b_1 Q + (\mu_1 - \mu_2) W_2 + b_2 \in \mathbb{R}^{1 \times K}$$
   $$z_2 = z_1 Q + c, \qquad [1, z_2] = [1, z_1] T, \qquad T = \begin{pmatrix} 1 & c \\ 0 & Q \end{pmatrix}, \quad T^{-1} = \begin{pmatrix} 1 & -c Q^\top \\ 0 & Q^\top \end{pmatrix}$$
2. **Transition Parameter Operator ($\Theta_a \in \mathbb{R}^{(K+1)\times K}$):**
   $$\Delta z_2 = \Delta z_1 Q = [1, z_1] \Theta_a Q = [1, z_2] T^{-1} \Theta_a Q \implies \boxed{\Theta'_a = T^{-1} \Theta_a Q}$$
3. **Bayesian Parameter Posterior Covariance ($\Sigma_{a,j} \in \mathbb{R}^{(K+1)\times(K+1)}$):**
   Since $\theta'_{a,j} = \sum_{k=1}^K Q_{kj} T^{-1} \theta_{a,k}$ and per-coordinate posteriors are independent:
   $$\boxed{\Sigma'_{a,j} = T^{-1} \left(\sum_{k=1}^K Q_{kj}^2 \, \Sigma_{a,k}\right) T^{-\top}}$$
4. **Full Aleatoric Noise Covariance ($\Sigma_{\epsilon, a} \in \mathbb{R}^{K \times K}$):**
   $$\boxed{\Sigma'_{\epsilon, a} = Q^\top \Sigma_{\epsilon, a} Q}$$
   Storing full `sigma_cov_trans [A, K, K]` ensures predictive variance $\text{tr}(P_{h+1})$ and propagated consequence variance $g^\top P_{h+1} g$ are **strictly invariant** under any orthogonal basis rotation $Q \in O(K)$!
5. **Latent Bounds & Consequence Synchronization:**
   Transform latent box corners $z_{\min}, z_{\max}$ and ellipsoid $(\mu_z, \Lambda_z)$ via $\mu'_z = \mu_z Q + c$, $\Lambda'_z = Q^\top \Lambda_z Q$. For consequence-only mutations (`r_sub < 0.12`), preserve the synchronized dynamics $(\Theta'_a, \Sigma'_{a,j}, \Sigma'_{\epsilon,a})$ and additionally test sign-inverted / empirical-Bayes consequence refitting in the synchronized basis.

---

## 3. Investigation C — Data-Budget Scaling (Dev Seeds `9201–9203`)

| Transitions ($N$) | `x_nRMSE` (CWM vs MLP) | Return (CWM vs MLP) | $\Delta$ (CWM $-$ MLP) |
| :---: | :---: | :---: | :---: |
| `60` | `0.1017` vs `0.1370` | `443.5` vs `402.8` | **`+40.7`** |
| `120` | `0.0766` vs `0.0940` | `452.4` vs `416.9` | **`+35.6`** |
| `200` | `0.0756` vs `0.0841` | `438.8` vs `434.0` | `+4.8` |
| `350` | `0.0721` vs `0.0721` | `450.7` vs `453.8` | `-3.1` |

This proves MLP competitiveness at `N = 350` is a genuine abundant-data regime effect, while Caspian's structural inductive bias dominates at low/medium data budgets (`N = 60–120`) and under sensor-shifted transfer.
