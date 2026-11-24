# Phase 18D (`CSP-P18D-E001`) — Stage 1 First-Principles Mathematical Foundations

## 1. Why Consequence Alignment Does Not Imply Dynamics Alignment
Let $R: \mathbb{R}^K \to \mathbb{R}$ be a scalar consequence rule with symmetry group $\mathcal{G}_R = \{Q \in O(K) : R(z Q) = R(z)\}$. Any representation alignment minimizing consequence loss identifies the latent coordinates $z_{\text{tgt}}$ only up to $Q \in \mathcal{G}_R$. By contrast, the action-conditioned transition family $\{T_a\}_{a=0}^{A-1}$ with interventional shift vectors $\delta_a = \theta_{a,0} - \theta_{0,0}$ has trivial joint symmetry group whenever $\text{span}\{\delta_1, \dots, \delta_{A-1}\} = \mathbb{R}^K$.

## 2. Exact Dynamics-Frame Alignment via Robust Leave-One-Out Orthogonal Procrustes
Suppose source engram dynamics $(\Theta_a^{\text{src}}, \Sigma_{a,k}^{\text{src}}, \Sigma_{\epsilon, a}^{\text{src}}, U_{\text{box}}^{\text{src}}, \ell^{\text{src}}, u^{\text{src}})$ and target latent transitions $(Z_t, A_t, Z_{\text{next}})$ are related (on at least $A - 2$ compatible actions) by an unknown affine-orthogonal map $z_{\text{tgt}} = z_{\text{src}} Q + c$:
1. **Action-Shift Extraction:** Estimate empirical target interventional shifts $\delta_a^{\text{tgt}}$ ($a = 1 \dots A-1$) from target transitions relative to NOOP ($a = 0$).
2. **Robust Orthogonal Procrustes (Handles Regime C Partial Action Mutation!):**
   Evaluate candidate subsets $S \subseteq \{1, \dots, A-1\}$ of size $A-1$ (all actions) and $A-2$ (leave-one-action-out, immune to a single mutated action!). For each subset $S$, solve the closed-form SVD Procrustes problem:
   $$Q_S^* = \arg\min_{Q \in O(K)} \sum_{a \in S} \|\delta_a^{\text{src}} Q - \delta_a^{\text{tgt}}\|_2^2 = U_S V_S^\top, \qquad U_S \Sigma_S V_S^\top = (\Delta_S^{\text{src}})^\top \Delta_S^{\text{tgt}}$$
3. **Closed-Form Offset $c_S^*$:**
   Given $Q_S^*$, the transformed linear drift matrix is $D_a' = Q_S^{*\top} \Theta_{a, 1:}^{\text{src}} Q_S^*$ and intercept is $b_a' = \Theta_{a, 0}^{\text{src}} Q_S^*$. The unknown offset $c \in \mathbb{R}^{1 \times K}$ satisfies:
   $$c D_a' \approx z_t D_a' + b_a' - (z_{\text{next}} - z_t)$$
   solved in closed form via regularized least squares across transitions in $S \cup \{0\}$.
4. **Selection:** Pick $(Q^*, c^*)$ minimizing trimmed/median transition residual across target transitions (strictly on target training data, zero test leakage), then transform the source dynamics candidate `cand_dyn` into the target frame via our exact Phase 18C transformation law:
   $$\Theta'_a = T^{-1} \Theta_a^{\text{src}} Q^*, \quad \Sigma'_{a,k} = T^{-1} \Sigma_{a,k}^{\text{src}} T^{-\top}, \quad \Sigma'_{\epsilon,a} = Q^{*\top} \Sigma_{\epsilon,a}^{\text{src}} Q^*, \quad U'_{\text{box}} = Q^{*\top} U_{\text{box}}^{\text{src}}$$
   $$\ell'_{\text{box}} = \ell_{\text{box}}^{\text{src}} + c^* U'_{\text{box}}, \quad u'_{\text{box}} = u_{\text{box}}^{\text{src}} + c^* U'_{\text{box}}$$

## 3. Uncertainty-Guided Component-Level Compatibility Tests

Once `cand_dyn` is aligned into the target frame, Caspian evaluates **two decoupled Bayesian compatibility tests** strictly on target training transitions:

### A. Per-Action Dynamics Compatibility (Distinguishes Regimes A/B vs. C vs. D)
For each action $a \in \{0, \dots, A-1\}$ with target training index set $I_a$:
- Compute the unclamped interior prediction $\hat{z}_i^{\text{cand}} = z_{t,i} + [1, z_{t,i}]\Theta'_a$ and exact Bayesian predictive covariance $C_{a, i} = Q_{\text{epi}}'^\top \text{diag}(\psi_i \Sigma'_{a,k} \psi_i^\top) Q_{\text{epi}}' + \Sigma'_{\epsilon, a}$.
- Compute the **normalized Bayesian Mahalanobis statistic** (evaluated on interior points not clipped by the boundary, using median/trimmed mean so boundary shifts in Regime B do not falsely reject valid interior dynamics!):
  $$m_a^2 = \text{median}_{i \in I_a} \left[ \frac{1}{K} (z_{\text{next}, i} - \hat{z}_i^{\text{cand}})^\top C_{a, i}^{-1} (z_{\text{next}, i} - \hat{z}_i^{\text{cand}}) \right]$$
- **Decision Rule:**
  - $m_a^2 \le \tau_{\text{preserve}}$ (`4.5`): **PRESERVE / TRANSFER** `cand_dyn.theta_trans[a]` (Bayesian precision-weighted posterior fusion with target observations if $|I_a| \ge 4$, keeping low-variance source knowledge!).
  - $\tau_{\text{preserve}} < m_a^2 \le \tau_{\text{adapt}}$ (`12.0`): **ADAPT** via Empirical Bayes shrinkage toward `cand_dyn.theta_trans[a]`.
  - $m_a^2 > \tau_{\text{adapt}}$ (`12.0`): **REJECT & RECONSTRUCT** action $a$ solely from target transitions!

### B. Boundary Compatibility Test (Distinguishes Regime A vs. Regime B/D)
Given the transferred box frame $U'_{\text{box}}$ and bounds $(\ell'_{\text{box}}, u'_{\text{box}})$:
- Project target next states $V_n = Z_{\text{next}} U'_{\text{box}}$ and predicted unclamped states $V_u = \tilde{Z} U'_{\text{box}}$ (using the accepted/reconstructed $\Theta_a$).
- Compare post-clipping MSE of **transferred bounds** $(\ell'_{\text{box}}, u'_{\text{box}})$ vs. **adapted target bounds** $(\hat{\ell}_{\text{tgt}}, \hat{u}_{\text{tgt}})$ (refined on target transitions in the transferred frame $U'_{\text{box}}$ via 1D coordinate clipping MSE!):
  - **Violation / Tightening Test:** If target points violate transferred bounds by $> 3 \hat{\sigma}$ (`max(V_n - u') > 0.15` or `max(l' - V_n) > 0.15`), OR adapted bounds reduce target post-clip MSE by $> 25\%$ (with $\ge 3$ boundary points) → **ADAPT** $(\ell_{\text{box}}, u_{\text{box}})$ while preserving $U'_{\text{box}}$ (Regime B)!
  - If all actions were rejected (Regime D) → **RECONSTRUCT** full $(U_{\text{box}}, \ell_{\text{box}}, u_{\text{box}})$ from target transitions!
  - Otherwise → **PRESERVE / TRANSFER** $(U'_{\text{box}}, \ell'_{\text{box}}, u'_{\text{box}})$ intact (Regime A)!
