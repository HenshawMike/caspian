# Phase 18 (`CSP-P18-E001`) — Stage 2 Mathematical Foundations & Specification

## 1. Controlled Neutral Dynamical Environment (`Phase18World`)

- **Hidden Latent State:** $s_t \in [-2, 2]^K$ ($K = 4$), internal state (energy) $e_t \in \mathbb{R}$.
- **Neutral Sensor Observation:** $x_t = s_t M_w + c_w \in \mathbb{R}^D$ ($D = 8$, orthonormal $M_w \in \mathbb{R}^{K \times D}$, optional offset $c_w$). The agent receives strictly neutral $o_t = (x_t, e_t, t, a_{t-1})$ audited by `assert_semantic_purity()`. Hidden $s_t, M_w, c_w, w$ (`world_id`), and true dynamics $(A_w, B_w, R_w)$ are strictly inaccessible.
- **Discrete Action Space ($\mathcal{A} = \{0, 1, 2, 3, 4\}$):**
  - $a = 0$ (`NOOP` / passive observation): coupled autonomous drift $s_{t+1} = \text{clip}(A_w s_t + \epsilon_s, -2, 2)$, $\epsilon_s \sim \mathcal{N}(0, \sigma_s^2 I)$.
  - $a \in \{1, 2, 3, 4\}$ (active causal intervention): $s_{t+1} = \text{clip}(A_w s_t + B_w \mathbf{e}_a + \epsilon_s, -2, 2)$, where $B_w$ applies sparse direct interventional shifts breaking observational correlations.
- **Consequence Dynamics:**
  $$\Delta e_t = R_w(s_{t+1}, a_t) + \epsilon_e, \qquad \epsilon_e \sim \mathcal{N}(0, \sigma_e^2)$$
  where $R_w$ is a degree-2 non-additive function of $s_{t+1}$ (with cross-terms such as $+5 s_0 s_1$ and a preparatory cost / basin structure so optimal cumulative return requires chaining $H \ge 2\text{–}3$ actions, defeating 1-step greedy reactive agents).

## 2. Integrated World Model (`models/world_model.py` — `CaspianWorldModel`)

`CaspianWorldModel` composes Phase 17C's `PersistentWorldMemory` with a compact per-engram Bayesian transition & uncertainty model (`WorldDynamicsEngram`):

1. **Perception & Context Retrieval:**
   $$z_t = (x_t - \mu_w) W_{\text{adapter}} + b_{\text{adapter}} \in \mathbb{R}^K$$
   Context engram selection uses zero-`world_id` probe MSE retrieval over `(X_probe, Y_probe)` (and transition residual when available).
2. **Action-Conditioned Causal Transition Model:**
   For each action $a \in \mathcal{A}$ and latent output coordinate $k \in \{0, \dots, K-1\}$, define design vector $\psi(z_t) = [1, z_t] \in \mathbb{R}^{K+1}$ and model $\Delta z_{t+1, k} = \psi(z_t)^\top \theta_{a,k}$.
   - **Causal Sparsity:** Interventional ISTA + Gauss-Lasso pruning on the action-vs-NOOP delta isolates direct causal effects of action $a$ on coordinate $k$, discarding spurious observational correlations.
   - **Empirical Bayes (Type-II ML) Posterior:** Closed-form marginal likelihood ridge yields posterior mean $\hat{\theta}_{a,k}$ and **exact posterior covariance**:
     $$\Sigma_{a,k} = \hat{\sigma}_{a,k}^2 \left(\Psi_a^\top \Psi_a + \lambda_{a,k} I\right)^{-1}$$
   - **Transition Mean & Variance:**
     $$\hat{z}_{t+1, k} = z_{t, k} + \psi(z_t)^\top \hat{\theta}_{a,k}, \qquad \hat{\sigma}^2_{z,k}(z_t, a) = \hat{\sigma}_{a,k}^2 + \psi(z_t)^\top \Sigma_{a,k} \psi(z_t)$$
   - **Analytical Gradients:** Exact analytical Jacobian $\partial \hat{z}_{t+1} / \partial z_t = I + \Theta_a^\top$ and parameter gradients $\partial L / \partial \theta_{a,k}$ verified against central finite differences.
3. **Consequence Model & Posterior Covariance:**
   Reuses the active `ModularTransferModelV3` engram's degree-2 basis $\Phi(z) \in \mathbb{R}^{14}$ and stores its exact Empirical Bayes posterior covariance $\Sigma_R = \hat{\sigma}_R^2 (\Phi_w^\top \Phi_w + \lambda_R I)^{-1}$:
   $$\Delta \hat{e}(\hat{z}_{t+1}, a) = \Phi(\hat{z}_{t+1})^\top w_{\text{eff}} + b_{\text{eff}} - c_{\text{act}}(a), \qquad \hat{\sigma}^2_{\Delta e}(\hat{z}_{t+1}) = \hat{\sigma}_R^2 + \Phi(\hat{z}_{t+1})^\top \Sigma_R \Phi(\hat{z}_{t+1})$$
4. **Multi-Step Uncertainty Propagation & Subspace OOD Score:**
   Starting from $P_0 = 0$:
   $$P_{h+1} = J_h P_h J_h^\top + \text{diag}(\hat{\sigma}_z^2(\hat{z}_h, a_h)), \qquad \hat{\sigma}^2_{\text{total}, h} = \hat{\sigma}^2_{\Delta e}(\hat{z}_{h+1}) + g_h^\top P_{h+1} g_h$$
   where $J_h = I + \Theta_{a_h}^\top$ and $g_h = \nabla_z \Delta \hat{e}(\hat{z}_{h+1})$.
   Out-of-subspace novelty is quantified directly via:
   $$u_{\text{ood}}(x_t) = \frac{\|(x_t - \mu_w)(I - W_{\text{adapter}} W_{\text{adapter}}^\top)\|_2}{\|x_t - \mu_w\|_2 + 10^{-8}}$$

## 3. Counterfactual Evaluator & Autonomous Planner (`models/planner.py`)

- **Counterfactual Evaluation:** Evaluates every $a \in \mathcal{A}$ internally without touching the environment. When given an observed transition $(x_t, a_t^{\text{obs}}, x_{t+1}^{\text{obs}}, \Delta e_t^{\text{obs}})$, uses Pearl's Abduction-Action-Prediction:
  $$\hat{u}_z = z_{t+1}^{\text{obs}} - \hat{z}_{t+1}(z_t, a_t^{\text{obs}}), \qquad \hat{z}_{t+1}^{\text{cf}}(a') = \hat{z}_{t+1}(z_t, a') + \hat{u}_z$$
- **Uncertainty-Aware Receding-Horizon MPC:**
  $$\mathbf{a}^* = \arg\max_{\mathbf{a} \in \mathcal{A}^H} \sum_{h=0}^{H-1} \gamma^h \left[ \Delta \hat{e}_{t+h} - \beta \cdot \hat{\sigma}_{\text{total}, t+h} \right]$$
  Executes only $a_0^*$, observes real $(x_{t+1}, \Delta e_t)$, and replans at $t+1$.
- **Closed-Loop Surprise Monitoring & Recovery:** Computes standardized surprise $S_t = |\Delta e_t - \Delta \hat{e}_t| / \max(\hat{\sigma}_{\text{total}, t}, 10^{-3})$. Sustained high surprise autonomously triggers `PersistentWorldMemory` re-retrieval or online outdated-engram adaptation.
