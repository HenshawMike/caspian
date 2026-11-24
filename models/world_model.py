"""Phase 18: Integrated Predictive World Model (models/world_model.py).

Composes Phase 17C's verified PersistentWorldMemory (models/memory.py) and
ModularTransferModelV3 (models/transfer_v3.py) with:
1. Action-conditioned Bayesian causal latent transition dynamics:
       z_{t+1} = z_t + [1, z_t] @ Theta_a
   with Type-II ML (Empirical Bayes) posterior covariance Sigma_{a,k} and
   interventional ISTA + Gauss-Lasso causal action-effect discovery.
2. Exact Empirical Bayes posterior covariance Sigma_R for consequence readout Phi(z).
3. First-order Jacobian uncertainty propagation across multi-step open-loop rollouts
   plus normalized out-of-subspace OOD detection u_ood(x).
4. Purely internal counterfactual evaluation (including Pearl Abduction-Action-Prediction)
   with zero environment access and zero world_id privilege.
"""
from dataclasses import dataclass
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.memory import PersistentWorldMemory
from models.transfer_v3 import ModularTransferModelV3


@dataclass
class BayesianDynamicsEngram:
    """Compact Bayesian action-conditioned dynamics + uncertainty state for one world engram."""
    engram_id: int
    n_actions: int
    n_latent: int
    theta_trans: np.ndarray        # (A, K+1, K): row 0 = intercept, rows 1..K = linear state-delta weights
    cov_trans: np.ndarray          # (A, K, K+1, K+1): exact Empirical Bayes posterior covariance
    sigma2_trans: np.ndarray       # (A, K): aleatoric transition noise variance
    causal_action_mask: np.ndarray # (A, K): binary mask of direct causal action effects vs NOOP
    cov_outcome: np.ndarray        # (phi_dim+1, phi_dim+1): exact EB posterior covariance for [1, Phi(z)]
    sigma2_outcome: float          # aleatoric consequence noise variance
    action_costs: np.ndarray       # (A,): learned direct per-action consequence offset
    z_min: np.ndarray              # (K,): empirical latent lower bound (with margin)
    z_max: np.ndarray              # (K,): empirical latent upper bound (with margin)
    z_mean: np.ndarray             # (K,): empirical latent mean (for Mahalanobis / OOD diagnostics)
    z_cov_inv: np.ndarray          # (K, K): regularized inverse latent covariance


class CaspianWorldModel:
    """Pure-NumPy uncertainty-aware predictive world model integrating PersistentWorldMemory."""

    def __init__(
        self,
        in_dim: int = 8,
        n_latent: int = 4,
        n_actions: int = 5,
        seed: int = 0,
        alignment_mode: str = "occam_hierarchy",
    ):
        self.in_dim = int(in_dim)
        self.n_latent = int(n_latent)
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self.memory = PersistentWorldMemory(
            in_dim=self.in_dim,
            n_latent=self.n_latent,
            seed=self.seed,
            alignment_mode=alignment_mode,
        )
        self.dynamics: Dict[int, BayesianDynamicsEngram] = {}
        self.raw_replay_samples_stored: int = 0
        self._last_z_cache: Optional[np.ndarray] = None
        self._last_a_cache: Optional[int] = None
        self.grad_theta_trans: Optional[np.ndarray] = None

    @property
    def active_engram_id(self) -> Optional[int]:
        return self.memory.active_engram_id

    @property
    def num_engrams(self) -> int:
        return self.memory.num_engrams

    @property
    def num_parameters(self) -> int:
        mem_p = self.memory.num_parameters
        dyn_p = sum(
            int(d.theta_trans.size + d.action_costs.size)
            for d in self.dynamics.values()
        )
        return int(mem_p + dyn_p)

    def _get_engram(self, eid: Optional[int] = None):
        use_id = self.memory.active_engram_id if eid is None else eid
        assert use_id is not None and len(self.memory.engrams) > 0, "No active engram in world model."
        return next(e for e in self.memory.engrams if e.engram_id == use_id)

    def encode(self, X: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        """Encode neutral sensor observation X (..., D) into canonical latent Z (..., K)."""
        X_arr = np.asarray(X, dtype=float)
        squeeze = (X_arr.ndim == 1)
        if squeeze:
            X_arr = X_arr.reshape(1, -1)
        e = self._get_engram(eid)
        Z = (X_arr - e.model.x_mean) @ e.model.W_adapter + e.model.b_adapter
        return Z[0] if squeeze else Z

    def decode_to_obs(self, Z: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        """Project canonical latent Z (..., K) back into neutral sensor observation space (..., D)."""
        Z_arr = np.asarray(Z, dtype=float)
        squeeze = (Z_arr.ndim == 1)
        if squeeze:
            Z_arr = Z_arr.reshape(1, -1)
        e = self._get_engram(eid)
        X_hat = (Z_arr - e.model.b_adapter) @ e.model.W_adapter.T + e.model.x_mean
        return X_hat[0] if squeeze else X_hat

    def subspace_ood_score(self, X: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        """Per-sample normalized out-of-subspace residual in [0, 1] (zero world_id, zero labels)."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        e = self._get_engram(eid)
        Xc = X_arr - e.model.x_mean
        W = e.model.W_adapter
        proj = (Xc @ W) @ W.T
        num = np.linalg.norm(Xc - proj, axis=1)
        den = np.linalg.norm(Xc, axis=1) + 1e-8
        return num / den

    @staticmethod
    def _eb_linear_posterior(
        Psi: np.ndarray,
        y: np.ndarray,
        lambdas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 0.1, 0.5, 2.0, 10.0, 50.0),
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Closed-form Empirical Bayes (Type-II ML) ridge posterior for y = Psi @ theta + eps.

        Returns:
            theta: (P,) posterior mean
            cov: (P, P) posterior covariance sigma2 * (Psi^T Psi + lam * I)^{-1}
            sigma2: estimated aleatoric variance
            best_lam: selected precision ratio
        """
        Psi = np.asarray(Psi, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        N, P = Psi.shape
        if N == 0:
            return np.zeros(P, dtype=float), np.eye(P, dtype=float), 1.0, 1.0

        # Estimate initial noise variance from OLS residual (with safe fallback for small N)
        if N > P + 2:
            beta_ols, *_ = np.linalg.lstsq(Psi, y, rcond=None)
            res = y - Psi @ beta_ols
            sigma2_est = float(np.sum(res ** 2) / max(N - P, 1))
        else:
            sigma2_est = 0.05 * max(float(np.var(y)), 1e-4)
        sigma2_est = max(sigma2_est, 1e-5)

        G = Psi.T @ Psi
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, 0.0)
        # Dual kernel eigenvalues for marginal likelihood
        Kmat = Psi @ Psi.T
        keig, kvec = np.linalg.eigh(Kmat)
        keig = np.maximum(keig, 0.0)
        proj2 = (kvec.T @ y) ** 2

        best_nll = float("inf")
        best_lam = lambdas[0]
        for lam in lambdas:
            var_k = sigma2_est * (1.0 + keig / lam)
            nll = 0.5 * float(np.sum(proj2 / var_k + np.log(var_k)))
            if nll < best_nll:
                best_nll = nll
                best_lam = lam

        A_inv = np.linalg.inv(G + best_lam * np.eye(P, dtype=float))
        theta = A_inv @ (Psi.T @ y)
        # Refine sigma2 using effective degrees of freedom gamma = sum(eig / (eig + lam))
        df_eff = float(np.sum(eigvals / (eigvals + best_lam)))
        res_post = y - Psi @ theta
        sigma2 = max(float(np.sum(res_post ** 2) / max(N - df_eff, 1.0)), 1e-5)
        cov = sigma2 * A_inv
        return theta, cov, float(sigma2), float(best_lam)

    @staticmethod
    def _discover_causal_action_mask(
        Z_t: np.ndarray,
        A_t: np.ndarray,
        Z_next: np.ndarray,
        theta_noop: np.ndarray,
        n_actions: int,
        l1_lambda: float = 0.08,
        effect_threshold: float = 0.22,
    ) -> np.ndarray:
        """Discover which latent coordinates k are directly intervened on by each action a
        via ISTA + Gauss-Lasso on action-vs-NOOP residuals (separating intervention from passive drift).
        """
        K = Z_t.shape[1]
        mask = np.zeros((n_actions, K), dtype=float)
        Psi_all = np.hstack([np.ones((len(Z_t), 1)), Z_t])
        dZ_noop_pred = Psi_all @ theta_noop  # passive drift baseline (N, K)
        dZ_resid = (Z_next - Z_t) - dZ_noop_pred

        # Build action indicator matrix for actions 1..A-1
        act_cols = []
        for a in range(1, n_actions):
            act_cols.append((A_t == a).astype(float))
        if not act_cols:
            return mask
        D_act = np.column_stack(act_cols)  # (N, A-1)
        N = float(len(Z_t))
        L_lip = float(np.linalg.norm(D_act, ord=2) ** 2 / max(N, 1.0)) + 1e-6
        lr = 0.5 / L_lip
        thr = l1_lambda * lr

        for k in range(K):
            yk = dZ_resid[:, k]
            w = np.zeros(n_actions - 1, dtype=float)
            for _ in range(150):
                grad = (D_act.T @ (D_act @ w - yk)) / max(N, 1.0)
                w_c = w - lr * grad
                w = np.sign(w_c) * np.maximum(0.0, np.abs(w_c) - thr)
            active = np.where(np.abs(w) >= effect_threshold * 0.5)[0]
            if len(active) > 0:
                beta, *_ = np.linalg.lstsq(D_act[:, active], yk, rcond=None)
                for idx_local, a_idx in enumerate(active):
                    if abs(float(beta[idx_local])) >= effect_threshold:
                        mask[a_idx + 1, k] = 1.0
        return mask

    def observe_transitions(
        self,
        X_t: np.ndarray,
        A_t: np.ndarray,
        X_next: np.ndarray,
        dE_t: np.ndarray,
        allow_update_outdated: bool = False,
        only_noop_obs: bool = False,
    ) -> int:
        """Learn or adapt a world dynamics + consequence engram from neutral transitions (zero world_id).

        Args:
            X_t: (N, D) current neutral sensor observations
            A_t: (N,) integer action codes in [0, n_actions - 1]
            X_next: (N, D) next neutral sensor observations
            dE_t: (N,) observed scalar internal-state consequences
            allow_update_outdated: allow updating an outdated same-subspace engram
            only_noop_obs: if True, fit transition without action-specific interventional separation
                           (used strictly for the H2 observational-only ablation)
        """
        X_t = np.asarray(X_t, dtype=float)
        A_t = np.asarray(A_t, dtype=int).reshape(-1)
        X_next = np.asarray(X_next, dtype=float)
        dE_t = np.asarray(dE_t, dtype=float).reshape(-1)

        # Step 1: Update/retrieve structural consequence engram via verified Phase 17C PersistentWorldMemory
        eid = self.memory.observe_world(X_next, dE_t, allow_update_outdated=allow_update_outdated)
        engram = self._get_engram(eid)

        # Step 2: Encode states into canonical latent coordinates
        Z_t = self.encode(X_t, eid=eid)
        Z_next = self.encode(X_next, eid=eid)
        dZ = Z_next - Z_t
        K = self.n_latent
        A = self.n_actions

        # Step 3: Fit Empirical Bayes action-conditioned latent transitions
        theta_trans = np.zeros((A, K + 1, K), dtype=float)
        cov_trans = np.zeros((A, K, K + 1, K + 1), dtype=float)
        sigma2_trans = np.ones((A, K), dtype=float) * 0.05

        Psi_all = np.hstack([np.ones((len(Z_t), 1)), Z_t])
        # Pooled fallback dynamics (for unseen actions or observational-only ablation)
        theta_pool = np.zeros((K + 1, K), dtype=float)
        cov_pool = np.zeros((K, K + 1, K + 1), dtype=float)
        sig_pool = np.ones(K, dtype=float) * 0.05
        for k in range(K):
            th, cv, sg, _ = self._eb_linear_posterior(Psi_all, dZ[:, k])
            theta_pool[:, k] = th
            cov_pool[k] = cv
            sig_pool[k] = sg

        # Check if we transferred from a source engram with existing dynamics (few-shot transfer)
        src_dyn = (
            self.dynamics.get(engram.source_engram_id)
            if engram.source_engram_id is not None
            else None
        )

        for a in range(A):
            if only_noop_obs:
                theta_trans[a] = theta_pool
                cov_trans[a] = cov_pool
                sigma2_trans[a] = sig_pool
                continue
            idx_a = np.where(A_t == a)[0]
            if len(idx_a) >= K + 3:
                Psi_a = Psi_all[idx_a]
                dZ_a = dZ[idx_a]
                for k in range(K):
                    th, cv, sg, _ = self._eb_linear_posterior(Psi_a, dZ_a[:, k])
                    theta_trans[a, :, k] = th
                    cov_trans[a, k] = cv
                    sigma2_trans[a, k] = sg
            elif src_dyn is not None:
                # Few-shot Bayesian prior inheritance from transferred source engram
                theta_trans[a] = src_dyn.theta_trans[a].copy()
                cov_trans[a] = src_dyn.cov_trans[a].copy() * 1.5
                sigma2_trans[a] = src_dyn.sigma2_trans[a].copy()
            elif len(idx_a) >= 2:
                # Low-sample action: borrow linear drift from pooled, fit intercept shift
                theta_trans[a] = theta_pool.copy()
                cov_trans[a] = cov_pool.copy() * 2.0
                sigma2_trans[a] = sig_pool.copy()
                res_a = dZ[idx_a] - (Psi_all[idx_a] @ theta_pool)
                theta_trans[a, 0, :] += np.mean(res_a, axis=0)
            else:
                theta_trans[a] = theta_pool.copy()
                cov_trans[a] = cov_pool.copy() * 4.0
                sigma2_trans[a] = sig_pool.copy() * 2.0

        # Step 4: Causal action-effect sparsity discovery (separating intervention from passive drift)
        causal_mask = self._discover_causal_action_mask(
            Z_t, A_t, Z_next, theta_trans[0], n_actions=A
        )

        # Step 5: Consequence Empirical Bayes posterior covariance & per-action cost offsets
        Phi_next = engram.model._compute_phi(Z_next)
        Psi_R = np.hstack([np.ones((len(Phi_next), 1)), Phi_next])
        pred_R = engram.model.forward(X_next).reshape(-1)
        res_R = dE_t - pred_R

        action_costs = np.zeros(A, dtype=float)
        for a in range(A):
            idx_a = np.where(A_t == a)[0]
            if len(idx_a) >= 3:
                action_costs[a] = float(-np.mean(res_R[idx_a]))

        # Compute exact Bayesian posterior covariance on consequence basis [1, Phi(z)]
        dE_adj = dE_t + action_costs[A_t]
        _, cov_outcome, sig2_outcome, _ = self._eb_linear_posterior(Psi_R, dE_adj)

        # Empirical latent envelope and Mahalanobis precision
        Z_both = np.vstack([Z_t, Z_next])
        z_min = np.min(Z_both, axis=0) - 0.25
        z_max = np.max(Z_both, axis=0) + 0.25
        z_mean = np.mean(Z_both, axis=0)
        z_cov = np.cov(Z_both.T) + 0.05 * np.eye(K, dtype=float)
        z_cov_inv = np.linalg.inv(z_cov)

        self.dynamics[eid] = BayesianDynamicsEngram(
            engram_id=eid,
            n_actions=A,
            n_latent=K,
            theta_trans=theta_trans,
            cov_trans=cov_trans,
            sigma2_trans=sigma2_trans,
            causal_action_mask=causal_mask,
            cov_outcome=cov_outcome,
            sigma2_outcome=sig2_outcome,
            action_costs=action_costs,
            z_min=z_min,
            z_max=z_max,
            z_mean=z_mean,
            z_cov_inv=z_cov_inv,
        )
        return eid

    def _consequence_and_grad(
        self, Z: np.ndarray, A: np.ndarray, eid: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute consequence mean, epistemic+aleatoric variance, and analytical d(dE)/dZ."""
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        Z_arr = np.asarray(Z, dtype=float)
        A_arr = np.asarray(A, dtype=int).reshape(-1)
        K = self.n_latent

        Phi = e.model._compute_phi(Z_arr)
        w_eff = (e.model.core_gate * e.model.w_core + e.model.w_world).reshape(-1)
        b_eff = float((e.model.core_gate * e.model.b_core + e.model.b_world).item())

        de_mean = Phi @ w_eff + b_eff - dyn.action_costs[A_arr]

        # Exact Bayesian predictive variance: sigma2_R + psi_R^T Sigma_R psi_R
        Psi_R = np.hstack([np.ones((len(Phi), 1)), Phi])
        epi_R = np.einsum("ni,ij,nj->n", Psi_R, dyn.cov_outcome, Psi_R)
        de_var = dyn.sigma2_outcome + np.maximum(epi_R, 0.0)

        # Analytical gradient d(dE)/dZ for variance propagation
        grad_z = np.tile(w_eff[:K], (len(Z_arr), 1))
        for idx, (i, j) in enumerate(e.model.pair_indices):
            wk = w_eff[K + idx]
            grad_z[:, i] += wk * Z_arr[:, j]
            grad_z[:, j] += wk * Z_arr[:, i]

        return de_mean, de_var, grad_z

    def predict_step_latent(
        self,
        Z_t: np.ndarray,
        A_t: np.ndarray,
        P_t: Optional[np.ndarray] = None,
        eid: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Vectorized 1-step latent transition + consequence prediction with uncertainty propagation."""
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        Z_arr = np.asarray(Z_t, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(1, -1)
        A_arr = np.asarray(A_t, dtype=int).reshape(-1)
        N, K = Z_arr.shape

        Psi = np.hstack([np.ones((N, 1)), Z_arr])  # (N, K+1)
        z_next = np.zeros((N, K), dtype=float)
        z_var = np.zeros((N, K), dtype=float)
        P_next = np.zeros((N, K, K), dtype=float)

        for a in range(self.n_actions):
            idx = np.where(A_arr == a)[0]
            if len(idx) == 0:
                continue
            Psi_a = Psi[idx]
            th_a = dyn.theta_trans[a]  # (K+1, K)
            dz_a = Psi_a @ th_a
            z_next[idx] = np.clip(Z_arr[idx] + dz_a, dyn.z_min, dyn.z_max)

            for k in range(K):
                epi_k = np.einsum("ni,ij,nj->n", Psi_a, dyn.cov_trans[a, k], Psi_a)
                z_var[idx, k] = dyn.sigma2_trans[a, k] + np.maximum(epi_k, 0.0)

            J_a = np.eye(K, dtype=float) + th_a[1:, :].T  # d(z_next)/d(z_t)
            if P_t is not None:
                P_prop = np.einsum("ij,njk,lk->nil", J_a, P_t[idx], J_a)
                P_next[idx] = P_prop
            for k in range(K):
                P_next[idx, k, k] += z_var[idx, k]

        de_mean, de_var_base, grad_z = self._consequence_and_grad(z_next, A_arr, eid=e.engram_id)
        # Propagated state uncertainty into consequence variance: g^T P_next g
        prop_var = np.einsum("ni,nij,nj->n", grad_z, P_next, grad_z)
        de_var_total = de_var_base + np.maximum(prop_var, 0.0)
        de_std_total = np.sqrt(np.maximum(de_var_total, 1e-10))

        return {
            "z_next": z_next,
            "z_var": z_var,
            "P_next": P_next,
            "de_mean": de_mean,
            "de_var": de_var_total,
            "de_std": de_std_total,
        }

    def predict_step(
        self, X_t: np.ndarray, A_t: np.ndarray, eid: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Predict next observation, latent state, consequence, and uncertainty from raw neutral X_t."""
        X_arr = np.asarray(X_t, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        A_arr = np.asarray(A_t, dtype=int).reshape(-1)
        e = self._get_engram(eid)
        Z_t = self.encode(X_arr, eid=e.engram_id)
        ood = self.subspace_ood_score(X_arr, eid=e.engram_id)
        out = self.predict_step_latent(Z_t, A_arr, P_t=None, eid=e.engram_id)
        out["x_next"] = self.decode_to_obs(out["z_next"], eid=e.engram_id)
        out["ood_score"] = ood
        # Add OOD inflation to uncertainty if observation lies outside the learned sensor subspace
        ood_inflate = (10.0 * ood) ** 2
        out["de_var"] = out["de_var"] + ood_inflate
        out["de_std"] = np.sqrt(np.maximum(out["de_var"], 1e-10))
        return out

    def rollout_latent(
        self,
        X_0: np.ndarray,
        action_sequences: np.ndarray,
        eid: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Purely internal open-loop multi-step rollout over N candidate action sequences of length H.

        Args:
            X_0: (D,) or (N, D) initial neutral observation(s)
            action_sequences: (N, H) integer action sequences
        """
        e = self._get_engram(eid)
        seqs = np.asarray(action_sequences, dtype=int)
        if seqs.ndim == 1:
            seqs = seqs.reshape(1, -1)
        N, H = seqs.shape
        X0_arr = np.asarray(X_0, dtype=float)
        if X0_arr.ndim == 1:
            X0_arr = np.tile(X0_arr.reshape(1, -1), (N, 1))

        Z_cur = self.encode(X0_arr, eid=e.engram_id)
        ood_0 = self.subspace_ood_score(X0_arr, eid=e.engram_id)
        K = self.n_latent
        P_cur = np.zeros((N, K, K), dtype=float)

        z_traj = np.zeros((N, H + 1, K), dtype=float)
        x_traj = np.zeros((N, H + 1, self.in_dim), dtype=float)
        de_traj = np.zeros((N, H), dtype=float)
        de_std_traj = np.zeros((N, H), dtype=float)
        z_var_traj = np.zeros((N, H, K), dtype=float)

        z_traj[:, 0, :] = Z_cur
        x_traj[:, 0, :] = self.decode_to_obs(Z_cur, eid=e.engram_id)

        for h in range(H):
            step_out = self.predict_step_latent(Z_cur, seqs[:, h], P_t=P_cur, eid=e.engram_id)
            Z_cur = step_out["z_next"]
            P_cur = step_out["P_next"]
            z_traj[:, h + 1, :] = Z_cur
            x_traj[:, h + 1, :] = self.decode_to_obs(Z_cur, eid=e.engram_id)
            de_traj[:, h] = step_out["de_mean"]
            de_std_traj[:, h] = np.sqrt(step_out["de_var"] + (10.0 * ood_0) ** 2)
            z_var_traj[:, h, :] = step_out["z_var"]

        return {
            "z_traj": z_traj,
            "x_traj": x_traj,
            "de_traj": de_traj,
            "de_std_traj": de_std_traj,
            "z_var_traj": z_var_traj,
        }

    def evaluate_counterfactuals(
        self,
        X_t: np.ndarray,
        observed_action: Optional[int] = None,
        X_next_obs: Optional[np.ndarray] = None,
        dE_obs: Optional[float] = None,
        eid: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Internally evaluate all A candidate actions from X_t without executing any of them.

        If (observed_action, X_next_obs, dE_obs) are supplied, applies Pearl's 3-step
        Abduction-Action-Prediction counterfactual inference.
        """
        e = self._get_engram(eid)
        x_row = np.asarray(X_t, dtype=float).reshape(1, -1)
        all_actions = np.arange(self.n_actions, dtype=int)
        X_rep = np.tile(x_row, (self.n_actions, 1))
        out = self.predict_step(X_rep, all_actions, eid=e.engram_id)

        if observed_action is not None and X_next_obs is not None:
            z_obs_next = self.encode(np.asarray(X_next_obs, dtype=float).reshape(1, -1), eid=e.engram_id)[0]
            u_z = z_obs_next - out["z_next"][int(observed_action)]
            z_cf = out["z_next"] + u_z.reshape(1, -1)
            de_cf, _, _ = self._consequence_and_grad(z_cf, all_actions, eid=e.engram_id)
            if dE_obs is not None:
                u_e = float(dE_obs) - float(de_cf[int(observed_action)])
                de_cf = de_cf + u_e
            out["z_next"] = z_cf
            out["x_next"] = self.decode_to_obs(z_cf, eid=e.engram_id)
            out["de_mean"] = de_cf
        return out

    def retrieve_context(
        self,
        X_probe: np.ndarray,
        A_probe: np.ndarray,
        X_next_probe: np.ndarray,
        dE_probe: np.ndarray,
    ) -> Tuple[Optional[int], float, bool]:
        """Autonomous zero-world_id context retrieval combining consequence MSE and transition residual."""
        if not self.memory.engrams:
            return None, float("inf"), False

        X_p = np.asarray(X_probe, dtype=float)
        A_p = np.asarray(A_probe, dtype=int).reshape(-1)
        Xn_p = np.asarray(X_next_probe, dtype=float)
        dE_p = np.asarray(dE_probe, dtype=float).reshape(-1)
        var_de = max(float(np.var(dE_p)), 1e-6)

        best_eid: Optional[int] = None
        best_score = float("inf")
        best_de_mse = float("inf")

        for e in self.memory.engrams:
            if e.engram_id not in self.dynamics:
                continue
            pred = self.predict_step(X_p, A_p, eid=e.engram_id)
            de_mse = float(np.mean((pred["de_mean"] - dE_p) ** 2))
            x_mse = float(np.mean((pred["x_next"] - Xn_p) ** 2))
            score = de_mse + 5.0 * x_mse
            if score < best_score:
                best_score = score
                best_de_mse = de_mse
                best_eid = e.engram_id

        assert best_eid is not None
        e_best = self._get_engram(best_eid)
        thresh = max(8.0 * e_best.sigma2_ref, 0.05 * var_de)
        confident = bool(best_de_mse <= thresh)
        self.memory.active_engram_id = best_eid
        self.memory.last_retrieved_id = best_eid
        return best_eid, best_de_mse, confident

    # Pure-NumPy analytical forward/backward pass for transition parameters (gradient-checked in tests)
    def forward_transition(self, Z: np.ndarray, action: int, eid: Optional[int] = None) -> np.ndarray:
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        Z_arr = np.asarray(Z, dtype=float)
        self._last_z_cache = Z_arr
        self._last_a_cache = int(action)
        Psi = np.hstack([np.ones((len(Z_arr), 1)), Z_arr])
        return Z_arr + Psi @ dyn.theta_trans[int(action)]

    def backward_transition(self, grad_z_next: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        assert self._last_z_cache is not None and self._last_a_cache is not None
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        Z_arr = self._last_z_cache
        a = self._last_a_cache
        Psi = np.hstack([np.ones((len(Z_arr), 1)), Z_arr])
        g_out = np.asarray(grad_z_next, dtype=float)
        self.grad_theta_trans = np.zeros_like(dyn.theta_trans)
        self.grad_theta_trans[a] = Psi.T @ g_out
        grad_z = g_out + g_out @ dyn.theta_trans[a, 1:, :].T
        return grad_z

    def memory_footprint_bytes(self) -> Dict[str, int]:
        """Exact byte & parameter accounting for persistent world model storage."""
        payload = {
            "engrams": [
                {
                    "id": e.engram_id,
                    "x_mean": e.model.x_mean,
                    "W": e.model.W_adapter,
                    "b": e.model.b_adapter,
                    "wc": e.model.w_core,
                    "bc": e.model.b_core,
                    "ww": e.model.w_world,
                    "bw": e.model.b_world,
                    "gate": e.model.core_gate,
                    "sig2": e.sigma2_ref,
                }
                for e in self.memory.engrams
            ],
            "dynamics": {
                eid: {
                    "theta": d.theta_trans,
                    "cov_t": d.cov_trans,
                    "sig_t": d.sigma2_trans,
                    "mask": d.causal_action_mask,
                    "cov_r": d.cov_outcome,
                    "sig_r": d.sigma2_outcome,
                    "costs": d.action_costs,
                    "zmin": d.z_min,
                    "zmax": d.z_max,
                }
                for eid, d in self.dynamics.items()
            },
        }
        minimal_pickle = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        return {
            "num_engrams": self.num_engrams,
            "trainable_parameters": self.num_parameters,
            "trainable_param_raw_float64_bytes": self.num_parameters * 8,
            "minimal_persistent_pickle_bytes": minimal_pickle,
            "raw_observations_stored_bytes": 0,
        }
