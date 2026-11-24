"""Phase 18 / 18B: Integrated Predictive World Model (models/world_model.py).

Composes Phase 17C's verified PersistentWorldMemory (models/memory.py) and
ModularTransferModelV3 (models/transfer_v3.py) WITHOUT modifying them, adding:
1. Action-conditioned Bayesian causal latent transition dynamics with exact
   Type-II ML (Empirical Bayes) posterior covariances Sigma_{a,k} and full KxK
   aleatoric noise covariance Sigma_{eps,a}.
2. Phase 18B Boundary-Aware O(K)-Invariant Degree-2 Transition Dynamics (boundary_aware=True):
   Uses the exact O(K)-invariant degree-2 basis psi(z) = [1, Phi(z)] (15 features,
   matching ModularTransferModelV3's basis) with Type-II ML Empirical Bayes shrinkage
   to capture asymmetric boundary saturation without any oracle boundary knowledge,
   paired with exact analytical Jacobians J_a(z) = I + d(Delta z)/dz.
   (When boundary_aware=False, uses linear psi(z) = [1, z] matching Phase 18 f059132.)
3. Phase 18B Atomic Affine-Orthogonal Basis Synchronization (basis_sync=True):
   Tracks (basis_version, W_basis, x_mean_basis, b_basis) and transforms
   (Theta_a, Sigma_{a,k}, Q_epi, Sigma_{eps,a}, mu_z, Lambda_z, z_min, z_max)
   via the exact invertible polynomial basis map M(Q, c):
       psi(z Q + c) = psi(z) @ M(Q, c)
       Theta'_a = M(Q, c)^{-1} Theta_a Q
       Sigma'_{a,k} = M(Q, c)^{-1} Sigma_{a,k} M(Q, c)^{-T}
       Q'_epi = Q_epi @ Q,   Sigma'_{eps,a} = Q^T Sigma_{eps,a} Q
   guaranteeing machine-precision (< 1e-12) predictive mean & full covariance
   equivalence across arbitrary orthogonal representation updates.
"""
from dataclasses import dataclass
import itertools
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.memory import PersistentWorldMemory
from models.transfer_v3 import ModularTransferModelV3


class BasisDesynchronizationError(RuntimeError):
    """Raised when BayesianDynamicsEngram basis does not match PersistentWorldMemory's adapter."""
    pass


@dataclass
class BayesianDynamicsEngram:
    """Compact Bayesian action-conditioned dynamics + uncertainty state for one world engram."""
    engram_id: int
    n_actions: int
    n_latent: int
    psi_dim: int                   # K+1 (linear) or 1+phi_dim (degree-2 boundary-aware)
    use_quad: bool
    theta_trans: np.ndarray        # (A, psi_dim, K)
    cov_trans: np.ndarray          # (A, K, psi_dim, psi_dim)
    sigma2_trans: np.ndarray       # (A, K): diagonal aleatoric transition noise variance
    causal_action_mask: np.ndarray # (A, K): binary mask of direct causal action effects vs NOOP
    cov_outcome: np.ndarray        # (phi_dim+1, phi_dim+1): exact EB posterior covariance for [1, Phi(z)]
    sigma2_outcome: float          # aleatoric consequence noise variance
    action_costs: np.ndarray       # (A,): learned direct per-action consequence offset
    z_min: np.ndarray              # (K,): empirical latent lower bound
    z_max: np.ndarray              # (K,): empirical latent upper bound
    z_mean: np.ndarray             # (K,): empirical latent mean
    z_cov_inv: np.ndarray          # (K, K): regularized inverse latent covariance
    # Phase 18B basis-synchronization metadata & full rotated covariance state
    basis_version: int
    W_basis: np.ndarray            # (D, K): snapshot of W_adapter
    x_mean_basis: np.ndarray       # (1, D): snapshot of x_mean
    b_basis: np.ndarray            # (1, K): snapshot of b_adapter
    sigma_cov_trans: np.ndarray    # (A, K, K): full rotated aleatoric transition noise covariance
    Q_epi: np.ndarray              # (K, K): cumulative orthogonal map from fit-time epistemic axes to current basis


class CaspianWorldModel:
    """Pure-NumPy uncertainty-aware predictive world model integrating PersistentWorldMemory."""

    def __init__(
        self,
        in_dim: int = 8,
        n_latent: int = 4,
        n_actions: int = 5,
        seed: int = 0,
        alignment_mode: str = "occam_hierarchy",
        boundary_aware: bool = True,
        basis_sync: bool = True,
    ):
        self.in_dim = int(in_dim)
        self.n_latent = int(n_latent)
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self.boundary_aware = bool(boundary_aware)
        self.basis_sync = bool(basis_sync)
        self.pair_indices: List[Tuple[int, int]] = list(
            itertools.combinations_with_replacement(range(self.n_latent), 2)
        )
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

    def _compute_psi(self, Z: np.ndarray, use_quad: bool) -> np.ndarray:
        """Compute transition design matrix [1, Z] (linear) or [1, Z, Z_i*Z_j] (degree-2 O(K)-invariant)."""
        Z_arr = np.asarray(Z, dtype=float)
        if not use_quad:
            return np.hstack([np.ones((len(Z_arr), 1), dtype=float), Z_arr])
        cols = [np.ones(len(Z_arr), dtype=float)]
        for k in range(self.n_latent):
            cols.append(Z_arr[:, k])
        for i, j in self.pair_indices:
            cols.append(Z_arr[:, i] * Z_arr[:, j])
        return np.column_stack(cols)

    def _basis_transform_matrix(self, Q: np.ndarray, c: np.ndarray, use_quad: bool) -> np.ndarray:
        """Exact invertible matrix M(Q, c) satisfying psi(z Q + c) = psi(z) @ M(Q, c)."""
        K = self.n_latent
        c_vec = c.reshape(-1)
        if not use_quad:
            M = np.eye(K + 1, dtype=float)
            M[0, 1:] = c_vec
            M[1:, 1:] = Q
            return M

        P = 1 + K + len(self.pair_indices)
        M = np.zeros((P, P), dtype=float)
        # Constant row 0: 1 -> [1, c_0..c_{K-1}, c_i * c_j]
        M[0, 0] = 1.0
        M[0, 1:1 + K] = c_vec
        for p_idx, (i, j) in enumerate(self.pair_indices):
            M[0, 1 + K + p_idx] = c_vec[i] * c_vec[j]
        # Linear rows 1..K: z_r -> [0, Q_{r,:}, c_i Q_{r,j} + c_j Q_{r,i}]
        for r in range(K):
            M[1 + r, 1:1 + K] = Q[r, :]
            for p_idx, (i, j) in enumerate(self.pair_indices):
                M[1 + r, 1 + K + p_idx] = c_vec[i] * Q[r, j] + c_vec[j] * Q[r, i]
        # Quadratic rows 1+K..P-1: z_r * z_s -> Q_{r,i} Q_{s,j} (+ Q_{s,i} Q_{r,j} if r != s)
        for q_idx, (r, s) in enumerate(self.pair_indices):
            row = 1 + K + q_idx
            for p_idx, (i, j) in enumerate(self.pair_indices):
                col = 1 + K + p_idx
                if r == s:
                    M[row, col] = Q[r, i] * Q[r, j]
                else:
                    M[row, col] = Q[r, i] * Q[s, j] + Q[s, i] * Q[r, j]
        return M

    def assert_basis_synchronized(self, eid: Optional[int] = None, tol: float = 1e-8) -> None:
        """Verify that BayesianDynamicsEngram's coordinate basis matches PersistentWorldMemory's adapter."""
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        err_w = float(np.linalg.norm(dyn.W_basis - e.model.W_adapter, ord="fro"))
        err_m = float(np.linalg.norm(dyn.x_mean_basis - e.model.x_mean, ord="fro"))
        err_b = float(np.linalg.norm(dyn.b_basis - e.model.b_adapter, ord="fro"))
        if max(err_w, err_m, err_b) > tol:
            raise BasisDesynchronizationError(
                f"Engram {e.engram_id} dynamics basis (v{dyn.basis_version}) is desynchronized from "
                f"memory adapter: dW={err_w:.3e}, dMean={err_m:.3e}, db={err_b:.3e}."
            )

    def sync_dynamics_to_basis(
        self,
        eid: int,
        new_x_mean: Optional[np.ndarray] = None,
        new_W_adapter: Optional[np.ndarray] = None,
        new_b_adapter: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        """Atomically transform BayesianDynamicsEngram(eid) into the current (or supplied) memory basis."""
        e = self._get_engram(eid)
        dyn = self.dynamics[eid]
        W_old, xm_old, b_old = dyn.W_basis, dyn.x_mean_basis, dyn.b_basis
        W_new = e.model.W_adapter if new_W_adapter is None else np.asarray(new_W_adapter, dtype=float)
        xm_new = e.model.x_mean if new_x_mean is None else np.asarray(new_x_mean, dtype=float)
        b_new = e.model.b_adapter if new_b_adapter is None else np.asarray(new_b_adapter, dtype=float)

        if (
            np.linalg.norm(W_old - W_new, ord="fro") < 1e-12
            and np.linalg.norm(xm_old - xm_new, ord="fro") < 1e-12
            and np.linalg.norm(b_old - b_new, ord="fro") < 1e-12
        ):
            return True, 0.0

        Q = W_old.T @ W_new
        K = self.n_latent
        orth_err = float(np.linalg.norm(Q.T @ Q - np.eye(K, dtype=float), ord="fro"))
        if orth_err > 1e-3:
            return False, orth_err

        c = (-b_old @ Q) + (xm_old - xm_new) @ W_new + b_new  # (1, K)
        # Exact inverse map: z_old = (z_new - c) @ Q^T = z_new @ Q^T + (-c @ Q^T)
        # So M(Q, c)^{-1} = M(Q^T, -c @ Q^T) — closed-form, no numerical inversion loss!
        M_inv = self._basis_transform_matrix(Q.T, -(c @ Q.T), use_quad=dyn.use_quad)

        theta_new = np.zeros_like(dyn.theta_trans)
        cov_new = np.zeros_like(dyn.cov_trans)
        sig_cov_new = np.zeros_like(dyn.sigma_cov_trans)
        sig2_new = np.zeros_like(dyn.sigma2_trans)

        for a in range(dyn.n_actions):
            theta_new[a] = M_inv @ dyn.theta_trans[a] @ Q
            for k in range(K):
                cov_new[a, k] = M_inv @ dyn.cov_trans[a, k] @ M_inv.T
            sig_cov_new[a] = Q.T @ dyn.sigma_cov_trans[a] @ Q
            sig2_new[a] = np.maximum(np.diag(sig_cov_new[a]), 1e-8)

        corners_old = np.array(
            [[dyn.z_min[k] if ((mask >> k) & 1) == 0 else dyn.z_max[k] for k in range(K)] for mask in range(1 << K)],
            dtype=float,
        )
        corners_new = corners_old @ Q + c

        dyn.theta_trans = theta_new
        dyn.cov_trans = cov_new
        dyn.sigma_cov_trans = sig_cov_new
        dyn.sigma2_trans = sig2_new
        dyn.Q_epi = dyn.Q_epi @ Q
        dyn.z_mean = (dyn.z_mean.reshape(1, -1) @ Q + c).reshape(-1)
        dyn.z_cov_inv = Q.T @ dyn.z_cov_inv @ Q
        dyn.z_min = np.min(corners_new, axis=0)
        dyn.z_max = np.max(corners_new, axis=0)
        dyn.W_basis = W_new.copy()
        dyn.x_mean_basis = xm_new.copy()
        dyn.b_basis = b_new.copy()
        dyn.basis_version += 1
        return True, orth_err

    def encode(self, X: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        squeeze = (X_arr.ndim == 1)
        if squeeze:
            X_arr = X_arr.reshape(1, -1)
        e = self._get_engram(eid)
        Z = (X_arr - e.model.x_mean) @ e.model.W_adapter + e.model.b_adapter
        return Z[0] if squeeze else Z

    def decode_to_obs(self, Z: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        Z_arr = np.asarray(Z, dtype=float)
        squeeze = (Z_arr.ndim == 1)
        if squeeze:
            Z_arr = Z_arr.reshape(1, -1)
        e = self._get_engram(eid)
        X_hat = (Z_arr - e.model.b_adapter) @ e.model.W_adapter.T + e.model.x_mean
        return X_hat[0] if squeeze else X_hat

    def subspace_ood_score(self, X: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
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
        """Closed-form Empirical Bayes (Type-II ML) ridge posterior for y = Psi @ theta + eps."""
        Psi = np.asarray(Psi, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        N, P = Psi.shape
        if N == 0:
            return np.zeros(P, dtype=float), np.eye(P, dtype=float), 1.0, 1.0

        if N > P + 2:
            beta_ols, *_ = np.linalg.lstsq(Psi, y, rcond=None)
            res = y - Psi @ beta_ols
            sigma2_est = float(np.sum(res ** 2) / max(N - P, 1))
        else:
            sigma2_est = 0.05 * max(float(np.var(y)), 1e-4)
        sigma2_est = max(sigma2_est, 1e-5)

        G = Psi.T @ Psi
        eigvals, _ = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, 0.0)
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
        Psi_all: np.ndarray,
        theta_noop: np.ndarray,
        n_actions: int,
        l1_lambda: float = 0.15,
        effect_threshold: float = 0.25,
    ) -> np.ndarray:
        K = Z_t.shape[1]
        mask = np.zeros((n_actions, K), dtype=float)
        dZ_noop_pred = Psi_all @ theta_noop
        dZ_resid = (Z_next - Z_t) - dZ_noop_pred

        for a in range(1, n_actions):
            idx_a = np.where(A_t == a)[0]
            if len(idx_a) < 3:
                continue
            delta_mean = np.mean(dZ_resid[idx_a], axis=0)
            delta_shrunk = np.sign(delta_mean) * np.maximum(0.0, np.abs(delta_mean) - l1_lambda)
            for k in range(K):
                if abs(float(delta_shrunk[k])) >= effect_threshold - l1_lambda and abs(float(delta_mean[k])) >= effect_threshold:
                    mask[a, k] = 1.0
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
        """Learn or adapt a world dynamics + consequence engram from neutral transitions (zero world_id)."""
        X_t = np.asarray(X_t, dtype=float)
        A_t = np.asarray(A_t, dtype=int).reshape(-1)
        X_next = np.asarray(X_next, dtype=float)
        dE_t = np.asarray(dE_t, dtype=float).reshape(-1)

        dE_for_mem = dE_t
        if allow_update_outdated and self.basis_sync and self.memory.engrams:
            for e_cand in self.memory.engrams:
                if e_cand.engram_id in self.dynamics:
                    r_sub = float(np.mean(self.subspace_ood_score(X_next, eid=e_cand.engram_id)))
                    if r_sub < 0.12:
                        dE_for_mem = dE_t + self.dynamics[e_cand.engram_id].action_costs[A_t]
                        break

        eid = self.memory.observe_world(X_next, dE_for_mem, allow_update_outdated=allow_update_outdated)
        engram = self._get_engram(eid)

        Z_t = self.encode(X_t, eid=eid)
        Z_next = self.encode(X_next, eid=eid)
        dZ = Z_next - Z_t
        K = self.n_latent
        A = self.n_actions

        existing_dyn = self.dynamics.get(eid) if allow_update_outdated else None
        use_quad = bool(self.boundary_aware and (len(Z_t) >= 100 or (existing_dyn is not None and existing_dyn.use_quad)))
        Psi_all = self._compute_psi(Z_t, use_quad=use_quad)
        P = Psi_all.shape[1]

        reuse_trans = False
        if existing_dyn is not None and not only_noop_obs and len(Z_t) < 80:
            if self.basis_sync:
                synced, _ = self.sync_dynamics_to_basis(eid)
                if synced:
                    Psi_chk = self._compute_psi(Z_t, use_quad=existing_dyn.use_quad)
                    pred_dz = np.zeros_like(dZ)
                    for a in range(A):
                        idx_a = np.where(A_t == a)[0]
                        if len(idx_a) > 0:
                            pred_dz[idx_a] = Psi_chk[idx_a] @ existing_dyn.theta_trans[a]
                    trans_mse = float(np.mean((pred_dz - dZ) ** 2))
                    if trans_mse < 4.0 * float(np.mean(existing_dyn.sigma2_trans)) + 0.04:
                        reuse_trans = True
            else:
                Psi_chk = self._compute_psi(Z_t, use_quad=existing_dyn.use_quad)
                pred_dz = np.zeros_like(dZ)
                for a in range(A):
                    idx_a = np.where(A_t == a)[0]
                    if len(idx_a) > 0:
                        pred_dz[idx_a] = Psi_chk[idx_a] @ existing_dyn.theta_trans[a]
                trans_mse = float(np.mean((pred_dz - dZ) ** 2))
                if trans_mse < 4.0 * float(np.mean(existing_dyn.sigma2_trans)) + 0.02:
                    reuse_trans = True

        if reuse_trans and existing_dyn is not None:
            theta_trans = existing_dyn.theta_trans.copy()
            cov_trans = existing_dyn.cov_trans.copy()
            sigma_cov_trans = existing_dyn.sigma_cov_trans.copy()
            sigma2_trans = existing_dyn.sigma2_trans.copy()
            Q_epi = existing_dyn.Q_epi.copy()
            causal_mask = existing_dyn.causal_action_mask.copy()
            z_min_use = existing_dyn.z_min.copy()
            z_max_use = existing_dyn.z_max.copy()
            z_mean_use = existing_dyn.z_mean.copy()
            z_cov_inv_use = existing_dyn.z_cov_inv.copy()
            b_ver = existing_dyn.basis_version
            use_quad = existing_dyn.use_quad
            P = existing_dyn.psi_dim
        else:
            theta_trans = np.zeros((A, P, K), dtype=float)
            cov_trans = np.zeros((A, K, P, P), dtype=float)
            sigma2_trans = np.ones((A, K), dtype=float) * 0.05
            sigma_cov_trans = np.zeros((A, K, K), dtype=float)
            Q_epi = np.eye(K, dtype=float)

            theta_pool = np.zeros((P, K), dtype=float)
            cov_pool = np.zeros((K, P, P), dtype=float)
            sig_pool = np.ones(K, dtype=float) * 0.05
            for k in range(K):
                th, cv, sg, _ = self._eb_linear_posterior(Psi_all, dZ[:, k])
                theta_pool[:, k] = th
                cov_pool[k] = cv
                sig_pool[k] = sg

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
                    sigma_cov_trans[a] = np.diag(sig_pool)
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
                    sigma_cov_trans[a] = np.diag(sigma2_trans[a])
                elif src_dyn is not None and src_dyn.psi_dim == P:
                    theta_trans[a] = src_dyn.theta_trans[a].copy()
                    cov_trans[a] = src_dyn.cov_trans[a].copy() * 1.5
                    sigma2_trans[a] = src_dyn.sigma2_trans[a].copy()
                    sigma_cov_trans[a] = src_dyn.sigma_cov_trans[a].copy()
                elif len(idx_a) >= 2:
                    theta_trans[a] = theta_pool.copy()
                    cov_trans[a] = cov_pool.copy() * 2.0
                    sigma2_trans[a] = sig_pool.copy()
                    res_a = dZ[idx_a] - (Psi_all[idx_a] @ theta_pool)
                    theta_trans[a, 0, :] += np.mean(res_a, axis=0)
                    sigma_cov_trans[a] = np.diag(sigma2_trans[a])
                else:
                    theta_trans[a] = theta_pool.copy()
                    cov_trans[a] = cov_pool.copy() * 4.0
                    sigma2_trans[a] = sig_pool.copy() * 2.0
                    sigma_cov_trans[a] = np.diag(sigma2_trans[a])

            causal_mask = self._discover_causal_action_mask(
                Z_t, A_t, Z_next, Psi_all, theta_trans[0], n_actions=A
            )
            Z_both = np.vstack([Z_t, Z_next])
            margin = 0.05 if self.boundary_aware else 0.25
            z_min_use = np.min(Z_both, axis=0) - margin
            z_max_use = np.max(Z_both, axis=0) + margin
            z_mean_use = np.mean(Z_both, axis=0)
            z_cov = np.cov(Z_both.T) + 0.05 * np.eye(K, dtype=float)
            z_cov_inv_use = np.linalg.inv(z_cov)
            b_ver = (existing_dyn.basis_version + 1) if existing_dyn is not None else 0

        Phi_next = engram.model._compute_phi(Z_next)
        Psi_R = np.hstack([np.ones((len(Phi_next), 1)), Phi_next])
        pred_R = engram.model.forward(X_next).reshape(-1)
        res_R = dE_t - pred_R

        if reuse_trans and existing_dyn is not None and self.basis_sync:
            action_costs = existing_dyn.action_costs.copy()
        else:
            action_costs = np.zeros(A, dtype=float)
            for a in range(A):
                idx_a = np.where(A_t == a)[0]
                if len(idx_a) >= 3:
                    action_costs[a] = float(-np.mean(res_R[idx_a]))

        dE_adj = dE_t + action_costs[A_t]
        _, cov_outcome, sig2_outcome, _ = self._eb_linear_posterior(Psi_R, dE_adj)

        self.dynamics[eid] = BayesianDynamicsEngram(
            engram_id=eid,
            n_actions=A,
            n_latent=K,
            psi_dim=P,
            use_quad=use_quad,
            theta_trans=theta_trans,
            cov_trans=cov_trans,
            sigma2_trans=sigma2_trans,
            causal_action_mask=causal_mask,
            cov_outcome=cov_outcome,
            sigma2_outcome=sig2_outcome,
            action_costs=action_costs,
            z_min=z_min_use,
            z_max=z_max_use,
            z_mean=z_mean_use,
            z_cov_inv=z_cov_inv_use,
            basis_version=b_ver,
            W_basis=engram.model.W_adapter.copy(),
            x_mean_basis=engram.model.x_mean.copy(),
            b_basis=engram.model.b_adapter.copy(),
            sigma_cov_trans=sigma_cov_trans,
            Q_epi=Q_epi,
        )
        return eid

    def _consequence_and_grad(
        self, Z: np.ndarray, A: np.ndarray, eid: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        Z_arr = np.asarray(Z, dtype=float)
        A_arr = np.asarray(A, dtype=int).reshape(-1)
        K = self.n_latent

        Phi = e.model._compute_phi(Z_arr)
        w_eff = (e.model.core_gate * e.model.w_core + e.model.w_world).reshape(-1)
        b_eff = float((e.model.core_gate * e.model.b_core + e.model.b_world).item())

        de_mean = Phi @ w_eff + b_eff - dyn.action_costs[A_arr]

        Psi_R = np.hstack([np.ones((len(Phi), 1)), Phi])
        epi_R = np.einsum("ni,ij,nj->n", Psi_R, dyn.cov_outcome, Psi_R)
        de_var = dyn.sigma2_outcome + np.maximum(epi_R, 0.0)

        grad_z = np.tile(w_eff[:K], (len(Z_arr), 1))
        for idx, (i, j) in enumerate(e.model.pair_indices):
            wk = w_eff[K + idx]
            grad_z[:, i] += wk * Z_arr[:, j]
            grad_z[:, j] += wk * Z_arr[:, i]

        return de_mean, de_var, grad_z

    def _transition_jacobians(self, Z_a: np.ndarray, th_a: np.ndarray, use_quad: bool) -> np.ndarray:
        """Exact analytical Jacobians J_n = d(z_next)/d(z_t) of shape (n_a, K, K)."""
        n_a, K = Z_a.shape
        J = np.broadcast_to(np.eye(K, dtype=float) + th_a[1:1 + K, :].T, (n_a, K, K)).copy()
        if use_quad:
            for p_idx, (i, j) in enumerate(self.pair_indices):
                w_p = th_a[1 + K + p_idx, :]  # (K,)
                # d(z_i z_j)/dz_i = z_j, d(z_i z_j)/dz_j = z_i
                J[:, :, i] += Z_a[:, j:j + 1] * w_p.reshape(1, K)
                J[:, :, j] += Z_a[:, i:i + 1] * w_p.reshape(1, K)
        return J

    def predict_step_latent(
        self,
        Z_t: np.ndarray,
        A_t: np.ndarray,
        P_t: Optional[np.ndarray] = None,
        eid: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        e = self._get_engram(eid)
        if self.basis_sync:
            self.assert_basis_synchronized(e.engram_id)
        dyn = self.dynamics[e.engram_id]
        Z_arr = np.asarray(Z_t, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(1, -1)
        A_arr = np.asarray(A_t, dtype=int).reshape(-1)
        N, K = Z_arr.shape

        Psi = self._compute_psi(Z_arr, use_quad=dyn.use_quad)
        z_next = np.zeros((N, K), dtype=float)
        z_var = np.zeros((N, K), dtype=float)
        P_next = np.zeros((N, K, K), dtype=float)
        Qe = dyn.Q_epi

        for a in range(self.n_actions):
            idx = np.where(A_arr == a)[0]
            if len(idx) == 0:
                continue
            Psi_a = Psi[idx]
            th_a = dyn.theta_trans[a]
            dz_a = Psi_a @ th_a
            z_unclamped = Z_arr[idx] + dz_a
            z_clamped = np.clip(z_unclamped, dyn.z_min, dyn.z_max)
            z_next[idx] = z_clamped

            epi_diag = np.zeros((len(idx), K), dtype=float)
            for k in range(K):
                epi_diag[:, k] = np.maximum(np.einsum("ni,ij,nj->n", Psi_a, dyn.cov_trans[a, k], Psi_a), 0.0)
            C_epi = np.einsum("ki,nk,kj->nij", Qe, epi_diag, Qe)
            C_step = C_epi + dyn.sigma_cov_trans[a].reshape(1, K, K)

            if P_t is not None:
                J_a = self._transition_jacobians(Z_arr[idx], th_a, use_quad=dyn.use_quad)
                P_prop = np.einsum("nij,njk,nlk->nil", J_a, P_t[idx], J_a)
                P_next[idx] = P_prop + C_step
            else:
                P_next[idx] = C_step

            for k in range(K):
                z_var[idx, k] = np.maximum(C_step[:, k, k], 1e-10)

        de_mean, de_var_base, grad_z = self._consequence_and_grad(z_next, A_arr, eid=e.engram_id)
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
        e = self._get_engram(eid)
        if self.basis_sync:
            self.assert_basis_synchronized(e.engram_id)
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
        Psi = self._compute_psi(Z_arr, use_quad=dyn.use_quad)
        return Z_arr + Psi @ dyn.theta_trans[int(action)]

    def backward_transition(self, grad_z_next: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        assert self._last_z_cache is not None and self._last_a_cache is not None
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        Z_arr = self._last_z_cache
        a = self._last_a_cache
        Psi = self._compute_psi(Z_arr, use_quad=dyn.use_quad)
        g_out = np.asarray(grad_z_next, dtype=float)
        self.grad_theta_trans = np.zeros_like(dyn.theta_trans)
        self.grad_theta_trans[a] = Psi.T @ g_out
        J = self._transition_jacobians(Z_arr, dyn.theta_trans[a], use_quad=dyn.use_quad)
        grad_z = np.einsum("nj,nji->ni", g_out, J)
        return grad_z

    def to_bytes(self) -> bytes:
        """Lossless minimal persistent serialization preserving synchronized basis metadata."""
        payload = {
            "cfg": (self.in_dim, self.n_latent, self.n_actions, self.seed, self.boundary_aware, self.basis_sync),
            "active_eid": self.memory.active_engram_id,
            "engrams": [
                {
                    "id": e.engram_id,
                    "src_id": e.source_engram_id,
                    "x_mean": e.model.x_mean,
                    "W": e.model.W_adapter,
                    "b": e.model.b_adapter,
                    "wc": e.model.w_core,
                    "bc": e.model.b_core,
                    "ww": e.model.w_world,
                    "bw": e.model.b_world,
                    "gate": e.model.core_gate,
                    "sig2": e.sigma2_ref,
                    "kurt": e.model._source_kurtosis,
                }
                for e in self.memory.engrams
            ],
            "dynamics": self.dynamics,
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CaspianWorldModel":
        from models.memory import WorldEngram
        payload = pickle.loads(data)
        in_d, n_lat, n_act, seed, b_aw, b_sy = payload["cfg"]
        obj = cls(in_dim=in_d, n_latent=n_lat, n_actions=n_act, seed=seed, boundary_aware=b_aw, basis_sync=b_sy)
        for ed in payload["engrams"]:
            m = ModularTransferModelV3(in_d, n_lat, seed=seed)
            m.x_mean = ed["x_mean"].copy()
            m.W_adapter = ed["W"].copy()
            m.b_adapter = ed["b"].copy()
            m.w_core = ed["wc"].copy()
            m.b_core = ed["bc"].copy()
            m.w_world = ed["ww"].copy()
            m.b_world = ed["bw"].copy()
            m.core_gate = float(ed["gate"])
            m._source_kurtosis = float(ed["kurt"])
            m._source_x_mean = m.x_mean.copy()
            m._source_W_adapter = m.W_adapter.copy()
            m._source_b_adapter = m.b_adapter.copy()
            obj.memory.engrams.append(
                WorldEngram(engram_id=ed["id"], model=m, sigma2_ref=float(ed["sig2"]), source_engram_id=ed["src_id"])
            )
        obj.memory.active_engram_id = payload["active_eid"]
        obj.dynamics = payload["dynamics"]
        return obj

    def memory_footprint_bytes(self) -> Dict[str, int]:
        minimal_pickle = len(self.to_bytes())
        return {
            "num_engrams": self.num_engrams,
            "trainable_parameters": self.num_parameters,
            "trainable_param_raw_float64_bytes": self.num_parameters * 8,
            "minimal_persistent_pickle_bytes": minimal_pickle,
            "raw_observations_stored_bytes": 0,
        }
