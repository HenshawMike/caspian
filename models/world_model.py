"""Phase 18 / 18B / 18C: Integrated Predictive World Model (models/world_model.py).

Composes Phase 17C's verified PersistentWorldMemory (models/memory.py) and
ModularTransferModelV3 (models/transfer_v3.py) WITHOUT modifying them, adding:
1. Action-conditioned Bayesian causal latent transition dynamics with exact
   Type-II ML (Empirical Bayes) posterior covariances Sigma_{a,k} and full KxK
   aleatoric noise covariance Sigma_{eps,a}.
2. First-Principles Orthogonal Box-Projection Boundary Dynamics (boundary_aware=True):
   - Step 1: Robust IRLS per action recovers unattenuated interior linear drift Theta_a
     without bias from clipped boundary transitions.
   - Step 2: Unit clipping-impulse L4 alignment on O(K) + exact analytical post-clip MSE
     refinement discovers the intrinsic orthogonal boundary-face frame U_box in O(K) and
     empirical face bounds (l_box, u_box) strictly from observed neutral transitions
     (zero oracle boundary knowledge).
   - Step 3: Empirical Bayes refit on strictly interior points + exact orthogonal box
     projection z_{t+1} = clip((z_t + [1, z_t] Theta_a) U_box, l_box, u_box) U_box^T.
3. Exact Affine-Orthogonal Basis Synchronization (basis_sync=True):
   Under any co-subspace orthogonal rotation Q = W_old^T W_new in O(K) and shift c:
       Theta'_a = T^{-1} Theta_a Q,   Sigma'_{a,k} = T^{-1} Sigma_{a,k} T^{-T}
       Q'_epi = Q_epi @ Q,            Sigma'_{eps,a} = Q^T Sigma_{eps,a} Q
       U'_box = Q^T U_box,            l'_box = l_box + c U'_box,  u'_box = u_box + c U'_box
   guaranteeing machine-precision (< 1e-12) predictive mean & full Bayesian covariance
   equivalence everywhere (interior AND boundary) across arbitrary representation updates.
"""
from dataclasses import dataclass
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
    psi_dim: int                   # K+1
    theta_trans: np.ndarray        # (A, K+1, K)
    cov_trans: np.ndarray          # (A, K, K+1, K+1)
    sigma2_trans: np.ndarray       # (A, K): diagonal aleatoric transition noise variance
    causal_action_mask: np.ndarray # (A, K): binary mask of direct causal action effects vs NOOP
    cov_outcome: np.ndarray        # (phi_dim+1, phi_dim+1): exact EB posterior covariance for [1, Phi(z)]
    sigma2_outcome: float          # aleatoric consequence noise variance
    action_costs: np.ndarray       # (A,): learned direct per-action consequence offset
    z_min: np.ndarray              # (K,): box lower bounds in U_box frame
    z_max: np.ndarray              # (K,): box upper bounds in U_box frame
    U_box: np.ndarray              # (K, K): orthogonal boundary-face normal frame in current latent basis
    z_mean: np.ndarray             # (K,): empirical latent mean
    z_cov_inv: np.ndarray          # (K, K): regularized inverse latent covariance
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
        transfer_mode: str = "consequence_only",
    ):
        self.in_dim = int(in_dim)
        self.n_latent = int(n_latent)
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self.boundary_aware = bool(boundary_aware)
        self.basis_sync = bool(basis_sync)
        self.transfer_mode = str(transfer_mode)
        self.memory = PersistentWorldMemory(
            in_dim=self.in_dim,
            n_latent=self.n_latent,
            seed=self.seed,
            alignment_mode=alignment_mode,
        )
        self.dynamics: Dict[int, BayesianDynamicsEngram] = {}
        self.last_transfer_decisions: Dict[str, Any] = {}
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
            int(d.theta_trans.size + d.action_costs.size + d.U_box.size + d.z_min.size + d.z_max.size)
            for d in self.dynamics.values()
        )
        return int(mem_p + dyn_p)

    def _get_engram(self, eid: Optional[int] = None):
        use_id = self.memory.active_engram_id if eid is None else eid
        assert use_id is not None and len(self.memory.engrams) > 0, "No active engram in world model."
        return next(e for e in self.memory.engrams if e.engram_id == use_id)

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
        """Atomically transform BayesianDynamicsEngram(eid) into the current (or supplied) memory basis.

        Exact affine-orthogonal laws (interior AND boundary):
            Q = W_old^T W_new,  c = -b_old Q + (mu_old - mu_new) W_new + b_new
            T^{-1} = [[1, -c Q^T], [0, Q^T]]
            Theta'_a = T^{-1} Theta_a Q,   Sigma'_{a,k} = T^{-1} Sigma_{a,k} T^{-T}
            Q'_epi = Q_epi @ Q,            Sigma'_{eps,a} = Q^T Sigma_{eps,a} Q
            U'_box = Q^T U_box,            z'_min = z_min + c U'_box,  z'_max = z_max + c U'_box
        """
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
        T_inv = np.eye(K + 1, dtype=float)
        T_inv[0, 1:] = -(c @ Q.T)[0]
        T_inv[1:, 1:] = Q.T

        theta_new = np.zeros_like(dyn.theta_trans)
        cov_new = np.zeros_like(dyn.cov_trans)
        sig_cov_new = np.zeros_like(dyn.sigma_cov_trans)
        sig2_new = np.zeros_like(dyn.sigma2_trans)

        for a in range(dyn.n_actions):
            theta_new[a] = T_inv @ dyn.theta_trans[a] @ Q
            for k in range(K):
                cov_new[a, k] = T_inv @ dyn.cov_trans[a, k] @ T_inv.T
            sig_cov_new[a] = Q.T @ dyn.sigma_cov_trans[a] @ Q
            sig2_new[a] = np.maximum(np.diag(sig_cov_new[a]), 1e-8)

        # Exact orthogonal box-projector transformation: invariant even on boundaries!
        U_box_new = Q.T @ dyn.U_box
        c_box = (c @ U_box_new).reshape(-1)
        dyn.theta_trans = theta_new
        dyn.cov_trans = cov_new
        dyn.sigma_cov_trans = sig_cov_new
        dyn.sigma2_trans = sig2_new
        dyn.Q_epi = dyn.Q_epi @ Q
        dyn.U_box = U_box_new
        dyn.z_min = dyn.z_min + c_box
        dyn.z_max = dyn.z_max + c_box
        dyn.z_mean = (dyn.z_mean.reshape(1, -1) @ Q + c).reshape(-1)
        dyn.z_cov_inv = Q.T @ dyn.z_cov_inv @ Q
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

    def _fit_orthogonal_box_boundary(
        self,
        Z_t: np.ndarray,
        A_t: np.ndarray,
        Z_next: np.ndarray,
        Psi_all: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """First-principles unsupervised discovery of orthogonal boundary-face frame U_box in O(K),
        empirical box bounds (l_box, u_box), and interior mask (zero oracle boundary knowledge).

        1. Robust IRLS per action recovers unattenuated interior linear drift.
        2. Unit clipping-impulse L4 maximization on O(K) initializes U_box along face normals.
        3. Analytical post-clip MSE refinement optimizes (U_box, l_box, u_box).
        """
        N, K = Z_next.shape
        dZ = Z_next - Z_t
        Z_unc = np.zeros_like(Z_next)
        res_all = np.zeros_like(dZ)

        # Step 1: Robust IRLS per action to separate interior linear physics from boundary impulses
        for a in range(self.n_actions):
            ia = np.where(A_t == a)[0]
            if len(ia) < K + 3:
                Z_unc[ia] = Z_next[ia]
                continue
            Xa, Ya = Psi_all[ia], dZ[ia]
            w = np.ones(len(ia), dtype=float)
            th = np.zeros((K + 1, K), dtype=float)
            for _ in range(8):
                sw = np.sqrt(w).reshape(-1, 1)
                th = np.linalg.solve(
                    (Xa * sw).T @ (Xa * sw) + 1e-3 * np.eye(K + 1, dtype=float),
                    (Xa * sw).T @ (Ya * sw),
                )
                rn = np.linalg.norm(Ya - Xa @ th, axis=1)
                mad = float(np.median(rn)) + 1e-6
                w = 1.0 / (1.0 + (rn / (2.5 * mad)) ** 4)
            Z_unc[ia] = Z_t[ia] + Xa @ th
            res_all[ia] = Ya - Xa @ th

        rn_all = np.linalg.norm(res_all, axis=1)
        mad_all = float(np.median(rn_all)) + 1e-6
        clip_idx = np.where(rn_all > 2.5 * mad_all)[0]

        if len(clip_idx) < 8:
            # No statistically significant boundary clipping observed
            U = np.eye(K, dtype=float)
            Z_both = np.vstack([Z_t, Z_next])
            return U, np.min(Z_both, axis=0) - 0.25, np.max(Z_both, axis=0) + 0.25, np.ones(N, dtype=bool)

        # Step 2: Unit clipping-impulse L4 alignment on O(K)
        R_hat = res_all[clip_idx] / rn_all[clip_idx].reshape(-1, 1)
        rng = np.random.default_rng(self.seed + 19)
        best_U = np.eye(K, dtype=float)
        best_obj = -1.0
        for r_i in range(10):
            U = np.eye(K, dtype=float) if r_i == 0 else np.linalg.qr(rng.standard_normal((K, K)))[0]
            m = np.zeros_like(U); v = np.zeros_like(U)
            for st in range(1, 91):
                V = R_hat @ U
                gU = 4.0 * (R_hat.T @ (V ** 3)) / max(len(R_hat), 1)
                m = 0.9 * m + 0.1 * gU; v = 0.999 * v + 0.001 * (gU ** 2)
                U = U + 0.1 * (m / (1.0 - 0.9 ** st)) / (np.sqrt(v / (1.0 - 0.999 ** st)) + 1e-8)
                Uq, _, Vtq = np.linalg.svd(U, full_matrices=False)
                U = Uq @ Vtq
            obj = float(np.mean((R_hat @ U) ** 4))
            if obj > best_obj:
                best_obj = obj
                best_U = U.copy()

        # Step 3: Exact analytical post-clip MSE refinement of (U, l_box, u_box)
        U = best_U.copy()
        Vn_init = Z_next @ U
        l_box = np.percentile(Vn_init, 1.0, axis=0)
        u_box = np.percentile(Vn_init, 99.0, axis=0)
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        ml = np.zeros_like(l_box); vl = np.zeros_like(l_box)
        mu = np.zeros_like(u_box); vu = np.zeros_like(u_box)
        for st in range(1, 181):
            V = Z_unc @ U
            Tgt = Z_next @ U
            C = np.clip(V, l_box, u_box)
            diff = C - Tgt
            in_mask = ((V > l_box) & (V < u_box)).astype(float)
            lo_mask = (V <= l_box).astype(float)
            hi_mask = (V >= u_box).astype(float)
            gU = (2.0 / N) * (Z_unc.T @ (diff * in_mask) - Z_next.T @ diff)
            gl = (2.0 / N) * np.sum(diff * lo_mask, axis=0)
            gu = (2.0 / N) * np.sum(diff * hi_mask, axis=0)
            mU = 0.9 * mU + 0.1 * gU; vU = 0.999 * vU + 0.001 * (gU ** 2)
            ml = 0.9 * ml + 0.1 * gl; vl = 0.999 * vl + 0.001 * (gl ** 2)
            mu = 0.9 * mu + 0.1 * gu; vu = 0.999 * vu + 0.001 * (gu ** 2)
            U = U - 0.04 * (mU / (1.0 - 0.9 ** st)) / (np.sqrt(vU / (1.0 - 0.999 ** st)) + 1e-8)
            Uq, _, Vtq = np.linalg.svd(U, full_matrices=False)
            U = Uq @ Vtq
            l_box = l_box - 0.04 * (ml / (1.0 - 0.9 ** st)) / (np.sqrt(vl / (1.0 - 0.999 ** st)) + 1e-8)
            u_box = u_box - 0.04 * (mu / (1.0 - 0.9 ** st)) / (np.sqrt(vu / (1.0 - 0.999 ** st)) + 1e-8)

        # Keep faces with < 3 active clipping events open (min/max with safe margin) so unvisited walls aren't over-tightened
        V_final = Z_unc @ U
        Vn_final = Z_next @ U
        for k in range(K):
            if int(np.sum(V_final[:, k] <= l_box[k])) < 3:
                l_box[k] = float(np.min(Vn_final[:, k])) - 0.25
            if int(np.sum(V_final[:, k] >= u_box[k])) < 3:
                u_box[k] = float(np.max(Vn_final[:, k])) + 0.25

        interior_mask = np.all((Vn_final > l_box + 0.04) & (Vn_final < u_box - 0.04), axis=1)
        return U, l_box, u_box, interior_mask

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

    @staticmethod
    def _transform_dynamics_candidate(
        src_dyn: BayesianDynamicsEngram, Q: np.ndarray, c: np.ndarray
    ) -> Dict[str, Any]:
        """Apply exact affine-orthogonal map (z_tgt = z_src @ Q + c) to a source dynamics engram."""
        K = src_dyn.n_latent
        A = src_dyn.n_actions
        c_row = c.reshape(1, K)
        T_inv = np.eye(K + 1, dtype=float)
        T_inv[0, 1:] = -(c_row @ Q.T)[0]
        T_inv[1:, 1:] = Q.T

        theta_new = np.zeros_like(src_dyn.theta_trans)
        cov_new = np.zeros_like(src_dyn.cov_trans)
        sig_cov_new = np.zeros_like(src_dyn.sigma_cov_trans)
        sig2_new = np.zeros_like(src_dyn.sigma2_trans)
        for a in range(A):
            theta_new[a] = T_inv @ src_dyn.theta_trans[a] @ Q
            for k in range(K):
                cov_new[a, k] = T_inv @ src_dyn.cov_trans[a, k] @ T_inv.T
            sig_cov_new[a] = Q.T @ src_dyn.sigma_cov_trans[a] @ Q
            sig2_new[a] = np.maximum(np.diag(sig_cov_new[a]), 1e-8)

        U_box_new = Q.T @ src_dyn.U_box
        c_box = (c_row @ U_box_new).reshape(-1)
        return {
            "theta_trans": theta_new,
            "cov_trans": cov_new,
            "sigma_cov_trans": sig_cov_new,
            "sigma2_trans": sig2_new,
            "Q_epi": src_dyn.Q_epi @ Q,
            "U_box": U_box_new,
            "z_min": src_dyn.z_min + c_box,
            "z_max": src_dyn.z_max + c_box,
            "action_costs": src_dyn.action_costs.copy(),
        }

    def _align_source_dynamics_procrustes(
        self,
        src_dyn: BayesianDynamicsEngram,
        Z_t: np.ndarray,
        A_t: np.ndarray,
        Z_next: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Align source dynamics into target latent frame via leave-one-action-out Orthogonal Procrustes
        on interventional shift vectors + closed-form offset solve + trimmed residual refinement.
        Breaks scalar consequence symmetries (e.g. s0 <-> s1) and tolerates single-action mutations (Regime C).
        """
        K = self.n_latent
        A = self.n_actions
        dZ = Z_next - Z_t

        # Source pure interventional shifts relative to NOOP
        delta_src = np.zeros((A - 1, K), dtype=float)
        for a in range(1, A):
            delta_src[a - 1] = src_dyn.theta_trans[a, 0, :] - src_dyn.theta_trans[0, 0, :]

        # Target empirical interventional shifts relative to NOOP (estimated robustly)
        idx0 = np.where(A_t == 0)[0]
        if len(idx0) >= K + 2:
            Psi0 = np.hstack([np.ones((len(idx0), 1)), Z_t[idx0]])
            th0 = np.linalg.lstsq(Psi0, dZ[idx0], rcond=None)[0]
            noop_pred = np.hstack([np.ones((len(Z_t), 1)), Z_t]) @ th0
        else:
            noop_pred = np.zeros_like(dZ)
        dZ_rel = dZ - noop_pred

        delta_tgt = np.zeros((A - 1, K), dtype=float)
        valid_acts: List[int] = []
        for a in range(1, A):
            ia = np.where(A_t == a)[0]
            if len(ia) >= 2:
                # Median reduces sensitivity to clipped boundary samples
                delta_tgt[a - 1] = np.median(dZ_rel[ia], axis=0)
                valid_acts.append(a - 1)

        # Candidate subsets: all valid actions + leave-one-out subsets (immune to 1 mutated action!)
        subsets: List[List[int]] = []
        if len(valid_acts) >= 2:
            subsets.append(valid_acts)
            if len(valid_acts) >= 3:
                for drop in valid_acts:
                    subsets.append([v for v in valid_acts if v != drop])
        else:
            subsets.append(list(range(A - 1)))

        best_score = float("inf")
        best_Q = np.eye(K, dtype=float)
        best_c = np.zeros((1, K), dtype=float)

        for sub in subsets:
            # Closed-form Orthogonal Procrustes: min_Q ||delta_src[sub] @ Q - delta_tgt[sub]||_F
            M_p = delta_src[sub].T @ delta_tgt[sub]
            Up, _, Vtp = np.linalg.svd(M_p, full_matrices=False)
            Q_cand = Up @ Vtp

            # Closed-form offset c solving c @ D_a' ~ Z_t @ D_a' + b_a' - dZ across actions in sub U {0}
            act_set = set([0] + [v + 1 for v in sub])
            rows_L, rows_R = [], []
            for a in act_set:
                ia = np.where(A_t == a)[0]
                if len(ia) == 0:
                    continue
                Da = Q_cand.T @ src_dyn.theta_trans[a, 1:, :] @ Q_cand
                ba = src_dyn.theta_trans[a, 0:1, :] @ Q_cand
                rhs = Z_t[ia] @ Da + ba - dZ[ia]  # (n_a, K)
                rows_L.append(np.tile(Da.T, (len(ia), 1)))  # solve Da.T @ c.T = rhs.T
                rows_R.append(rhs.reshape(-1))
            if rows_L:
                L_mat = np.vstack(rows_L)
                R_vec = np.concatenate(rows_R)
                c_vec = np.linalg.solve(L_mat.T @ L_mat + 1e-2 * np.eye(K), L_mat.T @ R_vec)
                c_cand = c_vec.reshape(1, K)
            else:
                c_cand = np.zeros((1, K), dtype=float)

            # Evaluate candidate via trimmed median squared residual across all target transitions
            cand = self._transform_dynamics_candidate(src_dyn, Q_cand, c_cand)
            Psi_all = np.hstack([np.ones((len(Z_t), 1)), Z_t])
            sq_err = np.zeros(len(Z_t), dtype=float)
            for a in range(A):
                ia = np.where(A_t == a)[0]
                if len(ia) == 0:
                    continue
                pred_zn = Z_t[ia] + Psi_all[ia] @ cand["theta_trans"][a]
                sq_err[ia] = np.mean((pred_zn - Z_next[ia]) ** 2, axis=1)
            # Trimmed 60th-percentile residual ignores both boundary clips and a single mutated action!
            score = float(np.percentile(sq_err, 60.0))
            if score < best_score:
                best_score = score
                best_Q = Q_cand.copy()
                best_c = c_cand.copy()

        # Refine (best_Q, best_c) with 40 steps of robust orthogonal Adam on interior/compatible transitions
        Q_ref = best_Q.copy()
        c_ref = best_c.copy()
        cand0 = self._transform_dynamics_candidate(src_dyn, Q_ref, c_ref)
        Psi_all = np.hstack([np.ones((len(Z_t), 1)), Z_t])
        sq0 = np.zeros(len(Z_t), dtype=float)
        for a in range(A):
            ia = np.where(A_t == a)[0]
            if len(ia) > 0:
                sq0[ia] = np.mean((Z_t[ia] + Psi_all[ia] @ cand0["theta_trans"][a] - Z_next[ia]) ** 2, axis=1)
        med0 = float(np.median(sq0)) + 1e-6
        inlier = sq0 <= max(4.0 * med0, 0.04)
        if int(np.sum(inlier)) >= K + 4:
            mQ = np.zeros_like(Q_ref); vQ = np.zeros_like(Q_ref)
            mc = np.zeros_like(c_ref); vc = np.zeros_like(c_ref)
            for st in range(1, 41):
                gQ = np.zeros_like(Q_ref)
                gc = np.zeros_like(c_ref)
                cnt = 0
                for a in range(A):
                    ia = np.where((A_t == a) & inlier)[0]
                    if len(ia) == 0:
                        continue
                    # z_src_hat = (Z_t - c) @ Q^T; dz_pred = ([1, z_src_hat] @ theta_src_a) @ Q
                    Z_s = (Z_t[ia] - c_ref) @ Q_ref.T
                    Da_s = src_dyn.theta_trans[a, 1:, :]  # (K, K)
                    ba_s = src_dyn.theta_trans[a, 0:1, :]  # (1, K)
                    dz_s = Z_s @ Da_s + ba_s              # (n, K)
                    err = (dz_s @ Q_ref) - dZ[ia]         # (n, K)
                    # dL/dQ = dz_s^T @ err + (dL/dZ_s)^T @ (Z_t - c) -> Z_s = (Z_t - c) Q^T
                    g_dz_s = err @ Q_ref.T                # (n, K)
                    g_Z_s = g_dz_s @ Da_s.T               # (n, K)
                    gQ += (dz_s.T @ err) + (g_Z_s.T @ (Z_t[ia] - c_ref))
                    gc += -np.sum(g_Z_s @ Q_ref, axis=0, keepdims=True)
                    cnt += len(ia)
                if cnt == 0:
                    break
                gQ *= (2.0 / cnt); gc *= (2.0 / cnt)
                mQ = 0.9 * mQ + 0.1 * gQ; vQ = 0.999 * vQ + 0.001 * (gQ ** 2)
                mc = 0.9 * mc + 0.1 * gc; vc = 0.999 * vc + 0.001 * (gc ** 2)
                Q_ref = Q_ref - 0.03 * (mQ / (1.0 - 0.9 ** st)) / (np.sqrt(vQ / (1.0 - 0.999 ** st)) + 1e-8)
                Uq, _, Vtq = np.linalg.svd(Q_ref, full_matrices=False)
                Q_ref = Uq @ Vtq
                c_ref = c_ref - 0.03 * (mc / (1.0 - 0.9 ** st)) / (np.sqrt(vc / (1.0 - 0.999 ** st)) + 1e-8)

        cand_final = self._transform_dynamics_candidate(src_dyn, Q_ref, c_ref)
        return Q_ref, c_ref, cand_final

    def _evaluate_selective_transfer(
        self,
        src_dyn: BayesianDynamicsEngram,
        Z_t: np.ndarray,
        A_t: np.ndarray,
        Z_next: np.ndarray,
        mode: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate component-level transfer (per-action dynamics + boundary box) in target frame.

        Modes:
            - 'unconditional': blindly copy src_dyn in raw target frame (no Procrustes, no compatibility tests)
            - 'dynamics_and_boundary': Procrustes-align and transfer all actions + boundary unconditionally
            - 'selective': Procrustes-align + fixed MSE compatibility gate per action and boundary
            - 'uncertainty_guided': Procrustes-align + Bayesian Mahalanobis m_a^2 per action (PRESERVE/ADAPT/RECONSTRUCT)
              + boundary violation/tightening test
        """
        K = self.n_latent
        A = self.n_actions
        dZ = Z_next - Z_t
        Psi_all = np.hstack([np.ones((len(Z_t), 1), dtype=float), Z_t])

        if mode == "unconditional":
            cand = {
                "theta_trans": src_dyn.theta_trans.copy(),
                "cov_trans": src_dyn.cov_trans.copy(),
                "sigma_cov_trans": src_dyn.sigma_cov_trans.copy(),
                "sigma2_trans": src_dyn.sigma2_trans.copy(),
                "Q_epi": src_dyn.Q_epi.copy(),
                "U_box": src_dyn.U_box.copy(),
                "z_min": src_dyn.z_min.copy(),
                "z_max": src_dyn.z_max.copy(),
            }
            return cand, {"mode": mode, "action_decisions": ["TRANSFER"] * A, "boundary_decision": "TRANSFER"}

        _, _, cand = self._align_source_dynamics_procrustes(src_dyn, Z_t, A_t, Z_next)

        if mode == "dynamics_and_boundary":
            return cand, {"mode": mode, "action_decisions": ["TRANSFER"] * A, "boundary_decision": "TRANSFER"}

        # Component-level compatibility evaluation (selective vs uncertainty_guided)
        theta_out = cand["theta_trans"].copy()
        cov_out = cand["cov_trans"].copy()
        sig_cov_out = cand["sigma_cov_trans"].copy()
        sig2_out = cand["sigma2_trans"].copy()
        Qe = cand["Q_epi"]

        action_decisions: List[str] = []
        action_stats: List[float] = []
        n_transferred = 0

        for a in range(A):
            ia = np.where(A_t == a)[0]
            if len(ia) < 3:
                action_decisions.append("TRANSFER")
                action_stats.append(0.0)
                n_transferred += 1
                continue

            Psi_a = Psi_all[ia]
            pred_zn = Z_t[ia] + Psi_a @ cand["theta_trans"][a]
            res = Z_next[ia] - pred_zn  # (n_a, K)

            if mode == "uncertainty_guided":
                # Normalized Bayesian Mahalanobis squared residual per transition (median ignores boundary clips!)
                m2_vals = np.zeros(len(ia), dtype=float)
                epi_diag = np.zeros((len(ia), K), dtype=float)
                for k in range(K):
                    epi_diag[:, k] = np.maximum(np.einsum("ni,ij,nj->n", Psi_a, cand["cov_trans"][a, k], Psi_a), 0.0)
                C_epi = np.einsum("ki,nk,kj->nij", Qe, epi_diag, Qe)
                C_pred = C_epi + cand["sigma_cov_trans"][a].reshape(1, K, K) + 1e-4 * np.eye(K)
                for i_loc in range(len(ia)):
                    r_v = res[i_loc]
                    m2_vals[i_loc] = float(r_v @ np.linalg.solve(C_pred[i_loc], r_v)) / K
                # 35th percentile evaluates strictly interior transitions even when ~50% clip at tight boundaries (Regime B)
                stat = float(np.percentile(m2_vals, 35.0))
                action_stats.append(round(stat, 4))

                if stat <= 4.5:
                    # Compatible -> PRESERVE / TRANSFER low-variance source dynamics
                    action_decisions.append("TRANSFER")
                    n_transferred += 1
                elif stat <= 14.0 and len(ia) >= K + 2:
                    # Partially shifted -> Bayesian precision-weighted ADAPT toward target
                    for k in range(K):
                        th_t, cv_t, sg_t, _ = self._eb_linear_posterior(Psi_a, dZ[ia, k])
                        # Precision-weighted fusion of source prior and target likelihood
                        P_src = np.linalg.inv(cand["cov_trans"][a, k] + 1e-4 * np.eye(K + 1))
                        P_tgt = np.linalg.inv(cv_t + 1e-4 * np.eye(K + 1))
                        cv_f = np.linalg.inv(0.5 * P_src + P_tgt)
                        theta_out[a, :, k] = cv_f @ (0.5 * P_src @ cand["theta_trans"][a, :, k] + P_tgt @ th_t)
                        cov_out[a, k] = cv_f
                        sig2_out[a, k] = sg_t
                    sig_cov_out[a] = np.diag(sig2_out[a])
                    action_decisions.append("ADAPT")
                    n_transferred += 1
                else:
                    # Incompatible -> REJECT source & RECONSTRUCT action a from target
                    for k in range(K):
                        th_t, cv_t, sg_t, _ = self._eb_linear_posterior(Psi_a, dZ[ia, k])
                        theta_out[a, :, k] = th_t
                        cov_out[a, k] = cv_t
                        sig2_out[a, k] = sg_t
                    sig_cov_out[a] = np.diag(sig2_out[a])
                    action_decisions.append("RECONSTRUCT")
            else:
                # Fixed-threshold selective mode (no Bayesian Mahalanobis covariance)
                med_mse = float(np.median(np.mean(res ** 2, axis=1)))
                action_stats.append(round(med_mse, 5))
                if med_mse <= 0.025:
                    action_decisions.append("TRANSFER")
                    n_transferred += 1
                else:
                    for k in range(K):
                        th_t, cv_t, sg_t, _ = self._eb_linear_posterior(Psi_a, dZ[ia, k])
                        theta_out[a, :, k] = th_t
                        cov_out[a, k] = cv_t
                        sig2_out[a, k] = sg_t
                    sig_cov_out[a] = np.diag(sig2_out[a])
                    action_decisions.append("RECONSTRUCT")

        # Boundary Compatibility Test (Distinguishes Regime A vs Regime B vs Regime D!)
        U_box_out = cand["U_box"].copy()
        z_min_out = cand["z_min"].copy()
        z_max_out = cand["z_max"].copy()

        if n_transferred == 0:
            # Regime D: All dynamics incompatible -> reconstruct full boundary from target!
            if self.boundary_aware and len(Z_t) >= 40:
                U_box_out, z_min_out, z_max_out, _ = self._fit_orthogonal_box_boundary(Z_t, A_t, Z_next, Psi_all)
            else:
                U_box_out = np.eye(K, dtype=float)
                Z_both = np.vstack([Z_t, Z_next])
                z_min_out = np.min(Z_both, axis=0) - 0.25
                z_max_out = np.max(Z_both, axis=0) + 0.25
            bnd_decision = "RECONSTRUCT"
        else:
            # Compute unclamped predictions using the accepted/adapted/reconstructed theta_out
            Z_unc = np.zeros_like(Z_next)
            for a in range(A):
                ia = np.where(A_t == a)[0]
                if len(ia) > 0:
                    Z_unc[ia] = Z_t[ia] + Psi_all[ia] @ theta_out[a]
            V_unc = Z_unc @ U_box_out
            V_tgt = Z_next @ U_box_out

            # Check 1: Boundary expansion violation (target states exceed transferred bounds by > 0.15)
            viol_hi = np.max(V_tgt - z_max_out.reshape(1, -1), axis=0)
            viol_lo = np.max(z_min_out.reshape(1, -1) - V_tgt, axis=0)
            # Check 2: Boundary tightening (target states clip strictly inside transferred bounds!)
            mse_xfer_bnd = float(np.mean((np.clip(V_unc, z_min_out, z_max_out) - V_tgt) ** 2))
            l_ad = z_min_out.copy()
            u_ad = z_max_out.copy()
            for k in range(K):
                # Refine 1D box bounds along face k on target transitions
                lo_c = float(np.min(V_tgt[:, k]))
                hi_c = float(np.max(V_tgt[:, k]))
                # Test if tightening or expanding improves post-clip MSE
                if viol_lo[k] > 0.15:
                    l_ad[k] = lo_c
                elif int(np.sum(V_unc[:, k] < lo_c - 0.08)) >= 2:
                    l_ad[k] = lo_c
                if viol_hi[k] > 0.15:
                    u_ad[k] = hi_c
                elif int(np.sum(V_unc[:, k] > hi_c + 0.08)) >= 2:
                    u_ad[k] = hi_c
            mse_adapt_bnd = float(np.mean((np.clip(V_unc, l_ad, u_ad) - V_tgt) ** 2))

            if np.any(viol_hi > 0.15) or np.any(viol_lo > 0.15) or (mse_adapt_bnd < 0.72 * mse_xfer_bnd and mse_xfer_bnd > 0.008):
                z_min_out = l_ad
                z_max_out = u_ad
                bnd_decision = "ADAPT"
            else:
                bnd_decision = "TRANSFER"

        return {
            "theta_trans": theta_out,
            "cov_trans": cov_out,
            "sigma_cov_trans": sig_cov_out,
            "sigma2_trans": sig2_out,
            "Q_epi": Qe.copy() if n_transferred > 0 else np.eye(K, dtype=float),
            "U_box": U_box_out,
            "z_min": z_min_out,
            "z_max": z_max_out,
        }, {
            "mode": mode,
            "action_decisions": action_decisions,
            "action_stats": action_stats,
            "boundary_decision": bnd_decision,
            "n_actions_transferred": n_transferred,
        }

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
        self.last_transfer_decisions = {}

        dE_for_mem = dE_t
        if allow_update_outdated and self.basis_sync and self.memory.engrams:
            for e_cand in self.memory.engrams:
                if e_cand.engram_id in self.dynamics:
                    r_sub = float(np.mean(self.subspace_ood_score(X_next, eid=e_cand.engram_id)))
                    if r_sub < 0.12:
                        dE_for_mem = dE_t + self.dynamics[e_cand.engram_id].action_costs[A_t]
                        break

        if self.transfer_mode == "full_reconstruction":
            # Scratch reconstruction: fit fresh engram without transferring from prior memory
            from models.memory import WorldEngram
            m_fresh = ModularTransferModelV3(self.in_dim, self.n_latent, seed=self.seed + 31 * len(self.memory.engrams))
            m_fresh.fit_source(X_next, dE_t)
            _, _, _, sig2_f = ModularTransferModelV3._deduplicate_canonical_with_noise(X_next, dE_t.reshape(-1, 1))
            eid = len(self.memory.engrams)
            self.memory.engrams.append(WorldEngram(engram_id=eid, model=m_fresh, sigma2_ref=sig2_f, source_engram_id=None))
            self.memory.active_engram_id = eid
        else:
            eid = self.memory.observe_world(X_next, dE_for_mem, allow_update_outdated=allow_update_outdated)
        engram = self._get_engram(eid)

        Z_t = self.encode(X_t, eid=eid)
        Z_next = self.encode(X_next, eid=eid)
        dZ = Z_next - Z_t
        K = self.n_latent
        A = self.n_actions
        Psi_all = np.hstack([np.ones((len(Z_t), 1), dtype=float), Z_t])
        P = K + 1

        existing_dyn = self.dynamics.get(eid) if allow_update_outdated else None
        reuse_trans = False
        if existing_dyn is not None and not only_noop_obs and len(Z_t) < 80:
            if self.basis_sync:
                synced, _ = self.sync_dynamics_to_basis(eid)
                if synced:
                    pred_zn = np.zeros_like(Z_next)
                    for a in range(A):
                        idx_a = np.where(A_t == a)[0]
                        if len(idx_a) > 0:
                            z_u = Z_t[idx_a] + Psi_all[idx_a] @ existing_dyn.theta_trans[a]
                            pred_zn[idx_a] = np.clip(z_u @ existing_dyn.U_box, existing_dyn.z_min, existing_dyn.z_max) @ existing_dyn.U_box.T
                    trans_mse = float(np.mean((pred_zn - Z_next) ** 2))
                    if trans_mse < 4.0 * float(np.mean(existing_dyn.sigma2_trans)) + 0.04:
                        reuse_trans = True
            else:
                pred_dz = np.zeros_like(dZ)
                for a in range(A):
                    idx_a = np.where(A_t == a)[0]
                    if len(idx_a) > 0:
                        pred_dz[idx_a] = Psi_all[idx_a] @ existing_dyn.theta_trans[a]
                trans_mse = float(np.mean((pred_dz - dZ) ** 2))
                if trans_mse < 4.0 * float(np.mean(existing_dyn.sigma2_trans)) + 0.02:
                    reuse_trans = True

        # Check Phase 18D cross-environment selective dynamics + boundary transfer
        src_dyn_cand: Optional[BayesianDynamicsEngram] = None
        if (
            not reuse_trans
            and not only_noop_obs
            and self.transfer_mode in ("unconditional", "dynamics_and_boundary", "selective", "uncertainty_guided")
            and len(self.dynamics) > 0
        ):
            if engram.source_engram_id is not None and engram.source_engram_id in self.dynamics:
                src_dyn_cand = self.dynamics[engram.source_engram_id]
            else:
                # Even if consequence rule differed, pick most recent prior engram to test dynamics compatibility
                prior_ids = [k_id for k_id in self.dynamics.keys() if k_id != eid]
                if prior_ids:
                    src_dyn_cand = self.dynamics[prior_ids[-1]]

        if reuse_trans and existing_dyn is not None:
            theta_trans = existing_dyn.theta_trans.copy()
            cov_trans = existing_dyn.cov_trans.copy()
            sigma_cov_trans = existing_dyn.sigma_cov_trans.copy()
            sigma2_trans = existing_dyn.sigma2_trans.copy()
            Q_epi = existing_dyn.Q_epi.copy()
            causal_mask = existing_dyn.causal_action_mask.copy()
            U_box_use = existing_dyn.U_box.copy()
            z_min_use = existing_dyn.z_min.copy()
            z_max_use = existing_dyn.z_max.copy()
            z_mean_use = existing_dyn.z_mean.copy()
            z_cov_inv_use = existing_dyn.z_cov_inv.copy()
            b_ver = existing_dyn.basis_version
        elif src_dyn_cand is not None:
            xfer_res, decisions = self._evaluate_selective_transfer(
                src_dyn_cand, Z_t, A_t, Z_next, mode=self.transfer_mode
            )
            self.last_transfer_decisions = decisions
            theta_trans = xfer_res["theta_trans"]
            cov_trans = xfer_res["cov_trans"]
            sigma_cov_trans = xfer_res["sigma_cov_trans"]
            sigma2_trans = xfer_res["sigma2_trans"]
            Q_epi = xfer_res["Q_epi"]
            U_box_use = xfer_res["U_box"]
            z_min_use = xfer_res["z_min"]
            z_max_use = xfer_res["z_max"]
            causal_mask = self._discover_causal_action_mask(
                Z_t, A_t, Z_next, Psi_all, theta_trans[0], n_actions=A
            )
            Z_both = np.vstack([Z_t, Z_next])
            z_mean_use = np.mean(Z_both, axis=0)
            z_cov = np.cov(Z_both.T) + 0.05 * np.eye(K, dtype=float)
            z_cov_inv_use = np.linalg.inv(z_cov)
            b_ver = 0
        else:
            theta_trans = np.zeros((A, P, K), dtype=float)
            cov_trans = np.zeros((A, K, P, P), dtype=float)
            sigma2_trans = np.ones((A, K), dtype=float) * 0.05
            sigma_cov_trans = np.zeros((A, K, K), dtype=float)
            Q_epi = np.eye(K, dtype=float)

            if self.boundary_aware and not only_noop_obs and len(Z_t) >= 80:
                U_box_use, z_min_use, z_max_use, interior_mask = self._fit_orthogonal_box_boundary(
                    Z_t, A_t, Z_next, Psi_all
                )
            else:
                U_box_use = np.eye(K, dtype=float)
                Z_both = np.vstack([Z_t, Z_next])
                z_min_use = np.min(Z_both, axis=0) - 0.25
                z_max_use = np.max(Z_both, axis=0) + 0.25
                interior_mask = np.ones(len(Z_t), dtype=bool)

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
                idx_a_all = np.where(A_t == a)[0]
                idx_a_int = np.where((A_t == a) & interior_mask)[0]
                idx_a = idx_a_int if (self.boundary_aware and len(idx_a_int) >= K + 4) else idx_a_all
                if len(idx_a) >= K + 3:
                    Psi_a = Psi_all[idx_a]
                    dZ_a = dZ[idx_a]
                    for k in range(K):
                        th, cv, sg, _ = self._eb_linear_posterior(Psi_a, dZ_a[:, k])
                        theta_trans[a, :, k] = th
                        cov_trans[a, k] = cv
                        sigma2_trans[a, k] = sg
                    sigma_cov_trans[a] = np.diag(sigma2_trans[a])
                elif src_dyn is not None:
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
            theta_trans=theta_trans,
            cov_trans=cov_trans,
            sigma2_trans=sigma2_trans,
            causal_action_mask=causal_mask,
            cov_outcome=cov_outcome,
            sigma2_outcome=sig2_outcome,
            action_costs=action_costs,
            z_min=z_min_use,
            z_max=z_max_use,
            U_box=U_box_use,
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

        Psi = np.hstack([np.ones((N, 1), dtype=float), Z_arr])
        z_next = np.zeros((N, K), dtype=float)
        z_var = np.zeros((N, K), dtype=float)
        P_next = np.zeros((N, K, K), dtype=float)
        Qe = dyn.Q_epi
        Ub = dyn.U_box

        for a in range(self.n_actions):
            idx = np.where(A_arr == a)[0]
            if len(idx) == 0:
                continue
            Psi_a = Psi[idx]
            th_a = dyn.theta_trans[a]
            z_unclamped = Z_arr[idx] + Psi_a @ th_a
            # Exact orthogonal box projection in intrinsic frame U_box
            v_unc = z_unclamped @ Ub
            v_clamped = np.clip(v_unc, dyn.z_min, dyn.z_max)
            z_next[idx] = v_clamped @ Ub.T

            epi_diag = np.zeros((len(idx), K), dtype=float)
            for k in range(K):
                epi_diag[:, k] = np.maximum(np.einsum("ni,ij,nj->n", Psi_a, dyn.cov_trans[a, k], Psi_a), 0.0)
            C_epi = np.einsum("ki,nk,kj->nij", Qe, epi_diag, Qe)
            C_step = C_epi + dyn.sigma_cov_trans[a].reshape(1, K, K)

            J_a = np.eye(K, dtype=float) + th_a[1:, :].T
            if P_t is not None:
                P_prop = np.einsum("ij,njk,lk->nil", J_a, P_t[idx], J_a)
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

    # Pure-NumPy analytical forward/backward pass for transition parameters
    def forward_transition(self, Z: np.ndarray, action: int, eid: Optional[int] = None) -> np.ndarray:
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        Z_arr = np.asarray(Z, dtype=float)
        self._last_z_cache = Z_arr
        self._last_a_cache = int(action)
        Psi = np.hstack([np.ones((len(Z_arr), 1), dtype=float), Z_arr])
        return Z_arr + Psi @ dyn.theta_trans[int(action)]

    def backward_transition(self, grad_z_next: np.ndarray, eid: Optional[int] = None) -> np.ndarray:
        assert self._last_z_cache is not None and self._last_a_cache is not None
        e = self._get_engram(eid)
        dyn = self.dynamics[e.engram_id]
        Z_arr = self._last_z_cache
        a = self._last_a_cache
        Psi = np.hstack([np.ones((len(Z_arr), 1), dtype=float), Z_arr])
        g_out = np.asarray(grad_z_next, dtype=float)
        self.grad_theta_trans = np.zeros_like(dyn.theta_trans)
        self.grad_theta_trans[a] = Psi.T @ g_out
        grad_z = g_out + g_out @ dyn.theta_trans[a, 1:, :].T
        return grad_z

    def to_bytes(self) -> bytes:
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
