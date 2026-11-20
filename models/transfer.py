"""Phase 16: Cross-World Transfer and Structural Generalisation (models/transfer.py).

Implements ModularTransferModel in pure NumPy with analytical forward and backward passes:
    Z = (X - x_mean) @ W_adapter + b_adapter
    Phi(Z) = [Z_0..Z_{K-1}, Z_i * Z_j for 0 <= i <= j < K]   (O(K)-invariant full degree-2 basis)
    y_hat = core_gate * (Phi(Z) @ w_core + b_core) + (Phi(Z) @ w_world + b_world)

Separates:
1. Sensor Adapter (W_adapter, b_adapter): aligns world-specific observation distributions on O(K).
2. Reusable Structural Core (w_core, b_core): preserves transferable structural rules across worlds.
3. World-Specific Residual (w_world, b_world): few-shot adaptation via sufficient-statistic
   unique-state Leave-One-Out Cross-Validation (LOOCV) ridge regression without overwriting w_core,
   plus an automatic negative-transfer gate that zeroes out w_core if source structure degrades target fit.
"""
import itertools
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class ModularTransferModel:
    """Pure-NumPy modular world-transfer architecture with analytical gradients."""

    def __init__(self, in_dim: int, n_latent: int = 4, l2_reg: float = 1e-2, seed: int = 0):
        self.in_dim = int(in_dim)
        self.n_latent = int(n_latent)
        self.l2_reg = float(l2_reg)
        self.seed = int(seed)

        # Full degree-2 quadratic indices (i <= j) so span{1, Phi(Z)} is strictly O(K)-invariant
        self.pair_indices: List[Tuple[int, int]] = list(
            itertools.combinations_with_replacement(range(self.n_latent), 2)
        )
        self.phi_dim = self.n_latent + len(self.pair_indices)

        rng = np.random.default_rng(seed)
        self.x_mean = np.zeros((1, self.in_dim), dtype=float)
        self.W_adapter = np.eye(self.in_dim, self.n_latent, dtype=float)
        self.b_adapter = np.zeros((1, self.n_latent), dtype=float)

        self.w_core = rng.normal(0.0, 0.05, size=(self.phi_dim, 1))
        self.b_core = np.zeros((1,), dtype=float)
        self.core_gate = 1.0

        self.w_world = np.zeros((self.phi_dim, 1), dtype=float)
        self.b_world = np.zeros((1,), dtype=float)

        # Stored source adapter so evaluating source retention uses the source sensory interface
        self._source_x_mean: Optional[np.ndarray] = None
        self._source_W_adapter: Optional[np.ndarray] = None
        self._source_b_adapter: Optional[np.ndarray] = None
        self.last_sensor_shift_detected: bool = False
        self.last_subspace_residual: float = 0.0

        # Gradient buffers
        self.grad_W_adapter = np.zeros_like(self.W_adapter)
        self.grad_b_adapter = np.zeros_like(self.b_adapter)
        self.grad_w_core = np.zeros_like(self.w_core)
        self.grad_b_core = np.zeros_like(self.b_core)
        self.grad_w_world = np.zeros_like(self.w_world)
        self.grad_b_world = np.zeros_like(self.b_world)

        self._xc_cache: Optional[np.ndarray] = None
        self._z_cache: Optional[np.ndarray] = None
        self._phi_cache: Optional[np.ndarray] = None
        self.input_dim = self.in_dim

    @property
    def num_parameters(self) -> int:
        return int(
            self.W_adapter.size + self.b_adapter.size
            + self.w_core.size + self.b_core.size
            + self.w_world.size + self.b_world.size
        )

    def fit_normalizer(self, X: np.ndarray) -> None:
        self.x_mean = np.mean(np.asarray(X, dtype=float), axis=0, keepdims=True)

    def _compute_phi(self, Z: np.ndarray) -> np.ndarray:
        cols = [Z[:, k] for k in range(self.n_latent)]
        for i, j in self.pair_indices:
            cols.append(Z[:, i] * Z[:, j])
        return np.column_stack(cols)

    @staticmethod
    def _deduplicate_states(Phi: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate identical feature rows into sufficient statistics (Phi_u, y_mean, count)."""
        keys = np.round(Phi / tol).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        G = len(counts)
        Phi_u = np.zeros((G, Phi.shape[1]), dtype=float)
        y_u = np.zeros((G, 1), dtype=float)
        for g in range(G):
            mask = (inv == g)
            Phi_u[g] = Phi[mask].mean(axis=0)
            y_u[g, 0] = y[mask].mean()
        return Phi_u, y_u, counts.astype(float)

    @classmethod
    def _loocv_ridge_unique(
        cls,
        Phi: np.ndarray,
        y: np.ndarray,
        lambdas: Tuple[float, ...] = (1e-3, 1e-2, 5e-2, 0.2, 0.5, 1.0, 2.0, 5.0, 15.0, 50.0, 1e4),
    ) -> Tuple[np.ndarray, float]:
        """Exact closed-form Leave-One-Group-Out (unique-state) weighted ridge CV."""
        Phi_u, y_u, w = cls._deduplicate_states(Phi, y)
        G = len(y_u)
        D = np.hstack([np.ones((G, 1)), Phi_u])
        sw = np.sqrt(w).reshape(-1, 1)
        Dw = D * sw
        yw = y_u * sw
        p = D.shape[1]
        best_loo = float("inf")
        best_beta = np.zeros((p, 1), dtype=float)
        for lam in lambdas:
            reg = lam * np.eye(p)
            reg[0, 0] = 0.0
            A_inv = np.linalg.inv(Dw.T @ Dw + reg)
            beta = A_inv @ (Dw.T @ yw)
            yhat = D @ beta
            H_diag = w * np.sum((D @ A_inv) * D, axis=1)
            denom = np.maximum(1.0 - H_diag, 1e-4)
            loo = float(np.mean(((y_u.reshape(-1) - yhat.reshape(-1)) / denom) ** 2))
            if loo < best_loo:
                best_loo = loo
                best_beta = beta
        return best_beta, best_loo

    def fit_source(self, X: np.ndarray, Y: np.ndarray, W_init: Optional[np.ndarray] = None) -> None:
        """Fit sensor adapter and reusable structural core on source world (X, Y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float).reshape(-1, 1)
        self.fit_normalizer(X)
        Xc = X - self.x_mean
        if W_init is not None:
            self.W_adapter = np.asarray(W_init, dtype=float).copy()
        else:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.W_adapter = Vt[:self.n_latent].T.copy()
        self.b_adapter = np.zeros((1, self.n_latent), dtype=float)

        Z = Xc @ self.W_adapter + self.b_adapter
        Phi = self._compute_phi(Z)
        beta, _ = self._loocv_ridge_unique(Phi, y, lambdas=(1e-4, 1e-3, 1e-2, 0.1, 1.0))
        self.b_core = np.array([float(beta[0, 0])], dtype=float)
        self.w_core = beta[1:].copy()
        self.core_gate = 1.0
        self.w_world.fill(0.0)
        self.b_world.fill(0.0)

        self._source_x_mean = self.x_mean.copy()
        self._source_W_adapter = self.W_adapter.copy()
        self._source_b_adapter = self.b_adapter.copy()

    def predict_source(self, X_src: np.ndarray) -> np.ndarray:
        """Evaluate source-world retention using the frozen source core and source sensor adapter."""
        X_src = np.asarray(X_src, dtype=float)
        xm = self._source_x_mean if self._source_x_mean is not None else self.x_mean
        Wa = self._source_W_adapter if self._source_W_adapter is not None else self.W_adapter
        ba = self._source_b_adapter if self._source_b_adapter is not None else self.b_adapter
        Z = (X_src - xm) @ Wa + ba
        Phi = self._compute_phi(Z)
        return (Phi @ self.w_core + self.b_core).reshape(-1)

    def align_sensor_adapter_few_shot(
        self,
        X_tgt: np.ndarray,
        Y_tgt: np.ndarray,
        n_restarts: int = 20,
        n_steps: int = 180,
        lr: float = 0.08,
    ) -> None:
        """Align target sensor adapter W_adapter = V_tgt @ R (R in O(K)) to the frozen source core
        via Adam + polar SVD retraction across balanced det(R) = +/-1 restarts.
        """
        X_tgt = np.asarray(X_tgt, dtype=float)
        y_tgt = np.asarray(Y_tgt, dtype=float).reshape(-1, 1)
        # Keep x_mean = 0 so 20-shot sample-mean noise does not corrupt the active subspace
        self.x_mean = np.zeros((1, self.in_dim), dtype=float)
        _, _, Vt = np.linalg.svd(X_tgt, full_matrices=False)
        V_tgt = Vt[:self.n_latent].T.copy()  # (in_dim, K)
        U_t = X_tgt @ V_tgt  # (N, K)

        K = self.n_latent
        rng = np.random.default_rng(self.seed + 71)
        best_R = np.eye(K, dtype=float)
        best_loss = float("inf")

        for r_idx in range(n_restarts):
            if r_idx == 0:
                R = np.eye(K, dtype=float)
            else:
                Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
                # Balance connected components det=+1 (SO(K)) and det=-1 (O(K)\SO(K))
                if (r_idx % 2 == 0 and np.linalg.det(Q) < 0) or (r_idx % 2 == 1 and np.linalg.det(Q) > 0):
                    Q[:, 0] *= -1.0
                R = Q

            m_adam = np.zeros_like(R)
            v_adam = np.zeros_like(R)
            for st in range(1, n_steps + 1):
                Z = U_t @ R + self.b_adapter
                Phi = self._compute_phi(Z)
                yp = Phi @ self.w_core + self.b_core
                g = 2.0 * (yp - y_tgt) / len(y_tgt)
                g_phi = g @ self.w_core.T
                g_z = g_phi[:, :K].copy()
                for idx, (i, j) in enumerate(self.pair_indices):
                    gp = g_phi[:, K + idx]
                    g_z[:, i] += gp * Z[:, j]
                    g_z[:, j] += gp * Z[:, i]
                g_R = U_t.T @ g_z
                m_adam = 0.9 * m_adam + 0.1 * g_R
                v_adam = 0.999 * v_adam + 0.001 * (g_R ** 2)
                mh = m_adam / (1.0 - 0.9 ** st)
                vh = v_adam / (1.0 - 0.999 ** st)
                R = R - lr * mh / (np.sqrt(vh) + 1e-8)
                Ur, _, Vtr = np.linalg.svd(R, full_matrices=False)
                R = Ur @ Vtr

            Z = U_t @ R + self.b_adapter
            loss = float(np.mean((self._compute_phi(Z) @ self.w_core + self.b_core - y_tgt) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_R = R.copy()

        self.W_adapter = V_tgt @ best_R

    def detect_sensor_shift(self, X_tgt: np.ndarray, threshold: float = 0.10) -> Tuple[bool, float]:
        """Unsupervised sensor-shift detector using only raw target observations X_tgt.

        Measures the normalized Frobenius residual of (X_tgt - _source_x_mean) orthogonal
        to the learned source sensor subspace col(_source_W_adapter):
            r = ||Xc (I - W_src W_src^T)||_F / max(||Xc||_F, 1e-8).
        """
        X_tgt = np.asarray(X_tgt, dtype=float)
        xm = self._source_x_mean if self._source_x_mean is not None else self.x_mean
        W_src = self._source_W_adapter if self._source_W_adapter is not None else self.W_adapter
        Xc = X_tgt - xm
        proj = (Xc @ W_src) @ W_src.T
        denom = float(np.linalg.norm(Xc, ord="fro"))
        if denom < 1e-8:
            return False, 0.0
        residual = float(np.linalg.norm(Xc - proj, ord="fro") / denom)
        return bool(residual > threshold), residual

    def adapt_few_shot(
        self,
        X_tgt: np.ndarray,
        Y_tgt: np.ndarray,
        align_sensors: Union[bool, str] = "auto",
        allow_negative_transfer_gate: bool = True,
    ) -> float:
        """Few-shot target adaptation:
        1. If align_sensors == 'auto', autonomously detect sensor shift from raw X_tgt alone
           (zero ground-truth hints). Align target sensor adapter on O(K) if shifted;
           otherwise preserve _source_x_mean to avoid 20-shot mean-shift noise.
        2. Evaluate negative-transfer gate via unique-state LOOCV comparing (core_gate=1 vs 0).
        3. Fit LOOCV-optimal world-specific residual (w_world, b_world) without overwriting w_core.
        """
        X_tgt = np.asarray(X_tgt, dtype=float)
        y_tgt = np.asarray(Y_tgt, dtype=float).reshape(-1, 1)

        if align_sensors == "auto":
            shifted, resid = self.detect_sensor_shift(X_tgt)
            self.last_sensor_shift_detected = shifted
            self.last_subspace_residual = resid
            do_align = shifted
        else:
            do_align = bool(align_sensors)
            _, resid = self.detect_sensor_shift(X_tgt)
            self.last_sensor_shift_detected = do_align
            self.last_subspace_residual = resid

        if do_align:
            self.align_sensor_adapter_few_shot(X_tgt, Y_tgt)
        elif self._source_x_mean is not None:
            self.x_mean = self._source_x_mean.copy()

        Xc = X_tgt - self.x_mean
        Z = Xc @ self.W_adapter + self.b_adapter
        Phi = self._compute_phi(Z)
        y_core = Phi @ self.w_core + self.b_core

        beta_with, loo_with = self._loocv_ridge_unique(Phi, y_tgt - y_core)
        if allow_negative_transfer_gate:
            beta_without, loo_without = self._loocv_ridge_unique(Phi, y_tgt)
            if loo_with < 0.90 * loo_without:
                self.core_gate = 1.0
                beta = beta_with
            else:
                self.core_gate = 0.0
                beta = beta_without
        else:
            self.core_gate = 1.0
            beta = beta_with

        self.b_world = np.array([float(beta[0, 0])], dtype=float)
        self.w_world = beta[1:].copy()
        return self.core_gate

    def clone_for_target(self) -> "ModularTransferModel":
        m = ModularTransferModel(self.in_dim, self.n_latent, self.l2_reg, self.seed)
        m.x_mean = self.x_mean.copy()
        m.W_adapter = self.W_adapter.copy()
        m.b_adapter = self.b_adapter.copy()
        m.w_core = self.w_core.copy()
        m.b_core = self.b_core.copy()
        m.core_gate = self.core_gate
        m.w_world = self.w_world.copy()
        m.b_world = self.b_world.copy()
        m._source_x_mean = self._source_x_mean.copy() if self._source_x_mean is not None else None
        m._source_W_adapter = self._source_W_adapter.copy() if self._source_W_adapter is not None else None
        m._source_b_adapter = self._source_b_adapter.copy() if self._source_b_adapter is not None else None
        return m

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self._xc_cache = X - self.x_mean
        self._z_cache = self._xc_cache @ self.W_adapter + self.b_adapter
        self._phi_cache = self._compute_phi(self._z_cache)
        y_core = self._phi_cache @ self.w_core + self.b_core
        y_world = self._phi_cache @ self.w_world + self.b_world
        return self.core_gate * y_core + y_world

    def get_latent_state(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - self.x_mean
        return Xc @ self.W_adapter + self.b_adapter

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._xc_cache is not None and self._z_cache is not None and self._phi_cache is not None
        g = np.asarray(grad_out, dtype=float).reshape(-1, 1)

        self.grad_w_world += self._phi_cache.T @ g
        self.grad_b_world += np.sum(g, axis=0)
        self.grad_w_core += self.core_gate * (self._phi_cache.T @ g)
        self.grad_b_core += self.core_gate * np.sum(g, axis=0)

        w_eff = self.core_gate * self.w_core + self.w_world
        g_phi = g @ w_eff.T

        g_z = g_phi[:, :self.n_latent].copy()
        for idx, (i, j) in enumerate(self.pair_indices):
            gp = g_phi[:, self.n_latent + idx]
            g_z[:, i] += gp * self._z_cache[:, j]
            g_z[:, j] += gp * self._z_cache[:, i]

        self.grad_W_adapter += self._xc_cache.T @ g_z
        self.grad_b_adapter += np.sum(g_z, axis=0, keepdims=True)
        return g_z @ self.W_adapter.T

    def zero_grad(self) -> None:
        self.grad_W_adapter.fill(0.0)
        self.grad_b_adapter.fill(0.0)
        self.grad_w_core.fill(0.0)
        self.grad_b_core.fill(0.0)
        self.grad_w_world.fill(0.0)
        self.grad_b_world.fill(0.0)

    def get_params(self):
        return [
            ("W_adapter", self.W_adapter, self.grad_W_adapter),
            ("b_adapter", self.b_adapter, self.grad_b_adapter),
            ("w_core", self.w_core, self.grad_w_core),
            ("b_core", self.b_core, self.grad_b_core),
            ("w_world", self.w_world, self.grad_w_world),
            ("b_world", self.b_world, self.grad_b_world),
        ]

    def state_arrays(self):
        return {n: p.copy() for n, p, _ in self.get_params()}
