"""Phase 16: Cross-World Transfer and Structural Generalisation (models/transfer.py).

Implements ModularTransferModel in pure NumPy with analytical forward and backward passes:
    Z = (X - x_mean) @ W_adapter + b_adapter
    Phi(Z) = [Z_0..Z_{K-1}, Z_i * Z_j for i < j]
    y_hat = core_gate * (Phi(Z) @ w_core + b_core) + (Phi(Z) @ w_world + b_world)

Separates:
1. Sensor Adapter (W_adapter, b_adapter): aligns world-specific observation distributions.
2. Reusable Structural Core (w_core, b_core): preserves transferable structural rules across worlds.
3. World-Specific Residual (w_world, b_world): few-shot adaptation to new rules without overwriting w_core,
   plus an automatic negative-transfer gate that zeroes out w_core if source structure degrades target fit.
"""
import itertools
from typing import Dict, List, Optional, Tuple
import numpy as np


class ModularTransferModel:
    """Pure-NumPy modular world-transfer architecture with analytical gradients."""

    def __init__(self, in_dim: int, n_latent: int = 4, l2_reg: float = 1e-2, seed: int = 0):
        self.in_dim = int(in_dim)
        self.n_latent = int(n_latent)
        self.l2_reg = float(l2_reg)
        self.seed = int(seed)

        self.pair_indices: List[Tuple[int, int]] = list(itertools.combinations(range(self.n_latent), 2))
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
        D = np.hstack([np.ones((len(X), 1)), Phi])
        reg = self.l2_reg * np.eye(D.shape[1])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(D.T @ D + reg, D.T @ y)
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
        n_steps: int = 80,
        lr: float = 0.04,
    ) -> None:
        """Align target sensor adapter W_adapter to the frozen source structural core (w_core, b_core)
        via analytical gradient descent on few-shot target samples (X_tgt, Y_tgt), keeping w_core frozen.
        """
        X_tgt = np.asarray(X_tgt, dtype=float)
        y_tgt = np.asarray(Y_tgt, dtype=float).reshape(-1, 1)
        self.fit_normalizer(X_tgt)
        Xc = X_tgt - self.x_mean
        # Initialize W_adapter in the target active subspace
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        V_tgt = Vt[:self.n_latent].T.copy()

        # Try starting from current W_adapter vs target subspace SVD and pick best
        best_W = self.W_adapter.copy()
        best_loss = float("inf")
        rng = np.random.default_rng(self.seed + 71)
        candidates = [self.W_adapter.copy(), V_tgt]
        for _ in range(4):
            Q, _ = np.linalg.qr(rng.standard_normal((self.n_latent, self.n_latent)))
            candidates.append(V_tgt @ Q)

        orig_gate = self.core_gate
        orig_w_world = self.w_world.copy()
        orig_b_world = self.b_world.copy()
        self.core_gate = 1.0
        self.w_world.fill(0.0)
        self.b_world.fill(0.0)

        for W0 in candidates:
            self.W_adapter = W0.copy()
            for st in range(n_steps):
                self.zero_grad()
                yp = self.forward(X_tgt)
                grad_out = 2.0 * (yp - y_tgt) / len(X_tgt)
                self.backward(grad_out)
                step_lr = lr * (0.985 ** st)
                self.W_adapter -= step_lr * self.grad_W_adapter
                # Re-orthogonalize columns of W_adapter onto the Stiefel manifold (orthonormal columns)
                U, _, Vt_w = np.linalg.svd(self.W_adapter, full_matrices=False)
                self.W_adapter = U @ Vt_w
            loss = float(np.mean((self.forward(X_tgt) - y_tgt) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_W = self.W_adapter.copy()

        self.W_adapter = best_W
        self.core_gate = orig_gate
        self.w_world = orig_w_world
        self.b_world = orig_b_world

    def adapt_few_shot(
        self,
        X_tgt: np.ndarray,
        Y_tgt: np.ndarray,
        align_sensors: bool = False,
        allow_negative_transfer_gate: bool = True,
    ) -> float:
        """Few-shot target adaptation:
        1. Optionally align target sensor adapter to the frozen source core (for sensor-shift worlds).
        2. Evaluate negative-transfer gate (core_gate = 1.0 if source core reduces target variance, else 0.0).
        3. Fit regularized world-specific residual (w_world, b_world) on the remaining target residual.
        """
        X_tgt = np.asarray(X_tgt, dtype=float)
        y_tgt = np.asarray(Y_tgt, dtype=float).reshape(-1, 1)

        if align_sensors:
            self.align_sensor_adapter_few_shot(X_tgt, Y_tgt)
        else:
            self.fit_normalizer(X_tgt)

        Xc = X_tgt - self.x_mean
        Z = Xc @ self.W_adapter + self.b_adapter
        Phi = self._compute_phi(Z)
        y_core = Phi @ self.w_core + self.b_core

        if allow_negative_transfer_gate:
            var_with_core = float(np.var(y_tgt - y_core))
            var_without_core = float(np.var(y_tgt))
            self.core_gate = 1.0 if var_with_core <= 0.90 * var_without_core else 0.0
        else:
            self.core_gate = 1.0

        y_res = y_tgt - self.core_gate * y_core
        D = np.hstack([np.ones((len(X_tgt), 1)), Phi])
        reg = max(self.l2_reg, 0.05) * np.eye(D.shape[1])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(D.T @ D + reg, D.T @ y_res)
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
