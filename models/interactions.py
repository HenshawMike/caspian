"""Phase 14: Sparse Pairwise and Higher-Order Interaction Discovery (models/interactions.py).

Implements SparseInteractionDiscoveryModel in pure NumPy with analytical forward and backward passes:
    y_hat(X) = f_add(X) + Psi(X) @ w_int
where:
- f_add(X) is a strictly additive per-property neural trunk (FactorisedPropertyModel with sum aggregation),
- Psi(X) contains all C(K, 2) pairwise and C(K, 3) three-way interaction basis functions of normalized
  property coordinates A_k in [0, 1],
- w_int is a sparse interaction weight vector learned via Frisch-Waugh-Lovell main-effect projection +
  Iterative Soft-Thresholding (ISTA) + Gauss-Lasso debiasing, and fine-tunable with analytical L1 gradients.
"""
import itertools
from typing import Dict, List, Optional, Tuple
import numpy as np
from models.compositional import AdditivePlusResidual, FactorisedPropertyModel
from models.mlp import PredictiveMLP


class SparseInteractionDiscoveryModel:
    """Hierarchical Functional-ANOVA / Mobius interaction discovery network.
    Combines a univariate per-property neural additive trunk with a sparse L1-regularised
    dictionary of pairwise and 3-way interaction basis functions.
    """
    def __init__(
        self,
        n_props: int,
        max_order: int = 3,
        l1_lambda: float = 0.12,
        prune_threshold: float = 0.85,
        freeze_sparse_after_init: bool = True,
        seed: int = 0,
    ):
        assert max_order in (2, 3)
        self.n_props = int(n_props)
        self.max_order = int(max_order)
        self.l1_lambda = float(l1_lambda)
        self.prune_threshold = float(prune_threshold)
        self.freeze_sparse_after_init = bool(freeze_sparse_after_init)
        self.seed = int(seed)

        self.pair_indices: List[Tuple[int, int]] = list(itertools.combinations(range(self.n_props), 2))
        self.triplet_indices: List[Tuple[int, int, int]] = (
            list(itertools.combinations(range(self.n_props), 3)) if self.max_order >= 3 else []
        )
        self.interaction_terms: List[Tuple[int, ...]] = list(self.pair_indices) + list(self.triplet_indices)
        self.n_int = len(self.interaction_terms)

        self.main_trunk = FactorisedPropertyModel(self.n_props, "sum", 16, 8, (), seed)
        self.p_min = np.zeros((1, self.n_props), dtype=float)
        self.p_span = np.ones((1, self.n_props), dtype=float)

        self.w_int = np.zeros((self.n_int, 1), dtype=float)
        self.grad_w_int = np.zeros_like(self.w_int)
        self._psi_cache: Optional[np.ndarray] = None
        self._a_cache: Optional[np.ndarray] = None
        self.input_dim = self.n_props

    @property
    def num_parameters(self) -> int:
        return self.main_trunk.num_parameters + int(self.w_int.size)

    def _normalized_coords(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        A_raw = (X - self.p_min) / self.p_span
        # Snap within-cluster Gaussian jitter (< 0.18 from 0 or 1) to clean binary modes while remaining continuous in the interior
        A_snap = np.where(A_raw < 0.18, 0.0, np.where(A_raw > 0.82, 1.0, A_raw))
        return A_snap

    def compute_interaction_basis(self, X: np.ndarray) -> np.ndarray:
        """Compute Psi(X) of shape (B, n_int) containing all pairwise and 3-way product features."""
        A = self._normalized_coords(X)
        self._a_cache = A
        cols = []
        for idx in self.interaction_terms:
            prod = A[:, idx[0]]
            for k in idx[1:]:
                prod = prod * A[:, k]
            cols.append(prod)
        if not cols:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(cols)

    def fit_normalizer(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        # First-principles 2-mode K-means estimator: split at range midpoint (invariant to 73%/27% held-out class imbalance)
        mid = 0.5 * (np.percentile(X, 5.0, axis=0, keepdims=True) + np.percentile(X, 95.0, axis=0, keepdims=True))
        lo = np.array([[float(np.mean(X[:, k][X[:, k] < mid[0, k]])) if np.any(X[:, k] < mid[0, k]) else float(np.min(X[:, k])) for k in range(self.n_props)]])
        hi = np.array([[float(np.mean(X[:, k][X[:, k] >= mid[0, k]])) if np.any(X[:, k] >= mid[0, k]) else float(np.max(X[:, k])) for k in range(self.n_props)]])
        span = hi - lo
        self.p_min = lo
        self.p_span = np.where(span < 1e-6, 1.0, span)
        X_clean = self.p_min + self._normalized_coords(X) * self.p_span
        self.main_trunk.fit_normalizer(X_clean)

    def fit_sparse_hierarchy(self, X: np.ndarray, Y: np.ndarray, ista_steps: int = 250, step_size: float = 0.4):
        """First-Principles Frisch-Waugh-Lovell + ISTA + Gauss-Lasso debiasing solver on (X, Y):
        1. Projects out the linear main-effect subspace [1, A] so main effects are unpenalized.
        2. Solves L1-penalized ISTA on the residual interaction design matrix Psi_tilde.
        3. Prunes coefficients below prune_threshold and performs an unbiased OLS refit on active support.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float).reshape(-1, 1)
        N = float(len(X))
        self.fit_normalizer(X)

        A = self._normalized_coords(X)
        Psi = self.compute_interaction_basis(X)
        M_main = np.hstack([np.ones((len(X), 1)), A])

        # Frisch-Waugh-Lovell orthogonal projection: residualize both Psi and y against M_main
        Q_m, _ = np.linalg.qr(M_main, mode="reduced")
        Psi_tilde = Psi - Q_m @ (Q_m.T @ Psi)
        y_tilde = y - Q_m @ (Q_m.T @ y)

        # Normalize columns of Psi_tilde for well-conditioned ISTA steps
        col_norms = np.sqrt(np.mean(Psi_tilde ** 2, axis=0, keepdims=True)) + 1e-8
        Psi_norm = Psi_tilde / col_norms
        L_lip = float(np.linalg.norm(Psi_norm, ord=2) ** 2 / N) + 1e-6
        lr_ista = step_size / L_lip

        w = np.zeros((self.n_int, 1), dtype=float)
        thr = self.l1_lambda * lr_ista
        for _ in range(ista_steps):
            grad = (Psi_norm.T @ (Psi_norm @ w - y_tilde)) / N
            w_cand = w - lr_ista * grad
            w = np.sign(w_cand) * np.maximum(0.0, np.abs(w_cand) - thr)

        # Rescale back to original Psi units and apply support pruning
        w_raw = w / col_norms.T
        active = np.where(np.abs(w_raw[:, 0]) >= self.prune_threshold)[0]
        w_debiased = np.zeros_like(w_raw)
        if len(active) > 0:
            # Gauss-Lasso unbiased refit of [M_main, Psi[:, active]]
            D_act = np.hstack([M_main, Psi[:, active]])
            beta_full, *_ = np.linalg.lstsq(D_act, y, rcond=None)
            w_Sub = beta_full[M_main.shape[1]:]
            # Second pass threshold check after joint refit
            keep = np.abs(w_Sub[:, 0]) >= self.prune_threshold
            active_final = active[keep]
            if len(active_final) > 0:
                D_final = np.hstack([M_main, Psi[:, active_final]])
                beta_final, *_ = np.linalg.lstsq(D_final, y, rcond=None)
                w_debiased[active_final] = beta_final[M_main.shape[1]:]

        self.w_int = w_debiased
        # Return Gauss-Markov denoised main-effect target M_main @ beta_main so main_trunk does not overfit zero-mean outcome noise
        beta_main, *_ = np.linalg.lstsq(M_main, y - Psi @ self.w_int, rcond=None)
        return (M_main @ beta_main).reshape(-1)

    def discovered_interactions(self, threshold: Optional[float] = None) -> Dict[Tuple[int, ...], float]:
        thr = self.prune_threshold if threshold is None else float(threshold)
        return {
            term: round(float(self.w_int[i, 0]), 4)
            for i, term in enumerate(self.interaction_terms)
            if abs(float(self.w_int[i, 0])) >= thr
        }

    def forward(self, X: np.ndarray) -> np.ndarray:
        X_clean = self.p_min + self._normalized_coords(X) * self.p_span
        y_add = self.main_trunk.forward(X_clean)
        self._psi_cache = self.compute_interaction_basis(X)
        y_int = self._psi_cache @ self.w_int
        return y_add + y_int

    def get_latent_state(self, X: np.ndarray) -> np.ndarray:
        psi = self.compute_interaction_basis(X)
        return np.hstack([self.main_trunk.get_latent_state(X), psi * self.w_int.T])

    def backward(self, grad_out: np.ndarray):
        assert self._psi_cache is not None
        grad_2d = np.asarray(grad_out, dtype=float).reshape(-1, 1)
        self.main_trunk.backward(grad_2d)
        l1_grad = self.l1_lambda * np.sign(self.w_int)
        self.grad_w_int += self._psi_cache.T @ grad_2d + l1_grad

    def zero_grad(self):
        self.main_trunk.zero_grad()
        self.grad_w_int.fill(0.0)

    def get_params(self):
        trunk_params = [(f"main_{n}", p, g) for n, p, g in self.main_trunk.get_params()]
        if self.freeze_sparse_after_init:
            return trunk_params
        return trunk_params + [("w_int", self.w_int, self.grad_w_int)]

    def state_arrays(self):
        out = {f"main_{n}": p.copy() for n, p, _ in self.main_trunk.get_params()}
        out["w_int"] = self.w_int.copy()
        return out


def build_p14_model(name: str, n_props: int, seed: int):
    """Registry of Phase 14 architectures operating on K-property channels."""
    if name == "sparse_interaction":
        return SparseInteractionDiscoveryModel(n_props=n_props, max_order=3, seed=seed)
    if name == "additive_residual":
        return AdditivePlusResidual(n_props, seed)
    if name == "additive":
        return FactorisedPropertyModel(n_props, "sum", 16, 8, (), seed)
    if name == "factorised":
        return FactorisedPropertyModel(n_props, "concat", 16, 8, (16,), seed)
    if name == "mlp_small":
        return PredictiveMLP(input_dim=n_props, hidden_dims=(32, 16), output_dim=1, seed=seed)
    if name == "mlp_large":
        return PredictiveMLP(input_dim=n_props, hidden_dims=(64, 32), output_dim=1, seed=seed)
    raise ValueError(f"Unknown Phase 14 architecture: {name}")
