"""Phase 15: Causal Discovery and Intervention Learning (models/causal.py).

Implements CausalStructuralModel in pure NumPy with analytical forward and backward passes,
supporting:
1. Sparse Causal Parent Discovery (via interventional ISTA + Gauss-Lasso debiasing) to separate
   direct causal parents Pa(Y) from spurious/confounded correlations.
2. Single- and multi-variable interventional prediction: E[Y | do(X = x)].
3. Pearl's 3-step Counterfactual Prediction (Abduction-Action-Prediction):
       u_hat = Y_obs - f_causal(X_obs)
       Y_cf  = f_causal(X_do) + u_hat
"""
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from models.layers import LinearLayer
from models.normalization import FeatureNormalizer


class CausalStructuralModel:
    """Pure-NumPy linear/sparse structural causal model with analytical gradients
    and Pearl's Abduction-Action-Prediction counterfactual inference.
    """
    def __init__(
        self,
        n_vars: int,
        l1_lambda: float = 0.25,
        edge_threshold: float = 1.25,
        freeze_mask: bool = True,
        seed: int = 0,
    ):
        self.n_vars = int(n_vars)
        self.l1_lambda = float(l1_lambda)
        self.edge_threshold = float(edge_threshold)
        self.freeze_mask = bool(freeze_mask)
        self.seed = int(seed)

        self.normalizer = FeatureNormalizer()
        self.linear = LinearLayer(self.n_vars, 1, use_bias=True, init_method="zeros", seed=seed)
        self.causal_mask = np.ones((self.n_vars, 1), dtype=float)
        self.input_dim = self.n_vars

    @property
    def num_parameters(self) -> int:
        return int(self.linear.weights.size + self.linear.bias.size)

    def fit_normalizer(self, X: np.ndarray) -> None:
        self.normalizer.fit(X)

    def fit_causal_structure(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        ista_steps: int = 300,
        step_size: float = 0.4,
    ) -> None:
        """Discover direct causal parents Pa(Y) and structural weights via standardized ISTA
        followed by Gauss-Lasso unbiased OLS refit on the active causal support.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float).reshape(-1, 1)
        N = float(len(X))

        self.fit_normalizer(X)
        Xn = self.normalizer.transform(X)
        y_mean = float(np.mean(y))
        yc = y - y_mean

        L_lip = float(np.linalg.norm(Xn, ord=2) ** 2 / N) + 1e-6
        lr_ista = step_size / L_lip

        w = np.zeros((self.n_vars, 1), dtype=float)
        thr = self.l1_lambda * lr_ista
        for _ in range(ista_steps):
            grad = (Xn.T @ (Xn @ w - yc)) / N
            w_cand = w - lr_ista * grad
            w = np.sign(w_cand) * np.maximum(0.0, np.abs(w_cand) - thr)

        # Convert normalized weights to raw feature scale to apply interpretable edge threshold
        w_raw_scale = w[:, 0] / self.normalizer.std
        active = np.where(np.abs(w_raw_scale) >= self.edge_threshold)[0]

        mask = np.zeros((self.n_vars, 1), dtype=float)
        w_final = np.zeros((self.n_vars, 1), dtype=float)
        b_final = np.array([y_mean], dtype=float)

        if len(active) > 0:
            D = np.hstack([np.ones((len(X), 1)), Xn[:, active]])
            beta, *_ = np.linalg.lstsq(D, y, rcond=None)
            w_act_raw = beta[1:, 0] / self.normalizer.std[active]
            keep = np.abs(w_act_raw) >= self.edge_threshold
            active_final = active[keep]
            if len(active_final) > 0:
                D2 = np.hstack([np.ones((len(X), 1)), Xn[:, active_final]])
                beta2, *_ = np.linalg.lstsq(D2, y, rcond=None)
                mask[active_final, 0] = 1.0
                w_final[active_final, 0] = beta2[1:, 0]
                b_final = np.array([float(beta2[0, 0])], dtype=float)

        self.causal_mask = mask
        self.linear.weights = w_final
        self.linear.bias = b_final

    def discovered_parents(self, threshold: Optional[float] = None) -> Set[int]:
        thr = self.edge_threshold if threshold is None else float(threshold)
        eff = self.effective_weights()
        return {int(k) for k in range(self.n_vars) if abs(float(eff[k])) >= thr}

    def effective_weights(self) -> np.ndarray:
        """Return causal coefficients in raw (unnormalized) input units."""
        w = self.linear.weights[:, 0] * self.causal_mask[:, 0]
        if self.normalizer.is_fit:
            return w / self.normalizer.std
        return w

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xn = self.normalizer.transform(X) if self.normalizer.is_fit else X
        Xm = Xn * self.causal_mask.T
        return self.linear.forward(Xm)

    def get_latent_state(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xn = self.normalizer.transform(X) if self.normalizer.is_fit else X
        return Xn * (self.linear.weights.T * self.causal_mask.T)

    def predict_counterfactual(
        self,
        X_factual: np.ndarray,
        Y_factual: np.ndarray,
        X_counterfactual: np.ndarray,
    ) -> np.ndarray:
        """Pearl's 3-step Abduction-Action-Prediction:
        1. Abduction: infer unobserved exogenous noise u_hat = Y_factual - f_causal(X_factual).
        2. Action & Prediction: evaluate Y_cf = f_causal(X_counterfactual) + u_hat.
        """
        y_obs_hat = np.asarray(self.forward(X_factual)).reshape(-1)
        u_hat = np.asarray(Y_factual, dtype=float).reshape(-1) - y_obs_hat
        y_cf_base = np.asarray(self.forward(X_counterfactual)).reshape(-1)
        return y_cf_base + u_hat

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_in = self.linear.backward(grad_out)
        if self.freeze_mask:
            self.linear.grad_weights *= self.causal_mask
        return grad_in * self.causal_mask.T

    def zero_grad(self) -> None:
        self.linear.zero_grad()

    def get_params(self):
        return [("causal_" + n, p, g) for n, p, g in self.linear.get_params()]

    def state_arrays(self):
        return {
            "weights": self.linear.weights.copy(),
            "bias": self.linear.bias.copy(),
            "causal_mask": self.causal_mask.copy(),
        }
