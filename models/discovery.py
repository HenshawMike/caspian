"""Phase 13: Autonomous Property Discovery and Representation Learning from Raw Observations.

All models are implemented in pure NumPy with analytical forward and backward passes,
compatible with the Phase 11-12 training loop (forward, backward, get_params, zero_grad,
fit_normalizer, get_latent_state, num_parameters).

Architectures provided:
- 'latent_additive': Orthogonal sub-Gaussian unmixing bottleneck (K_slots=6) + strictly additive per-slot MLPs.
- 'latent_residual': Orthogonal sub-Gaussian unmixing bottleneck (K_slots=6) + additive trunk + factorised residual branch.
- 'raw_naive_additive': Strictly additive model applied directly to raw entangled sensor channels without unmixing.
- 'pca_statistical': Unsupervised top-K (K=3) PCA feature extractor + additive per-slot MLP readout.
- 'untrained_encoder': Frozen random MLP hidden encoder + trainable linear readout head.
- 'random_repr': Frozen random linear projection (K=3) + trainable linear readout head.
- 'raw_mlp_small': End-to-end PredictiveMLP (32, 16) on raw sensor vector.
- 'raw_mlp_large': End-to-end PredictiveMLP (64, 32) on raw sensor vector.
"""
from typing import List, Optional, Tuple
import numpy as np
from models.compositional import AdditivePlusResidual, FactorisedPropertyModel
from models.layers import LinearLayer
from models.mlp import PredictiveMLP
from models.normalization import FeatureNormalizer


def _project_orthogonal(W: np.ndarray) -> np.ndarray:
    """Project a square matrix W onto the orthogonal group O(K) via SVD polar decomposition."""
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    return U @ Vt


def _kurtosis_loss_and_grad_z(Z: np.ndarray, eps: float = 1e-6) -> Tuple[float, np.ndarray]:
    """Scale-invariant fourth-moment kurtosis contrast across columns of Z (B, K):
        L_kurt = (1 / (4 * K)) * sum_m [ mean_b(Z_{b,m}^4) / (mean_b(Z_{b,m}^2) + eps)^2 ]
    Minimised when each column aligns with a single sub-Gaussian source.
    Returns (loss, dL/dZ).
    """
    B, K = Z.shape
    v = np.mean(Z ** 2, axis=0, keepdims=True) + eps          # (1, K)
    mu4 = np.mean(Z ** 4, axis=0, keepdims=True)              # (1, K)
    kappa = mu4 / (v ** 2)                                    # (1, K)
    loss = float(0.25 * np.mean(kappa))
    grad_z = (Z ** 3 - kappa * v * Z) / (B * K * (v ** 2))    # exact analytical derivative
    return loss, grad_z


class OrthogonalUnmixingLayer:
    """Projects raw D-dim observations onto the active K-dim sensor subspace V_sub (D, K)
    and applies a learnable unmixing matrix W_unmix (K, K) regularised by scale-invariant
    kurtosis contrast and Stiefel orthogonality penalty.
    """
    def __init__(self, raw_dim: int, n_slots: int = 6, kurt_weight: float = 0.05,
                 orth_weight: float = 0.1, seed: int = 0):
        self.raw_dim = int(raw_dim)
        self.n_slots = int(n_slots)
        self.kurt_weight = float(kurt_weight)
        self.orth_weight = float(orth_weight)
        self.seed = int(seed)

        rng = np.random.default_rng(seed)
        self.mean = np.zeros((1, self.raw_dim), dtype=float)
        # Default orthonormal basis before fit_subspace is called
        A = rng.standard_normal((self.raw_dim, self.n_slots))
        Q, _ = np.linalg.qr(A)
        self.V_sub = Q[:, :self.n_slots].copy()               # (D, K), fixed subspace basis

        W0 = _project_orthogonal(rng.standard_normal((self.n_slots, self.n_slots)))
        self.W = W0.copy()                                    # (K, K), learnable unmixing matrix
        self.grad_W = np.zeros_like(self.W)
        self.scale = np.ones((1, self.n_slots), dtype=float)  # per-slot standardization scale

        self._x_sub: Optional[np.ndarray] = None
        self._z_raw: Optional[np.ndarray] = None

    @property
    def num_parameters(self) -> int:
        return int(self.W.size)

    def fit_subspace(self, X: np.ndarray, n_restarts: int = 10, steps: int = 200, lr: float = 0.2):
        """Unsupervised initialization of subspace basis V_sub and orthogonal unmixing W on unlabeled X."""
        X = np.asarray(X, dtype=float)
        self.mean = np.zeros((1, self.raw_dim), dtype=float)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self.V_sub = Vt[:self.n_slots].T.copy()               # (D, K) orthonormal columns
        X_sub = X @ self.V_sub                                # (N, K)

        best_W = self.W.copy()
        best_loss = float("inf")
        rng = np.random.default_rng(self.seed + 17)
        for r in range(n_restarts):
            W = np.eye(self.n_slots) if r == 0 else _project_orthogonal(rng.standard_normal((self.n_slots, self.n_slots)))
            for _ in range(steps):
                Z = X_sub @ W
                _, gz = _kurtosis_loss_and_grad_z(Z)
                gW = X_sub.T @ gz
                # Riemannian tangent step + polar retraction onto O(K)
                W = _project_orthogonal(W - lr * (gW - W @ gW.T @ W))
            Z_final = X_sub @ W
            loss_final, _ = _kurtosis_loss_and_grad_z(Z_final)
            if loss_final < best_loss:
                best_loss = loss_final
                best_W = W.copy()

        self.W = best_W
        Z_init = X_sub @ self.W
        self.scale = np.std(Z_init, axis=0, keepdims=True)
        self.scale = np.where(self.scale < 1e-6, 1.0, self.scale)

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self._x_sub = (X - self.mean) @ self.V_sub            # (B, K)
        self._z_raw = self._x_sub @ self.W                    # (B, K)
        return self._z_raw / self.scale

    def auxiliary_loss(self) -> float:
        """Compute unsupervised kurtosis + orthogonality regularisation loss on current batch."""
        assert self._z_raw is not None
        lk, _ = _kurtosis_loss_and_grad_z(self._z_raw)
        diff = self.W.T @ self.W - np.eye(self.n_slots)
        lo = 0.25 * float(np.sum(diff ** 2))
        return self.kurt_weight * lk + self.orth_weight * lo

    def backward(self, grad_z_norm: np.ndarray) -> np.ndarray:
        assert self._x_sub is not None and self._z_raw is not None
        grad_z_raw = grad_z_norm / self.scale
        if self.kurt_weight > 0.0:
            _, gk = _kurtosis_loss_and_grad_z(self._z_raw)
            grad_z_raw = grad_z_raw + self.kurt_weight * gk
        grad_W_orth = self.orth_weight * (self.W @ (self.W.T @ self.W - np.eye(self.n_slots)))
        self.grad_W += self._x_sub.T @ grad_z_raw + grad_W_orth
        return (grad_z_raw @ self.W.T) @ self.V_sub.T

    def zero_grad(self):
        self.grad_W.fill(0.0)

    def get_params(self):
        return [("unmix_W", self.W, self.grad_W)]


class LatentDiscoveryModel:
    """End-to-end discovery architecture:
    Raw sensors X (B, D) -> OrthogonalUnmixingLayer -> discovered slots Z (B, K_slots)
    -> structured compositional readout ('additive' or 'additive_residual').
    """
    def __init__(self, raw_dim: int, n_slots: int = 6, mode: str = "additive",
                 kurt_weight: float = 0.05, orth_weight: float = 0.1,
                 freeze_unmix: bool = True, seed: int = 0):
        assert mode in ("additive", "residual")
        self.raw_dim = int(raw_dim)
        self.n_slots = int(n_slots)
        self.mode = mode
        self.freeze_unmix = bool(freeze_unmix)
        self.unmix = OrthogonalUnmixingLayer(raw_dim, n_slots, kurt_weight=kurt_weight,
                                             orth_weight=orth_weight, seed=seed)
        if mode == "additive":
            self.head = FactorisedPropertyModel(n_slots, "sum", 16, 8, (), seed + 100)
        else:
            self.head = AdditivePlusResidual(n_slots, seed + 100)
        self.input_dim = raw_dim

    @property
    def num_parameters(self) -> int:
        return self.unmix.num_parameters + self.head.num_parameters

    def fit_normalizer(self, X: np.ndarray):
        self.unmix.fit_subspace(X)
        Z = self.unmix.forward(X)
        self.head.fit_normalizer(Z)

    def get_discovered_slots(self, X: np.ndarray) -> np.ndarray:
        """Return the K_slots discovered latent coordinates Z (B, K_slots)."""
        return self.unmix.forward(X)

    def slot_contributions(self, X: np.ndarray) -> np.ndarray:
        """Return per-slot univariate readout contributions (B, K_slots) by probing each slot individually."""
        Z = self.unmix.forward(X)
        B, K = Z.shape
        Z0 = np.zeros_like(Z)
        base = np.asarray(self.head.forward(Z0)).reshape(-1, 1)
        contribs = []
        for m in range(K):
            Zm = np.zeros_like(Z)
            Zm[:, m] = Z[:, m]
            ym = np.asarray(self.head.forward(Zm)).reshape(-1, 1) - base
            contribs.append(ym)
        return np.hstack(contribs)

    def get_latent_state(self, X: np.ndarray) -> np.ndarray:
        """Return the per-slot functional contributions (B, K_slots) so task-relevant slots dominate CKA/probing."""
        return self.slot_contributions(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = self.unmix.forward(X)
        return self.head.forward(Z)

    def backward(self, grad_out: np.ndarray):
        if self.mode == "additive":
            grad_z = self._backward_factorised(self.head, grad_out)
        else:
            grad_z = self._backward_factorised(self.head.add, grad_out) + self._backward_factorised(self.head.res, grad_out)
        self.unmix.backward(grad_z)

    @staticmethod
    def _backward_factorised(fm: FactorisedPropertyModel, grad_out: np.ndarray) -> np.ndarray:
        g = fm.head.backward(grad_out)
        grad_cols = []
        for k, enc in enumerate(fm.encoders):
            gk = g[:, k * fm.emb_dim:(k + 1) * fm.emb_dim] if fm.aggregation == "concat" else g
            grad_cols.append(enc.backward(gk))
        grad_z_norm = np.concatenate(grad_cols, axis=1)
        return grad_z_norm / fm.normalizer.std if fm.normalizer.is_fit else grad_z_norm

    def zero_grad(self):
        self.unmix.zero_grad()
        self.head.zero_grad()

    def get_params(self):
        head_params = [(f"head_{n}", p, g) for n, p, g in self.head.get_params()]
        if self.freeze_unmix:
            return head_params
        return self.unmix.get_params() + head_params

    def state_arrays(self):
        return {n: p.copy() for n, p, _ in (self.unmix.get_params() + [(f"head_{n}", p, g) for n, p, g in self.head.get_params()])}


class PCAPropertyModel:
    """Statistical feature extraction baseline: unsupervised top-K PCA on raw observations
    followed by a strictly additive per-component model."""
    def __init__(self, raw_dim: int, n_components: int = 3, seed: int = 0):
        self.raw_dim = int(raw_dim)
        self.n_components = int(n_components)
        self.mean = np.zeros((1, raw_dim), dtype=float)
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((raw_dim, n_components)))
        self.V_pca = Q[:, :n_components].copy()
        self.head = FactorisedPropertyModel(n_components, "sum", 16, 8, (), seed)
        self.input_dim = raw_dim

    @property
    def num_parameters(self) -> int:
        return self.head.num_parameters

    def fit_normalizer(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.mean = np.mean(X, axis=0, keepdims=True)
        Xc = X - self.mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.V_pca = Vt[:self.n_components].T.copy()
        Z = Xc @ self.V_pca
        self.head.fit_normalizer(Z)

    def get_latent_state(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) @ self.V_pca

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.head.forward(self.get_latent_state(X))

    def backward(self, grad_out: np.ndarray):
        self.head.backward(grad_out)

    def zero_grad(self):
        self.head.zero_grad()

    def get_params(self):
        return self.head.get_params()

    def state_arrays(self):
        return self.head.state_arrays()


class FrozenFeatureReadoutModel:
    """Baseline model for 'random_repr' (frozen random linear projection to K dims) and
    'untrained_encoder' (frozen random MLP hidden layer of dim 16), with a trainable linear readout."""
    def __init__(self, raw_dim: int, mode: str = "random_repr", feat_dim: int = 3, seed: int = 0):
        assert mode in ("random_repr", "untrained_encoder")
        self.raw_dim = int(raw_dim)
        self.mode = mode
        self.feat_dim = int(feat_dim)
        self.normalizer = FeatureNormalizer()
        rng = np.random.default_rng(seed)
        if mode == "random_repr":
            W = rng.standard_normal((raw_dim, feat_dim)) / np.sqrt(raw_dim)
            self.proj_W = W
            self.encoder_mlp = None
            out_dim = feat_dim
        else:
            self.proj_W = None
            self.encoder_mlp = PredictiveMLP(input_dim=raw_dim, hidden_dims=(32, 16), output_dim=1, seed=seed)
            out_dim = 16
        self.readout = LinearLayer(out_dim, 1, init_method="xavier", seed=seed + 50)
        self.input_dim = raw_dim

    @property
    def num_parameters(self) -> int:
        return int(self.readout.weights.size + self.readout.bias.size)

    def fit_normalizer(self, X: np.ndarray):
        self.normalizer.fit(X)
        if self.encoder_mlp is not None:
            self.encoder_mlp.fit_normalizer(X)

    def get_latent_state(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mode == "random_repr":
            Xn = self.normalizer.transform(X) if self.normalizer.is_fit else X
            return Xn @ self.proj_W
        return self.encoder_mlp.get_latent_state(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = self.get_latent_state(X)
        return self.readout.forward(Z)

    def backward(self, grad_out: np.ndarray):
        self.readout.backward(grad_out)

    def zero_grad(self):
        self.readout.zero_grad()

    def get_params(self):
        return [("readout_W", self.readout.weights, self.readout.grad_weights),
                ("readout_b", self.readout.bias, self.readout.grad_bias)]

    def state_arrays(self):
        return {n: p.copy() for n, p, _ in self.get_params()}


def build_discovery_model(name: str, raw_dim: int, seed: int, n_slots: int = 6):
    """Registry of Phase 13 architectures operating on raw entangled observations (dim=raw_dim)."""
    if name == "latent_additive":
        return LatentDiscoveryModel(raw_dim, n_slots=n_slots, mode="additive", seed=seed)
    if name == "latent_residual":
        return LatentDiscoveryModel(raw_dim, n_slots=n_slots, mode="residual", seed=seed)
    if name == "raw_naive_additive":
        return FactorisedPropertyModel(raw_dim, "sum", 16, 8, (), seed)
    if name == "pca_statistical":
        return PCAPropertyModel(raw_dim, n_components=3, seed=seed)
    if name == "untrained_encoder":
        return FrozenFeatureReadoutModel(raw_dim, mode="untrained_encoder", feat_dim=16, seed=seed)
    if name == "random_repr":
        return FrozenFeatureReadoutModel(raw_dim, mode="random_repr", feat_dim=3, seed=seed)
    if name == "raw_mlp_small":
        return PredictiveMLP(input_dim=raw_dim, hidden_dims=(32, 16), output_dim=1, seed=seed)
    if name == "raw_mlp_large":
        return PredictiveMLP(input_dim=raw_dim, hidden_dims=(64, 32), output_dim=1, seed=seed)
    raise ValueError(f"Unknown Phase 13 architecture: {name}")
