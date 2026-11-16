"""Architectures for Phase 11 (architectural inductive bias study).

All models expose the interface PredictiveTrainer needs (forward/backward/get_params/zero_grad/fit_normalizer/num_parameters)
and are built by composing PredictiveMLP modules, so no new backprop code is introduced.

ASSUMPTIONS (documented, not discovered): input columns are property channels, one slot per channel; the slot index is
given to the model by construction.
  - 'factorised': one encoder per channel; embeddings are CONCATENATED and passed to a shared MLP head.
        Modular, but the head can learn any interaction between properties.
  - 'additive': one encoder per channel; embeddings are SUMMED; the prediction head is linear (strictly additive overall).
  - 'additive_mlp': embeddings SUMMED; prediction head is a small MLP (additive in the embedding space only).
"""
from typing import List, Tuple, Optional
import numpy as np
from models.mlp import PredictiveMLP
from models.normalization import FeatureNormalizer


class FactorisedPropertyModel:
    def __init__(self, n_props: int, aggregation: str = "concat", enc_hidden: int = 16, emb_dim: int = 8,
                 head_hidden: Tuple[int, ...] = (16,), seed: int = 0):
        assert aggregation in ("concat", "sum")
        self.n_props, self.aggregation, self.emb_dim = int(n_props), aggregation, int(emb_dim)
        rng = np.random.default_rng(seed)
        s = lambda: int(rng.integers(0, 2**31 - 1))
        self.encoders = [PredictiveMLP(input_dim=1, hidden_dims=(enc_hidden,), output_dim=emb_dim, seed=s()) for _ in range(n_props)]
        agg_dim = emb_dim * n_props if aggregation == "concat" else emb_dim
        self.head = PredictiveMLP(input_dim=agg_dim, hidden_dims=tuple(head_hidden), output_dim=1, seed=s())
        self.normalizer = FeatureNormalizer()
        self.input_dim = n_props
        self._z = None

    @property
    def num_parameters(self) -> int:
        return sum(m.num_parameters for m in self.encoders) + self.head.num_parameters

    def fit_normalizer(self, X):
        self.normalizer.fit(X)

    def _norm(self, x):
        return self.normalizer.transform(x) if self.normalizer.is_fit else x

    def embeddings(self, x):
        x = self._norm(x)
        return [enc.forward(x[:, k:k + 1]) for k, enc in enumerate(self.encoders)]

    def forward(self, x):
        z = self.embeddings(x)
        agg = np.concatenate(z, axis=1) if self.aggregation == "concat" else np.sum(z, axis=0)
        return self.head.forward(agg)

    def get_latent_state(self, x):
        z = self.embeddings(x)
        return np.concatenate(z, axis=1) if self.aggregation == "concat" else np.sum(z, axis=0)

    def backward(self, grad_out):
        g = self.head.backward(grad_out)
        for k, enc in enumerate(self.encoders):
            gk = g[:, k * self.emb_dim:(k + 1) * self.emb_dim] if self.aggregation == "concat" else g
            enc.backward(gk)

    def zero_grad(self):
        for m in self.encoders + [self.head]:
            m.zero_grad()

    def get_params(self):
        out = []
        for k, enc in enumerate(self.encoders):
            out += [(f"enc{k}_{n}", p, g) for n, p, g in enc.get_params()]
        out += [(f"head_{n}", p, g) for n, p, g in self.head.get_params()]
        return out

    def state_arrays(self):
        return {n: p.copy() for n, p, _ in self.get_params()}


class AdditivePlusResidual:
    """Phase 12: strictly additive model PLUS a concat-factorised residual model; the output is their sum.
    Can represent both additive and interactive rules. Built by composing two FactorisedPropertyModel objects."""
    def __init__(self, n_props: int, seed: int = 0):
        self.add = FactorisedPropertyModel(n_props, "sum", 16, 8, (), seed)
        self.res = FactorisedPropertyModel(n_props, "concat", 16, 8, (16,), seed + 1)
        self.input_dim = n_props

    @property
    def num_parameters(self) -> int:
        return self.add.num_parameters + self.res.num_parameters

    def fit_normalizer(self, X):
        self.add.fit_normalizer(X); self.res.fit_normalizer(X)

    def forward(self, x):
        return self.add.forward(x) + self.res.forward(x)

    def get_latent_state(self, x):
        return np.concatenate([self.add.get_latent_state(x), self.res.get_latent_state(x)], axis=1)

    def backward(self, grad_out):
        self.add.backward(grad_out); self.res.backward(grad_out)

    def zero_grad(self):
        self.add.zero_grad(); self.res.zero_grad()

    def get_params(self):
        return [(f"add_{n}", p, g) for n, p, g in self.add.get_params()] + [(f"res_{n}", p, g) for n, p, g in self.res.get_params()]

    def state_arrays(self):
        return {n: p.copy() for n, p, _ in self.get_params()}


def build_model(name: str, n_props: int, seed: int):
    """Registry of Phase 11 architectures operating on the property-channel vector."""
    if name == "mlp_small":        # (32,16) -- matches the Phase 10 hidden sizes
        return PredictiveMLP(input_dim=n_props, hidden_dims=(32, 16), output_dim=1, seed=seed)
    if name == "mlp_large":        # capacity control
        return PredictiveMLP(input_dim=n_props, hidden_dims=(64, 32), output_dim=1, seed=seed)
    if name == "factorised":
        return FactorisedPropertyModel(n_props, "concat", 16, 8, (16,), seed)
    if name == "additive":
        return FactorisedPropertyModel(n_props, "sum", 16, 8, (), seed)
    if name == "additive_mlp":
        return FactorisedPropertyModel(n_props, "sum", 16, 8, (16,), seed)
    if name == "additive_residual":
        return AdditivePlusResidual(n_props, seed)
    raise ValueError(name)
