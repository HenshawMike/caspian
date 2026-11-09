r"""Permutation-invariant (DeepSets-style) predictive model for Project Caspian.

The fixed-slot encoding used by PredictiveMLP assigns each entity to a specific
input position. CSP-P3-E003 showed that even after fixing slot *assignment*
(nearest-first instead of entity_id order), the model still degrades
substantially when more entities are present simultaneously than it saw during
training — because the model was trained to read from "slot 0" specifically,
not to treat entities as an unordered set.

This module implements the standard DeepSets construction (Zaheer et al.),
which is invariant to entity order and robust to entity *count* by
construction rather than by hoping training positions transfer:

    e_i = phi(entity_features_i)                for each entity i present
    pooled = sum_i mask_i * e_i                  (permutation-invariant, count-robust)
    y_hat = rho(concat(global_features, pooled))

phi and rho are small 2-layer MLPs built from the existing LinearLayer /
activation primitives, with a manual (analytically-derived) backward pass,
matching the rest of the from-scratch model stack.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from models.layers import LinearLayer
from models.activations import get_activation
from models.normalization import FeatureNormalizer


class DeepSetsPredictiveModel:
    """Permutation-invariant predictive model: phi (per-entity) + sum-pool + rho (head)."""

    def __init__(
        self,
        entity_feature_dim: int = 4,
        entity_hidden_dim: int = 16,
        global_feature_dim: int = 12,
        head_hidden_dim: int = 24,
        output_dim: int = 1,
        max_entities: int = 5,
        seed: Optional[int] = None,
    ):
        self.entity_feature_dim = int(entity_feature_dim)
        self.entity_hidden_dim = int(entity_hidden_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.output_dim = int(output_dim)
        self.max_entities = int(max_entities)
        self.seed = seed

        rng = np.random.default_rng(seed)

        def next_seed() -> Optional[int]:
            return int(rng.integers(0, 2**31 - 1)) if seed is not None else None

        # phi: shared per-entity encoder.
        self.entity_l1 = LinearLayer(self.entity_feature_dim, self.entity_hidden_dim, seed=next_seed())
        self.entity_l2 = LinearLayer(self.entity_hidden_dim, self.entity_hidden_dim, seed=next_seed())

        # rho: prediction head over [global_features ; pooled entity embedding].
        self.head_l1 = LinearLayer(self.global_feature_dim + self.entity_hidden_dim, self.head_hidden_dim, seed=next_seed())
        self.head_l2 = LinearLayer(self.head_hidden_dim, self.output_dim, seed=next_seed())

        self.relu = get_activation("relu")
        self._cache: Dict[str, Any] = {}
        self.entity_normalizer = FeatureNormalizer()
        self.global_normalizer = FeatureNormalizer()

    @property
    def num_parameters(self) -> int:
        total = 0
        for layer in (self.entity_l1, self.entity_l2, self.head_l1, self.head_l2):
            total += layer.weights.size + (layer.bias.size if layer.use_bias else 0)
        return total

    def fit_normalizer(self, entity_features: np.ndarray, presence_mask: np.ndarray, global_features: np.ndarray) -> None:
        """Fit entity/global normalizers from training data. Entity stats are computed
        from PRESENT slots only -- most slots are zero-padding (max_entities=5, usually
        1-2 entities actually present), which would otherwise skew the mean toward zero
        and understate the std of real entity features. Called automatically by
        DeepSetsTrainer at the start of fit().
        """
        flat_ef = entity_features.reshape(-1, entity_features.shape[-1])
        flat_mask = presence_mask.reshape(-1)
        present_ef = flat_ef[flat_mask > 0]
        if len(present_ef) > 0:
            self.entity_normalizer.fit(present_ef)
        self.global_normalizer.fit(global_features)

    def forward(self, entity_features: np.ndarray, presence_mask: np.ndarray, global_features: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            entity_features: Shape (N, max_entities, entity_feature_dim).
            presence_mask: Shape (N, max_entities); 1.0 where a real entity occupies that slot, else 0.0.
            global_features: Shape (N, global_feature_dim).

        Returns:
            np.ndarray: Predictions of shape (N, output_dim).
        """
        N, M, F = entity_features.shape
        if M != self.max_entities or F != self.entity_feature_dim:
            raise ValueError(f"Expected entity_features shape (N, {self.max_entities}, {self.entity_feature_dim}), got {entity_features.shape}")

        if self.entity_normalizer.is_fit:
            entity_features = self.entity_normalizer.transform(entity_features.reshape(-1, F)).reshape(N, M, F)
        if self.global_normalizer.is_fit:
            global_features = self.global_normalizer.transform(global_features)

        flat_ent = entity_features.reshape(N * M, F)
        z1 = self.entity_l1.forward(flat_ent)
        a1 = self.relu.forward(z1)
        z2 = self.entity_l2.forward(a1)
        a2 = self.relu.forward(z2)  # (N*M, entity_hidden_dim)

        ent_emb = a2.reshape(N, M, self.entity_hidden_dim)

        # Max-pool (not mean-pool) over present entities. Mean-pooling keeps the pooled
        # vector's *scale* stable across entity counts, but averaging two genuinely
        # different entity embeddings still lands at a point in embedding space the head
        # never saw during single-entity training -- the head has zero exposure to "the
        # combination of two things," no matter how much single-entity data it trains on.
        # Max-pooling (per-channel winner-take-all) instead always hands the head exactly
        # one entity's signal -- whichever is strongest per channel -- which is precisely
        # the kind of input the head was trained on, regardless of how many entities are
        # actually present.
        NEG_INF = -1e18
        masked_emb = np.where(presence_mask[:, :, None] > 0, ent_emb, NEG_INF)
        pooled = np.max(masked_emb, axis=1)  # (N, entity_hidden_dim)
        argmax_idx = np.argmax(masked_emb, axis=1)  # (N, entity_hidden_dim) -- winning slot per channel

        no_entity = (np.sum(presence_mask, axis=1) == 0)  # degenerate case: nothing present
        if np.any(no_entity):
            pooled[no_entity] = 0.0

        combined = np.concatenate([global_features, pooled], axis=1)
        hz1 = self.head_l1.forward(combined)
        ha1 = self.relu.forward(hz1)
        hz2 = self.head_l2.forward(ha1)  # identity output activation

        self._cache = {
            "N": N, "M": M, "argmax_idx": argmax_idx,
            "z1": z1, "a1": a1, "z2": z2, "a2": a2,
            "hz1": hz1, "ha1": ha1,
            "global_dim": global_features.shape[1],
        }
        return hz2

    def backward(self, grad_output: np.ndarray) -> None:
        """Full backward pass, accumulating gradients into all four LinearLayers."""
        c = self._cache
        if not c:
            raise RuntimeError("Must call forward() before backward().")

        dha1 = self.head_l2.backward(grad_output)
        dhz1 = dha1 * self.relu.derivative(c["hz1"], c["ha1"])
        dcombined = self.head_l1.backward(dhz1)  # (N, global_dim + entity_hidden_dim)

        dpooled = dcombined[:, c["global_dim"]:]  # (N, entity_hidden_dim)

        # Max-pooling gradient: for each (sample, channel), gradient flows only to the
        # entity slot that won the max in that channel -- a per-channel scatter, not a
        # broadcast like sum/mean pooling.
        N, M, H = c["N"], c["M"], self.entity_hidden_dim
        dent_emb = np.zeros((N, M, H), dtype=np.float64)
        n_idx = np.arange(N)[:, None]
        h_idx = np.arange(H)[None, :]
        dent_emb[n_idx, c["argmax_idx"], h_idx] = dpooled

        d_a2 = dent_emb.reshape(N * M, H)

        d_z2 = d_a2 * self.relu.derivative(c["z2"], c["a2"])
        d_a1 = self.entity_l2.backward(d_z2)
        d_z1 = d_a1 * self.relu.derivative(c["z1"], c["a1"])
        self.entity_l1.backward(d_z1)

    def zero_grad(self) -> None:
        for layer in (self.entity_l1, self.entity_l2, self.head_l1, self.head_l2):
            layer.zero_grad()

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        params = []
        for prefix, layer in (
            ("entity_l1", self.entity_l1), ("entity_l2", self.entity_l2),
            ("head_l1", self.head_l1), ("head_l2", self.head_l2),
        ):
            for name, p, g in layer.get_params():
                params.append((f"{prefix}_{name}", p, g))
        return params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_feature_dim": self.entity_feature_dim,
            "entity_hidden_dim": self.entity_hidden_dim,
            "global_feature_dim": self.global_feature_dim,
            "head_hidden_dim": self.head_hidden_dim,
            "output_dim": self.output_dim,
            "max_entities": self.max_entities,
            "num_parameters": self.num_parameters,
            "entity_l1": self.entity_l1.to_dict(),
            "entity_l2": self.entity_l2.to_dict(),
            "head_l1": self.head_l1.to_dict(),
            "head_l2": self.head_l2.to_dict(),
            "entity_normalizer": self.entity_normalizer.to_dict(),
            "global_normalizer": self.global_normalizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepSetsPredictiveModel":
        model = cls(
            entity_feature_dim=data["entity_feature_dim"],
            entity_hidden_dim=data["entity_hidden_dim"],
            global_feature_dim=data["global_feature_dim"],
            head_hidden_dim=data["head_hidden_dim"],
            output_dim=data["output_dim"],
            max_entities=data["max_entities"],
        )
        model.entity_l1 = LinearLayer.from_dict(data["entity_l1"])
        model.entity_l2 = LinearLayer.from_dict(data["entity_l2"])
        model.head_l1 = LinearLayer.from_dict(data["head_l1"])
        model.head_l2 = LinearLayer.from_dict(data["head_l2"])
        if data.get("entity_normalizer"):
            model.entity_normalizer = FeatureNormalizer.from_dict(data["entity_normalizer"])
        if data.get("global_normalizer"):
            model.global_normalizer = FeatureNormalizer.from_dict(data["global_normalizer"])
        return model

    def save_checkpoint(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {"model_state": self.to_dict(), "metadata": metadata or {}}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> Tuple["DeepSetsPredictiveModel", Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls.from_dict(payload["model_state"])
        return model, payload.get("metadata", {})
