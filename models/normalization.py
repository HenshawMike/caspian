"""Feature normalization for Project Caspian's model stack.

Root-cause fix for a project-wide dead-ReLU / saturated-tanh pathology found
during Phase 5: every model trains on the raw, unnormalized feature vector, in
which internal_state (range 0-100, std ~12) sits alongside position/distance/
flag features (range ~0-5, std ~0.3-1.8). The huge-magnitude feature dominates
the pre-activation sum of the first layer, pushing most units into permanent
death (ReLU) or saturation (tanh) early in training, well before other
features get a fair chance to matter.

FeatureNormalizer applies a fixed (non-trainable) z-score transform, fit once
on training data, inside each model's forward() -- so every existing caller
across every phase (baselines, rollout, planning, evaluators) benefits
automatically without touching those call sites. Because it is a fixed affine
map applied before the first trainable layer, it needs no backward-pass
changes: gradients are never taken with respect to the raw input.
"""

from typing import Dict, Any, Optional
import numpy as np


class FeatureNormalizer:
    """Per-column z-score normalization: (x - mean) / std, fit from training data."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    @property
    def is_fit(self) -> bool:
        return self.mean is not None

    def fit(self, X: np.ndarray) -> None:
        """Compute per-column mean/std from X. Columns with ~zero variance (e.g. a
        one-hot slot that's constant in this dataset) are left unscaled (std=1) rather
        than blown up by dividing by a near-zero number.
        """
        self.mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-8] = 1.0
        self.std = std

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("FeatureNormalizer.transform() called before fit().")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureNormalizer":
        norm = cls()
        if data.get("mean") is not None:
            norm.mean = np.array(data["mean"], dtype=np.float64)
            norm.std = np.array(data["std"], dtype=np.float64)
        return norm
