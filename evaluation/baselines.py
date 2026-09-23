"""Baseline predictor implementations for Project Caspian (Phase 1).

Provides trivial and reference prediction models to assess whether learned models
demonstrate genuine environmental understanding.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any
import numpy as np


class BasePredictor(ABC):
    """Abstract base class for all outcome predictors."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes given input feature matrix X of shape (N, input_dim).

        Args:
            X: Input feature array.

        Returns:
            np.ndarray: Predicted values of shape (N, 1) or (N, output_dim).
        """
        pass


class PersistencePredictor(BasePredictor):
    """Persistence Baseline: predicts zero state variable change (Delta E = 0.0).

    Assumes the environment state remains completely static.
    """

    def __init__(self, output_dim: int = 1, default_value: float = 0.0):
        self.output_dim = output_dim
        self.default_value = float(default_value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return np.full((self.output_dim,), self.default_value, dtype=np.float64)
        elif X.ndim == 2:
            return np.full((X.shape[0], self.output_dim), self.default_value, dtype=np.float64)
        elif X.ndim == 3:
            B, T, _ = X.shape
            return np.full((B, T, self.output_dim), self.default_value, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported input dimension: {X.ndim}")


class RandomPredictor(BasePredictor):
    """Random Baseline: predicts stochastic values within a configured interval."""

    def __init__(
        self,
        output_dim: int = 1,
        low: float = -1.0,
        high: float = 10.0,
        seed: Optional[int] = 42,
    ):
        self.output_dim = output_dim
        self.low = float(low)
        self.high = float(high)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return self.rng.uniform(self.low, self.high, (self.output_dim,))
        elif X.ndim == 2:
            return self.rng.uniform(self.low, self.high, (X.shape[0], self.output_dim))
        elif X.ndim == 3:
            B, T, _ = X.shape
            return self.rng.uniform(self.low, self.high, (B, T, self.output_dim))
        else:
            raise ValueError(f"Unsupported input dimension: {X.ndim}")


class ReactivePredictor(BasePredictor):
    """Reactive / Empirical Mean Baseline: predicts the empirical mean delta from training data."""

    def __init__(self, output_dim: int = 1, mean_value: float = -1.0):
        self.output_dim = output_dim
        self.mean_value = float(mean_value)

    def fit(self, Y_train: np.ndarray) -> None:
        """Fit empirical mean from training target values."""
        self.mean_value = float(np.mean(Y_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return np.full((self.output_dim,), self.mean_value, dtype=np.float64)
        elif X.ndim == 2:
            return np.full((X.shape[0], self.output_dim), self.mean_value, dtype=np.float64)
        elif X.ndim == 3:
            B, T, _ = X.shape
            return np.full((B, T, self.output_dim), self.mean_value, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported input dimension: {X.ndim}")


class FeedForwardBaseline(BasePredictor):
    """Model A — No-Memory Baseline.

    A feed-forward model (PredictiveMLP) using only the current observation and current action:
    y_hat_t = f_theta(O_t, A_t).
    Has zero persistent internal state or explicit memory across time steps.
    """

    def __init__(self, model: Any):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes for 2D flat or 3D sequence tensors independently per timestep."""
        if X.ndim == 1 or X.ndim == 2:
            return self.model.forward(X)
        elif X.ndim == 3:
            B, T, D = X.shape
            X_flat = X.reshape(B * T, D)
            y_flat = self.model.forward(X_flat)
            return y_flat.reshape(B, T, -1)
        else:
            raise ValueError(f"Unsupported input dimension: {X.ndim}")
