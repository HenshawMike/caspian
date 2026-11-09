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
        is_1d = (X.ndim == 1)
        n_samples = 1 if is_1d else X.shape[0]
        preds = np.full((n_samples, self.output_dim), self.default_value, dtype=np.float64)
        return preds.squeeze(0) if is_1d else preds


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
        is_1d = (X.ndim == 1)
        n_samples = 1 if is_1d else X.shape[0]
        preds = self.rng.uniform(self.low, self.high, (n_samples, self.output_dim))
        return preds.squeeze(0) if is_1d else preds


class ReactivePredictor(BasePredictor):
    """Reactive / Empirical Mean Baseline: predicts the empirical mean delta from training data."""

    def __init__(self, output_dim: int = 1, mean_value: float = -1.0):
        self.output_dim = output_dim
        self.mean_value = float(mean_value)

    def fit(self, Y_train: np.ndarray) -> None:
        """Fit empirical mean from training target values."""
        self.mean_value = float(np.mean(Y_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        is_1d = (X.ndim == 1)
        n_samples = 1 if is_1d else X.shape[0]
        preds = np.full((n_samples, self.output_dim), self.mean_value, dtype=np.float64)
        return preds.squeeze(0) if is_1d else preds
