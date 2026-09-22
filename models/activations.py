"""Mathematical activation functions and analytical derivatives for Project Caspian.

Implemented using NumPy vector and matrix operations.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class Activation(ABC):
    """Abstract base class for activation functions."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the activation output given input tensor x."""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray, output: np.ndarray) -> np.ndarray:
        """Compute the local derivative df/dx given input x and/or forward output."""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class ReLU(Activation):
    """Rectified Linear Unit: f(x) = max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray, output: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, 1.0, 0.0)


class LeakyReLU(Activation):
    """Leaky Rectified Linear Unit: f(x) = x if x >= 0 else alpha * x."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = float(alpha)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, x, self.alpha * x)

    def derivative(self, x: np.ndarray, output: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, 1.0, self.alpha)


class Sigmoid(Activation):
    """Logistic Sigmoid: f(x) = 1 / (1 + exp(-x))."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow/underflow in exp
        clipped_x = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped_x))

    def derivative(self, x: np.ndarray, output: np.ndarray) -> np.ndarray:
        return output * (1.0 - output)


class Tanh(Activation):
    """Hyperbolic Tangent: f(x) = tanh(x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray, output: np.ndarray) -> np.ndarray:
        return 1.0 - np.square(output)


class Identity(Activation):
    """Identity / Linear Activation: f(x) = x."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.copy()

    def derivative(self, x: np.ndarray, output: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


def get_activation(name: Union[str, Activation]) -> Activation:
    """Factory helper to obtain an Activation instance from a name string."""
    if isinstance(name, Activation):
        return name
    name_clean = name.strip().lower()
    if name_clean == "relu":
        return ReLU()
    elif name_clean == "leaky_relu" or name_clean == "leakyrelu":
        return LeakyReLU()
    elif name_clean == "sigmoid":
        return Sigmoid()
    elif name_clean == "tanh":
        return Tanh()
    elif name_clean in ("identity", "linear", "none"):
        return Identity()
    raise ValueError(f"Unknown activation function: {name}")
