"""Optimizers and loss functions for Project Caspian.

Implements SGD (with momentum and weight decay), Adam, and loss functions (MSE, Huber)
derived directly from mathematical optimization primitives.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class LossFunction(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the scalar loss value."""
        pass

    @abstractmethod
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient dL/dy_pred with respect to predictions."""
        pass

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)


class MSELoss(LossFunction):
    """Mean Squared Error Loss: L = (1/N) * sum((y_pred - y_true)^2)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_p = np.asarray(y_pred, dtype=np.float64)
        y_t = np.asarray(y_true, dtype=np.float64)
        return float(np.mean(np.square(y_p - y_t)))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_p = np.asarray(y_pred, dtype=np.float64)
        y_t = np.asarray(y_true, dtype=np.float64)
        n = y_p.size
        if n == 0:
            return np.zeros_like(y_p)
        return (2.0 / n) * (y_p - y_t)


class HuberLoss(LossFunction):
    """Huber Loss (Smooth L1): robust to large outliers."""

    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_p = np.asarray(y_pred, dtype=np.float64)
        y_t = np.asarray(y_true, dtype=np.float64)
        error = y_p - y_t
        abs_error = np.abs(error)
        linear_mask = abs_error > self.delta
        quadratic = 0.5 * np.square(error)
        linear = self.delta * (abs_error - 0.5 * self.delta)
        loss = np.where(linear_mask, linear, quadratic)
        return float(np.mean(loss))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_p = np.asarray(y_pred, dtype=np.float64)
        y_t = np.asarray(y_true, dtype=np.float64)
        error = y_p - y_t
        abs_error = np.abs(error)
        n = y_p.size
        if n == 0:
            return np.zeros_like(y_p)
        grad = np.where(abs_error <= self.delta, error, self.delta * np.sign(error))
        return grad / n


class Optimizer(ABC):
    """Abstract base class for parameter optimizers."""

    def __init__(self, params: List[Tuple[str, np.ndarray, np.ndarray]], lr: float = 0.01):
        self.params = params
        self.lr = float(lr)

    @abstractmethod
    def step(self) -> None:
        """Perform a single optimization update step."""
        pass

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero."""
        for _, _, grad in self.params:
            grad.fill(0.0)


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum and L2 weight decay.

    Update equations:
        v_t = momentum * v_{t-1} + (grad + weight_decay * theta)
        theta_t = theta_{t-1} - lr * v_t
    """

    def __init__(
        self,
        params: List[Tuple[str, np.ndarray, np.ndarray]],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.velocities = [np.zeros_like(p) for _, p, _ in self.params]

    def step(self) -> None:
        for i, (name, param, grad) in enumerate(self.params):
            g = grad.copy()
            if self.weight_decay > 0.0 and name == "weight":
                g += self.weight_decay * param

            if self.momentum > 0.0:
                self.velocities[i] = self.momentum * self.velocities[i] + g
                update = self.velocities[i]
            else:
                update = g

            param -= self.lr * update


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) Optimizer.

    Update equations:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g
        v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: List[Tuple[str, np.ndarray, np.ndarray]],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p) for _, p, _ in self.params]
        self.v = [np.zeros_like(p) for _, p, _ in self.params]

    def step(self) -> None:
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps

        # Bias correction terms
        bias_correction1 = 1.0 - (b1 ** self.t)
        bias_correction2 = 1.0 - (b2 ** self.t)

        for i, (name, param, grad) in enumerate(self.params):
            g = grad.copy()
            if self.weight_decay > 0.0 and name == "weight":
                g += self.weight_decay * param

            # Update biased 1st and 2nd moment estimates
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * np.square(g)

            # Compute bias-corrected estimates
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            # Apply parameter update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
