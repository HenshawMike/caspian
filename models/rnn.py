r"""Recurrent memory-capable predictive model for Project Caspian (Phase 2).

Phase 1 established that a stateless MLP can learn Delta E = f(O_t, A_t) when the
current observation is always sufficient. Phase 2 asks whether Caspian can retain
information across timesteps when it is not (Section 29 of the Development Plan).

Per the project's "earn complexity" principle (Section 57), this module implements
the smallest architecture in the "recurrent networks" family suggested by the plan:
a single-layer Elman RNN with a linear read-out head, trained with full
backpropagation-through-time (BPTT). Gated memory (GRU/LSTM) is deliberately
deferred until evidence shows a plain recurrent state is insufficient.

Mathematics:
    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)      (recurrent state update)
    y_t = W_hy h_t + b_y                            (linear prediction head)

    with h_0 = 0 at the start of every episode.

BPTT (batched over episodes, full unroll length T):
    dL/dW_hy = sum_t h_t^T dL/dy_t
    dh_t     = dL/dy_t @ W_hy^T + dh_{t+1} @ W_hh^T   (dh_{T+1} := 0)
    dz_t     = dh_t * (1 - h_t^2)                     (tanh derivative)
    dL/dW_xh = sum_t x_t^T dz_t
    dL/dW_hh = sum_t h_{t-1}^T dz_t
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from models.activations import Tanh
from models.normalization import FeatureNormalizer


class RecurrentPredictiveModel:
    """Elman RNN + linear read-out, trained end-to-end with BPTT.

    Operates on full episode sequences: input of shape (batch, T, input_dim),
    output of shape (batch, T, output_dim). Exposes the same get_params() /
    zero_grad() / to_dict() / from_dict() / checkpoint interface as
    ``models.mlp.PredictiveMLP`` so it can be driven by the same Optimizer
    classes in ``models.optimizers``.
    """

    def __init__(
        self,
        input_dim: int = 37,
        hidden_dim: int = 24,
        output_dim: int = 1,
        seed: Optional[int] = None,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.seed = seed
        self._tanh = Tanh()
        self.normalizer = FeatureNormalizer()

        rng = np.random.default_rng(seed)

        # Xavier/Glorot initialization (appropriate for tanh nonlinearity).
        def xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_in, fan_out))

        self.W_xh = xavier(self.input_dim, self.hidden_dim)
        self.W_hh = xavier(self.hidden_dim, self.hidden_dim)
        self.b_h = np.zeros((self.hidden_dim,), dtype=np.float64)
        self.W_hy = xavier(self.hidden_dim, self.output_dim)
        self.b_y = np.zeros((self.output_dim,), dtype=np.float64)

        self.grad_W_xh = np.zeros_like(self.W_xh)
        self.grad_W_hh = np.zeros_like(self.W_hh)
        self.grad_b_h = np.zeros_like(self.b_h)
        self.grad_W_hy = np.zeros_like(self.W_hy)
        self.grad_b_y = np.zeros_like(self.b_y)

        # BPTT caches, populated by forward_sequence().
        self._cache_X: Optional[np.ndarray] = None
        self._cache_H: Optional[np.ndarray] = None  # (batch, T+1, hidden_dim), H[:,0,:] = h_0

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return (
            self.W_xh.size + self.W_hh.size + self.b_h.size
            + self.W_hy.size + self.b_y.size
        )

    def fit_normalizer(self, X: np.ndarray) -> None:
        """Fit the input feature normalizer from training data (flattened over batch and
        time). Called automatically by RecurrentTrainer at the start of fit().
        """
        flat = X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X
        self.normalizer.fit(flat)

    def forward_sequence(self, X: np.ndarray, h0: Optional[np.ndarray] = None) -> np.ndarray:
        """Run the recurrence over a full batch of episode sequences.

        Args:
            X: Input array of shape (batch, T, input_dim).
            h0: Optional initial hidden state of shape (batch, hidden_dim). Defaults
                to zeros (start of a new episode).

        Returns:
            np.ndarray: Predictions of shape (batch, T, output_dim).
        """
        batch, T, in_dim = X.shape
        if in_dim != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {in_dim}")

        if self.normalizer.is_fit:
            X = self.normalizer.transform(X.reshape(-1, in_dim)).reshape(batch, T, in_dim)

        H = np.zeros((batch, T + 1, self.hidden_dim), dtype=np.float64)
        if h0 is not None:
            H[:, 0, :] = h0

        Y = np.zeros((batch, T, self.output_dim), dtype=np.float64)
        for t in range(T):
            z_t = X[:, t, :] @ self.W_xh + H[:, t, :] @ self.W_hh + self.b_h
            h_t = self._tanh.forward(z_t)
            H[:, t + 1, :] = h_t
            Y[:, t, :] = h_t @ self.W_hy + self.b_y

        self._cache_X = X
        self._cache_H = H
        return Y

    def predict_step(self, x_t: np.ndarray, h_prev: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Single-timestep inference (no BPTT caching) for online/closed-loop use.

        Args:
            x_t: Input vector of shape (input_dim,) or (batch, input_dim).
            h_prev: Previous hidden state, shape matching x_t's batch dim, or None
                for h_prev = 0 (start of episode).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_t, h_t) — prediction and new hidden state,
                with the same leading batch/no-batch shape convention as x_t.
        """
        is_1d = (x_t.ndim == 1)
        x2d = x_t.reshape(1, -1) if is_1d else x_t
        h_prev2d = np.zeros((x2d.shape[0], self.hidden_dim)) if h_prev is None else (
            h_prev.reshape(1, -1) if is_1d else h_prev
        )

        if self.normalizer.is_fit:
            x2d = self.normalizer.transform(x2d)

        z_t = x2d @ self.W_xh + h_prev2d @ self.W_hh + self.b_h
        h_t = self._tanh.forward(z_t)
        y_t = h_t @ self.W_hy + self.b_y

        return (y_t.squeeze(0), h_t.squeeze(0)) if is_1d else (y_t, h_t)

    def get_hidden_states(self) -> np.ndarray:
        """Return cached hidden states h_1..h_T from the last forward_sequence() call.

        Returns:
            np.ndarray: Shape (batch, T, hidden_dim).
        """
        if self._cache_H is None:
            raise RuntimeError("Must call forward_sequence() before get_hidden_states().")
        return self._cache_H[:, 1:, :]

    def backward_sequence(self, grad_Y: np.ndarray) -> None:
        """Full BPTT: accumulate parameter gradients given dL/dY.

        Args:
            grad_Y: Upstream gradient of shape (batch, T, output_dim), matching the
                shape returned by forward_sequence().
        """
        if self._cache_X is None or self._cache_H is None:
            raise RuntimeError("Must call forward_sequence() before backward_sequence().")

        X = self._cache_X
        H = self._cache_H
        batch, T, _ = grad_Y.shape

        dh_next = np.zeros((batch, self.hidden_dim), dtype=np.float64)
        for t in reversed(range(T)):
            h_t = H[:, t + 1, :]
            h_prev = H[:, t, :]
            x_t = X[:, t, :]
            dy_t = grad_Y[:, t, :]

            # Read-out head gradients.
            self.grad_W_hy += h_t.T @ dy_t
            self.grad_b_y += np.sum(dy_t, axis=0)

            # dL/dh_t = dL/dy_t @ W_hy^T + dh_next (from the following timestep).
            dh_t = dy_t @ self.W_hy.T + dh_next

            # tanh derivative: 1 - h_t^2.
            dz_t = dh_t * self._tanh.derivative(None, h_t)

            self.grad_W_xh += x_t.T @ dz_t
            self.grad_W_hh += h_prev.T @ dz_t
            self.grad_b_h += np.sum(dz_t, axis=0)

            # Gradient flowing to h_{t-1}.
            dh_next = dz_t @ self.W_hh.T

    def zero_grad(self) -> None:
        """Reset all accumulated gradients to zero."""
        self.grad_W_xh.fill(0.0)
        self.grad_W_hh.fill(0.0)
        self.grad_b_h.fill(0.0)
        self.grad_W_hy.fill(0.0)
        self.grad_b_y.fill(0.0)

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Return (name, parameter, gradient) tuples for the Optimizer interface."""
        return [
            ("weight", self.W_xh, self.grad_W_xh),
            ("weight", self.W_hh, self.grad_W_hh),
            ("bias", self.b_h, self.grad_b_h),
            ("weight", self.W_hy, self.grad_W_hy),
            ("bias", self.b_y, self.grad_b_y),
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize architecture and weights."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_parameters": self.num_parameters,
            "W_xh": self.W_xh.tolist(),
            "W_hh": self.W_hh.tolist(),
            "b_h": self.b_h.tolist(),
            "W_hy": self.W_hy.tolist(),
            "b_y": self.b_y.tolist(),
            "normalizer": self.normalizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecurrentPredictiveModel":
        """Reconstruct model from serialized dictionary."""
        model = cls(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            output_dim=data["output_dim"],
        )
        model.W_xh = np.array(data["W_xh"], dtype=np.float64)
        model.W_hh = np.array(data["W_hh"], dtype=np.float64)
        model.b_h = np.array(data["b_h"], dtype=np.float64)
        model.W_hy = np.array(data["W_hy"], dtype=np.float64)
        model.b_y = np.array(data["b_y"], dtype=np.float64)
        if data.get("normalizer"):
            model.normalizer = FeatureNormalizer.from_dict(data["normalizer"])
        return model

    def save_checkpoint(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {"model_state": self.to_dict(), "metadata": metadata or {}}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> Tuple["RecurrentPredictiveModel", Dict[str, Any]]:
        """Load model and metadata from a JSON checkpoint file."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls.from_dict(payload["model_state"])
        return model, payload.get("metadata", {})
