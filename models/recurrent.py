"""Recurrent Neural Network primitives and GRU memory models for Project Caspian (Phase 2).

Implements Gated Recurrent Unit (GRU) cells and RecurrentPredictor networks with analytical
Backpropagation Through Time (BPTT), hidden state caching, serialization, and parameter tracking.
"""

import json
import os
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from models.layers import LinearLayer
from models.activations import Sigmoid, Tanh, get_activation


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def _tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent function."""
    return np.tanh(x)


class GRULayer:
    """Gated Recurrent Unit (GRU) recurrent neural network layer.

    Equations:
        z_t = sigmoid(x_t @ W_z + h_{t-1} @ U_z + b_z)  (Update Gate)
        r_t = sigmoid(x_t @ W_r + h_{t-1} @ U_r + b_r)  (Reset Gate)
        h_reset_t = r_t * h_{t-1}
        h_tilde_t = tanh(x_t @ W_h + h_reset_t @ U_h + b_h)  (Candidate Hidden State)
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t  (Updated Hidden State)

    Supports sequence processing of shape (batch_size, seq_len, in_features)
    and full analytical Backpropagation Through Time (BPTT).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        init_method: str = "xavier",
        seed: Optional[int] = None,
    ):
        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.init_method = init_method.lower()
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Initialize input weights W_z, W_r, W_h
        if self.init_method == "xavier" or self.init_method == "glorot":
            lim_in = np.sqrt(6.0 / (self.in_features + self.hidden_dim))
            lim_hid = np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim))
            self.W_z = rng.uniform(-lim_in, lim_in, (self.in_features, self.hidden_dim))
            self.W_r = rng.uniform(-lim_in, lim_in, (self.in_features, self.hidden_dim))
            self.W_h = rng.uniform(-lim_in, lim_in, (self.in_features, self.hidden_dim))
            self.U_z = rng.uniform(-lim_hid, lim_hid, (self.hidden_dim, self.hidden_dim))
            self.U_r = rng.uniform(-lim_hid, lim_hid, (self.hidden_dim, self.hidden_dim))
            self.U_h = rng.uniform(-lim_hid, lim_hid, (self.hidden_dim, self.hidden_dim))
        else:
            std = np.sqrt(2.0 / self.in_features)
            self.W_z = rng.normal(0.0, std, (self.in_features, self.hidden_dim))
            self.W_r = rng.normal(0.0, std, (self.in_features, self.hidden_dim))
            self.W_h = rng.normal(0.0, std, (self.in_features, self.hidden_dim))
            std_h = np.sqrt(2.0 / self.hidden_dim)
            self.U_z = rng.normal(0.0, std_h, (self.hidden_dim, self.hidden_dim))
            self.U_r = rng.normal(0.0, std_h, (self.hidden_dim, self.hidden_dim))
            self.U_h = rng.normal(0.0, std_h, (self.hidden_dim, self.hidden_dim))

        self.b_z = np.zeros((self.hidden_dim,), dtype=np.float64)
        self.b_r = np.zeros((self.hidden_dim,), dtype=np.float64)
        self.b_h = np.zeros((self.hidden_dim,), dtype=np.float64)

        # Gradients
        self.grad_W_z = np.zeros_like(self.W_z)
        self.grad_W_r = np.zeros_like(self.W_r)
        self.grad_W_h = np.zeros_like(self.W_h)
        self.grad_U_z = np.zeros_like(self.U_z)
        self.grad_U_r = np.zeros_like(self.U_r)
        self.grad_U_h = np.zeros_like(self.U_h)
        self.grad_b_z = np.zeros_like(self.b_z)
        self.grad_b_r = np.zeros_like(self.b_r)
        self.grad_b_h = np.zeros_like(self.b_h)

        # Forward caches for BPTT
        self._cached_inputs: Optional[np.ndarray] = None  # (B, T, D_in)
        self._cached_h: Optional[List[np.ndarray]] = None  # length T+1, each (B, D_h)
        self._cached_z: Optional[List[np.ndarray]] = None  # length T
        self._cached_r: Optional[List[np.ndarray]] = None  # length T
        self._cached_h_tilde: Optional[List[np.ndarray]] = None  # length T
        self._cached_h_reset: Optional[List[np.ndarray]] = None  # length T

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute single-step GRU recurrence: (x_t, h_{t-1}) -> h_t.

        Args:
            x_t: Input vector of shape (batch_size, in_features).
            h_prev: Previous hidden state of shape (batch_size, hidden_dim).

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: (h_t, step_cache)
        """
        # Update gate
        alpha_z = np.matmul(x_t, self.W_z) + np.matmul(h_prev, self.U_z) + self.b_z
        z_t = _sigmoid(alpha_z)

        # Reset gate
        alpha_r = np.matmul(x_t, self.W_r) + np.matmul(h_prev, self.U_r) + self.b_r
        r_t = _sigmoid(alpha_r)

        # Candidate state
        h_reset = r_t * h_prev
        alpha_h = np.matmul(x_t, self.W_h) + np.matmul(h_reset, self.U_h) + self.b_h
        h_tilde = _tanh(alpha_h)

        # New hidden state
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde

        cache = {
            "z": z_t,
            "r": r_t,
            "h_reset": h_reset,
            "h_tilde": h_tilde,
        }
        return h_t, cache

    def forward(
        self,
        x_seq: np.ndarray,
        h_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass over sequence: x_seq -> h_seq.

        Args:
            x_seq: Input sequence array of shape (batch_size, seq_len, in_features)
                   or (seq_len, in_features).
            h_init: Optional initial hidden state of shape (batch_size, hidden_dim)
                    or (hidden_dim,). If None, defaults to zeros.

        Returns:
            np.ndarray: Sequence of hidden states of shape (batch_size, seq_len, hidden_dim)
                        or (seq_len, hidden_dim).
        """
        is_2d = (x_seq.ndim == 2)
        x_3d = x_seq.reshape(1, x_seq.shape[0], x_seq.shape[1]) if is_2d else x_seq
        B, T, D = x_3d.shape

        if D != self.in_features:
            raise ValueError(f"Input feature dim mismatch: expected {self.in_features}, got {D}")

        if h_init is not None:
            h_0 = h_init.reshape(1, -1) if h_init.ndim == 1 else h_init
        else:
            h_0 = np.zeros((B, self.hidden_dim), dtype=np.float64)

        self._cached_inputs = x_3d
        self._cached_h = [h_0]
        self._cached_z = []
        self._cached_r = []
        self._cached_h_tilde = []
        self._cached_h_reset = []

        h_curr = h_0
        for t in range(T):
            x_t = x_3d[:, t, :]
            h_next, step_cache = self.step(x_t, h_curr)

            self._cached_h.append(h_next)
            self._cached_z.append(step_cache["z"])
            self._cached_r.append(step_cache["r"])
            self._cached_h_reset.append(step_cache["h_reset"])
            self._cached_h_tilde.append(step_cache["h_tilde"])

            h_curr = h_next

        # Stack hidden states from t=1 to T: shape (B, T, D_h)
        h_seq = np.stack(self._cached_h[1:], axis=1)
        return h_seq.squeeze(0) if is_2d else h_seq

    def backward(self, grad_h_seq: np.ndarray) -> np.ndarray:
        """Perform analytical Backpropagation Through Time (BPTT).

        Args:
            grad_h_seq: Upstream gradient dL/dh of shape (B, T, D_h) or (T, D_h).

        Returns:
            np.ndarray: Gradient with respect to inputs dL/dx of shape (B, T, D_in) or (T, D_in).
        """
        if self._cached_inputs is None or self._cached_h is None:
            raise RuntimeError("Must call forward() before backward().")

        is_2d = (grad_h_seq.ndim == 2)
        grad_3d = grad_h_seq.reshape(1, grad_h_seq.shape[0], grad_h_seq.shape[1]) if is_2d else grad_h_seq
        B, T, _ = grad_3d.shape

        grad_inputs = np.zeros_like(self._cached_inputs)
        dh_next = np.zeros((B, self.hidden_dim), dtype=np.float64)

        for t in reversed(range(T)):
            x_t = self._cached_inputs[:, t, :]
            h_prev = self._cached_h[t]
            h_t = self._cached_h[t + 1]
            z_t = self._cached_z[t]
            r_t = self._cached_r[t]
            h_reset = self._cached_h_reset[t]
            h_tilde = self._cached_h_tilde[t]

            # Total upstream gradient into h_t
            dh_t = grad_3d[:, t, :] + dh_next

            # 1. Candidate hidden state gradients
            dh_tilde = dh_t * z_t
            dalpha_h = dh_tilde * (1.0 - h_tilde ** 2)

            # 2. Update gate gradients
            dz_t = dh_t * (h_tilde - h_prev)
            dalpha_z = dz_t * z_t * (1.0 - z_t)

            # 3. Reset gate gradients
            dh_reset = np.matmul(dalpha_h, self.U_h.T)
            dr_t = dh_reset * h_prev
            dalpha_r = dr_t * r_t * (1.0 - r_t)

            # 4. Hidden state at previous step gradient
            dh_prev = (
                dh_t * (1.0 - z_t)
                + np.matmul(dalpha_z, self.U_z.T)
                + np.matmul(dalpha_r, self.U_r.T)
                + (dh_reset * r_t)
            )
            dh_next = dh_prev

            # 5. Accumulate parameter gradients
            self.grad_W_z += np.matmul(x_t.T, dalpha_z)
            self.grad_U_z += np.matmul(h_prev.T, dalpha_z)
            self.grad_b_z += np.sum(dalpha_z, axis=0)

            self.grad_W_r += np.matmul(x_t.T, dalpha_r)
            self.grad_U_r += np.matmul(h_prev.T, dalpha_r)
            self.grad_b_r += np.sum(dalpha_r, axis=0)

            self.grad_W_h += np.matmul(x_t.T, dalpha_h)
            self.grad_U_h += np.matmul(h_reset.T, dalpha_h)
            self.grad_b_h += np.sum(dalpha_h, axis=0)

            # 6. Input gradient at step t
            dx_t = (
                np.matmul(dalpha_z, self.W_z.T)
                + np.matmul(dalpha_r, self.W_r.T)
                + np.matmul(dalpha_h, self.W_h.T)
            )
            grad_inputs[:, t, :] = dx_t

        return grad_inputs.squeeze(0) if is_2d else grad_inputs

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        self.grad_W_z.fill(0.0)
        self.grad_W_r.fill(0.0)
        self.grad_W_h.fill(0.0)
        self.grad_U_z.fill(0.0)
        self.grad_U_r.fill(0.0)
        self.grad_U_h.fill(0.0)
        self.grad_b_z.fill(0.0)
        self.grad_b_r.fill(0.0)
        self.grad_b_h.fill(0.0)

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Return list of (name, parameter_tensor, gradient_tensor) tuples."""
        return [
            ("W_z", self.W_z, self.grad_W_z),
            ("W_r", self.W_r, self.grad_W_r),
            ("W_h", self.W_h, self.grad_W_h),
            ("U_z", self.U_z, self.grad_U_z),
            ("U_r", self.U_r, self.grad_U_r),
            ("U_h", self.U_h, self.grad_U_h),
            ("b_z", self.b_z, self.grad_b_z),
            ("b_r", self.b_r, self.grad_b_r),
            ("b_h", self.b_h, self.grad_b_h),
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize layer weights and configuration."""
        return {
            "in_features": self.in_features,
            "hidden_dim": self.hidden_dim,
            "init_method": self.init_method,
            "W_z": self.W_z.tolist(),
            "W_r": self.W_r.tolist(),
            "W_h": self.W_h.tolist(),
            "U_z": self.U_z.tolist(),
            "U_r": self.U_r.tolist(),
            "U_h": self.U_h.tolist(),
            "b_z": self.b_z.tolist(),
            "b_r": self.b_r.tolist(),
            "b_h": self.b_h.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GRULayer":
        """Reconstruct GRU layer from serialized dictionary."""
        layer = cls(
            in_features=data["in_features"],
            hidden_dim=data["hidden_dim"],
            init_method=data.get("init_method", "xavier"),
        )
        layer.W_z = np.array(data["W_z"], dtype=np.float64)
        layer.W_r = np.array(data["W_r"], dtype=np.float64)
        layer.W_h = np.array(data["W_h"], dtype=np.float64)
        layer.U_z = np.array(data["U_z"], dtype=np.float64)
        layer.U_r = np.array(data["U_r"], dtype=np.float64)
        layer.U_h = np.array(data["U_h"], dtype=np.float64)
        layer.b_z = np.array(data["b_z"], dtype=np.float64)
        layer.b_r = np.array(data["b_r"], dtype=np.float64)
        layer.b_h = np.array(data["b_h"], dtype=np.float64)
        return layer


class RecurrentPredictor:
    """Predictive Recurrent Neural Network (Model B — Memory Model).

    Combines a GRULayer encoder with a Linear prediction head to forecast
    environmental consequences from sequential observation-action histories:
        S_t = GRU(S_{t-1}, x_t)
        y_hat_t = S_t @ W_head + b_head
    """

    def __init__(
        self,
        input_dim: int = 37,
        hidden_dim: int = 32,
        output_dim: int = 1,
        init_method: str = "xavier",
        seed: Optional[int] = None,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.init_method = str(init_method)
        self.seed = seed

        rng = np.random.default_rng(seed)
        layer_seed = int(rng.integers(0, 2**31 - 1)) if seed is not None else None
        head_seed = int(rng.integers(0, 2**31 - 1)) if seed is not None else None

        self.gru = GRULayer(
            in_features=self.input_dim,
            hidden_dim=self.hidden_dim,
            init_method=self.init_method,
            seed=layer_seed,
        )

        self.head = LinearLayer(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
            use_bias=True,
            init_method=self.init_method,
            seed=head_seed,
        )

        # Internal online hidden state for step-by-step agent inference
        self._online_hidden: Optional[np.ndarray] = None
        self.reset_state()

        # Cached tensors for sequence backpropagation
        self._cached_h_seq: Optional[np.ndarray] = None

    @property
    def num_parameters(self) -> int:
        """Compute total number of trainable parameters."""
        total = 0
        for _, p, _ in self.get_params():
            total += p.size
        return total

    def reset_state(self) -> None:
        """Reset internal online hidden state to zero vector (episode isolation)."""
        self._online_hidden = np.zeros((1, self.hidden_dim), dtype=np.float64)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        """Online single-step update and prediction: (x_t, S_{t-1}) -> (y_hat_t, S_t).

        Args:
            x_t: Feature vector of shape (input_dim,) or (1, input_dim).

        Returns:
            np.ndarray: Predicted consequence vector of shape (output_dim,).
        """
        x_2d = x_t.reshape(1, -1) if x_t.ndim == 1 else x_t
        if self._online_hidden is None:
            self.reset_state()

        h_t, _ = self.gru.step(x_2d, self._online_hidden)
        self._online_hidden = h_t

        y_hat = self.head.forward(h_t)
        return y_hat.squeeze(0)

    def get_hidden_state(self) -> np.ndarray:
        """Retrieve copy of current online latent memory state S_t."""
        if self._online_hidden is None:
            self.reset_state()
        return self._online_hidden.copy().squeeze(0)

    def forward(
        self,
        x_seq: np.ndarray,
        h_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass over full sequence: x_seq -> y_hat_seq.

        Args:
            x_seq: Sequence of inputs of shape (batch_size, seq_len, input_dim)
                   or (seq_len, input_dim).
            h_init: Optional initial hidden state.

        Returns:
            np.ndarray: Predictions of shape (batch_size, seq_len, output_dim)
                        or (seq_len, output_dim).
        """
        is_2d = (x_seq.ndim == 2)
        x_3d = x_seq.reshape(1, x_seq.shape[0], x_seq.shape[1]) if is_2d else x_seq
        B, T, _ = x_3d.shape

        h_seq = self.gru.forward(x_3d, h_init=h_init)  # (B, T, D_h)
        self._cached_h_seq = h_seq

        # Flatten sequence for linear head: (B * T, D_h) -> (B * T, D_out)
        h_flat = h_seq.reshape(B * T, self.hidden_dim)
        y_flat = self.head.forward(h_flat)
        y_seq = y_flat.reshape(B, T, self.output_dim)

        return y_seq.squeeze(0) if is_2d else y_seq

    def get_latent_state(self, x_seq: np.ndarray) -> np.ndarray:
        """Extract sequence of latent memory states S_t."""
        is_2d = (x_seq.ndim == 2)
        x_3d = x_seq.reshape(1, x_seq.shape[0], x_seq.shape[1]) if is_2d else x_seq
        h_seq = self.gru.forward(x_3d)
        return h_seq.squeeze(0) if is_2d else h_seq

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Perform full analytical BPTT through linear head and GRU.

        Args:
            grad_output: Upstream gradient of shape (B, T, D_out) or (T, D_out).

        Returns:
            np.ndarray: Downstream gradient dL/dx of shape (B, T, D_in) or (T, D_in).
        """
        if self._cached_h_seq is None:
            raise RuntimeError("Must call forward() before backward().")

        is_2d = (grad_output.ndim == 2)
        grad_3d = grad_output.reshape(1, grad_output.shape[0], grad_output.shape[1]) if is_2d else grad_output
        B, T, _ = grad_3d.shape

        # Backward through linear prediction head
        grad_flat = grad_3d.reshape(B * T, self.output_dim)
        grad_h_flat = self.head.backward(grad_flat)
        grad_h_seq = grad_h_flat.reshape(B, T, self.hidden_dim)

        # Backward through GRU cell over time
        grad_inputs = self.gru.backward(grad_h_seq)
        return grad_inputs.squeeze(0) if is_2d else grad_inputs

    def zero_grad(self) -> None:
        """Zero gradients across GRU and Head."""
        self.gru.zero_grad()
        self.head.zero_grad()

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Collect all model parameters and corresponding gradient tensors."""
        params = []
        for name, p, g in self.gru.get_params():
            params.append((f"gru_{name}", p, g))
        for name, p, g in self.head.get_params():
            params.append((f"head_{name}", p, g))
        return params

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full recurrent predictor architecture and weights."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "init_method": self.init_method,
            "num_parameters": self.num_parameters,
            "gru": self.gru.to_dict(),
            "head": self.head.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecurrentPredictor":
        """Reconstruct recurrent predictor from dictionary."""
        model = cls(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            output_dim=data["output_dim"],
            init_method=data.get("init_method", "xavier"),
        )
        model.gru = GRULayer.from_dict(data["gru"])
        model.head = LinearLayer.from_dict(data["head"])
        return model

    def save_checkpoint(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model_type": "RecurrentPredictor",
            "model_state": self.to_dict(),
            "metadata": metadata or {},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> Tuple["RecurrentPredictor", Dict[str, Any]]:
        """Load model checkpoint from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls.from_dict(payload["model_state"])
        return model, payload.get("metadata", {})
