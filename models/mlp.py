"""Predictive Multi-Layer Perceptron (MLP) for Project Caspian (Phase 1).

Implements the neural architecture described in Section 24 of the development plan:
Input (Observation + Action) -> Encoder Layers -> Internal State (S_t) -> Prediction Head -> Predicted Outcome.
"""

import json
import os
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from models.layers import LinearLayer
from models.activations import Activation, get_activation


class PredictiveMLP:
    """Predictive Multi-Layer Perceptron network.

    Maps state-action representations to predicted consequences while exposing
    internal latent state representations for analysis.
    """

    def __init__(
        self,
        input_dim: int = 37,
        hidden_dims: Tuple[int, ...] = (32, 16),
        output_dim: int = 1,
        hidden_activation: str = "relu",
        output_activation: str = "identity",
        use_bias: bool = True,
        init_method: str = "he",
        seed: Optional[int] = None,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.output_dim = int(output_dim)
        self.hidden_activation_name = str(hidden_activation)
        self.output_activation_name = str(output_activation)
        self.use_bias = bool(use_bias)
        self.init_method = str(init_method)
        self.seed = seed

        self.hidden_act = get_activation(hidden_activation)
        self.output_act = get_activation(output_activation)

        # Build layers
        self.layers: List[LinearLayer] = []
        layer_dims = [self.input_dim] + list(self.hidden_dims) + [self.output_dim]

        rng = np.random.default_rng(seed)
        for i in range(len(layer_dims) - 1):
            layer_seed = int(rng.integers(0, 2**31 - 1)) if seed is not None else None
            layer = LinearLayer(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                use_bias=self.use_bias,
                init_method=self.init_method,
                seed=layer_seed,
            )
            self.layers.append(layer)

        # Internal caches for backpropagation
        self._cached_pre_activations: List[np.ndarray] = []
        self._cached_post_activations: List[np.ndarray] = []

    @property
    def num_parameters(self) -> int:
        """Calculate total number of trainable parameters in the model."""
        total = 0
        for layer in self.layers:
            total += layer.weights.size
            if layer.use_bias:
                total += layer.bias.size
        return total

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through encoder and prediction head.

        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            np.ndarray: Predicted output of shape (batch_size, output_dim) or (output_dim,).
        """
        is_1d = (x.ndim == 1)
        curr = x.reshape(1, -1) if is_1d else x

        self._cached_pre_activations = []
        self._cached_post_activations = [curr]

        num_hidden = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            z = layer.forward(curr)
            self._cached_pre_activations.append(z)

            if i < num_hidden:
                curr = self.hidden_act.forward(z)
            else:
                curr = self.output_act.forward(z)
            self._cached_post_activations.append(curr)

        return curr.squeeze(0) if is_1d else curr

    def get_latent_state(self, x: np.ndarray) -> np.ndarray:
        """Extract the internal representation S_t from the final hidden layer.

        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            np.ndarray: Latent state representation vector(s) S_t.
        """
        is_1d = (x.ndim == 1)
        curr = x.reshape(1, -1) if is_1d else x

        num_hidden = len(self.layers) - 1
        if num_hidden == 0:
            # If no hidden layer, input is the representation
            return curr.squeeze(0) if is_1d else curr

        for i in range(num_hidden):
            z = self.layers[i].forward(curr)
            curr = self.hidden_act.forward(z)

        return curr.squeeze(0) if is_1d else curr

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Perform analytical backpropagation through all layers.

        Args:
            grad_output: Upstream gradient dL/d(y_pred) of shape (batch_size, output_dim).

        Returns:
            np.ndarray: Downstream gradient dL/dx with respect to input.
        """
        if not self._cached_pre_activations:
            raise RuntimeError("Must call forward() before backward().")

        is_1d = (grad_output.ndim == 1)
        delta = grad_output.reshape(1, -1) if is_1d else grad_output

        num_layers = len(self.layers)
        num_hidden = num_layers - 1

        for i in reversed(range(num_layers)):
            z = self._cached_pre_activations[i]
            post = self._cached_post_activations[i + 1]

            # Compute dL/dz = dL/da * da/dz
            if i == num_hidden:
                act_deriv = self.output_act.derivative(z, post)
            else:
                act_deriv = self.hidden_act.derivative(z, post)

            dz = delta * act_deriv

            # Backprop through LinearLayer
            delta = self.layers[i].backward(dz)

        return delta.squeeze(0) if is_1d else delta

    def zero_grad(self) -> None:
        """Reset gradients across all layers."""
        for layer in self.layers:
            layer.zero_grad()

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Collect all parameter and gradient references."""
        all_params = []
        for i, layer in enumerate(self.layers):
            for name, param, grad in layer.get_params():
                all_params.append((f"layer_{i}_{name}", param, grad))
        return all_params

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete model architecture and weights."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
            "hidden_activation": self.hidden_activation_name,
            "output_activation": self.output_activation_name,
            "use_bias": self.use_bias,
            "init_method": self.init_method,
            "num_parameters": self.num_parameters,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictiveMLP":
        """Reconstruct model from serialized dictionary."""
        model = cls(
            input_dim=data["input_dim"],
            hidden_dims=tuple(data["hidden_dims"]),
            output_dim=data["output_dim"],
            hidden_activation=data.get("hidden_activation", "relu"),
            output_activation=data.get("output_activation", "identity"),
            use_bias=data.get("use_bias", True),
            init_method=data.get("init_method", "he"),
        )
        for i, layer_data in enumerate(data["layers"]):
            model.layers[i] = LinearLayer.from_dict(layer_data)
        return model

    def save_checkpoint(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model_state": self.to_dict(),
            "metadata": metadata or {},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> Tuple["PredictiveMLP", Dict[str, Any]]:
        """Load model and metadata from a JSON checkpoint file."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls.from_dict(payload["model_state"])
        return model, payload.get("metadata", {})
