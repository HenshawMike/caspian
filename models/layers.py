"""Neural network layer primitives for Project Caspian.

Implements fully-connected (dense) layers with analytical backward passes,
gradient accumulation, and standard parameter initialization schemes.
"""

from typing import Optional, Dict, Tuple, List, Any
import numpy as np


class LinearLayer:
    """Fully-connected (dense) linear transformation layer: y = x @ W + b.

    Attributes:
        in_features: Dimensionality of input feature vectors.
        out_features: Dimensionality of output feature vectors.
        weights: Weight matrix W of shape (in_features, out_features).
        bias: Bias vector b of shape (out_features,).
        grad_weights: Gradient dL/dW of shape (in_features, out_features).
        grad_bias: Gradient dL/db of shape (out_features,).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        init_method: str = "he",
        seed: Optional[int] = None,
    ):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.init_method = init_method.lower()

        rng = np.random.default_rng(seed)

        # Parameter Initialization
        if self.init_method == "he":
            std = np.sqrt(2.0 / self.in_features)
            self.weights = rng.normal(0.0, std, (self.in_features, self.out_features))
        elif self.init_method == "xavier" or self.init_method == "glorot":
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            self.weights = rng.uniform(-limit, limit, (self.in_features, self.out_features))
        elif self.init_method == "uniform":
            bound = 1.0 / np.sqrt(self.in_features)
            self.weights = rng.uniform(-bound, bound, (self.in_features, self.out_features))
        elif self.init_method == "zeros":
            self.weights = np.zeros((self.in_features, self.out_features), dtype=np.float64)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        if self.use_bias:
            self.bias = np.zeros((self.out_features,), dtype=np.float64)
        else:
            self.bias = np.zeros((self.out_features,), dtype=np.float64)

        # Gradient storage
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        # Forward cache
        self._cached_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass: y = x @ W + b.

        Args:
            x: Input array of shape (batch_size, in_features) or (in_features,).

        Returns:
            np.ndarray: Output array of shape (batch_size, out_features) or (out_features,).
        """
        is_1d = (x.ndim == 1)
        x_2d = x.reshape(1, -1) if is_1d else x

        if x_2d.shape[1] != self.in_features:
            raise ValueError(
                f"Input dimension mismatch: expected (*, {self.in_features}), got {x_2d.shape}"
            )

        self._cached_input = x_2d

        out = np.matmul(x_2d, self.weights)
        if self.use_bias:
            out = out + self.bias

        return out.squeeze(0) if is_1d else out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Perform analytical backward pass.

        Computes gradients dL/dW and dL/db, and returns dL/dx.

        Args:
            grad_output: Upstream gradient dL/dy of shape (batch_size, out_features) or (out_features,).

        Returns:
            np.ndarray: Downstream gradient dL/dx of shape matching the forward input.
        """
        if self._cached_input is None:
            raise RuntimeError("Cannot perform backward pass before calling forward().")

        is_1d = (grad_output.ndim == 1)
        grad_2d = grad_output.reshape(1, -1) if is_1d else grad_output
        x_2d = self._cached_input

        # Compute parameter gradients
        self.grad_weights += np.matmul(x_2d.T, grad_2d)
        if self.use_bias:
            self.grad_bias += np.sum(grad_2d, axis=0)

        # Compute gradient with respect to input: dL/dx = grad_output @ W^T
        grad_input = np.matmul(grad_2d, self.weights.T)

        return grad_input.squeeze(0) if is_1d else grad_input

    def zero_grad(self) -> None:
        """Reset accumulated gradients to zero."""
        self.grad_weights.fill(0.0)
        self.grad_bias.fill(0.0)

    def get_params(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Return list of (name, parameter_tensor, gradient_tensor) tuples."""
        params = [("weight", self.weights, self.grad_weights)]
        if self.use_bias:
            params.append(("bias", self.bias, self.grad_bias))
        return params

    def to_dict(self) -> Dict[str, Any]:
        """Serialize layer configuration and weights to dictionary."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "use_bias": self.use_bias,
            "init_method": self.init_method,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist() if self.use_bias else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinearLayer":
        """Reconstruct layer from serialized dictionary."""
        layer = cls(
            in_features=data["in_features"],
            out_features=data["out_features"],
            use_bias=data.get("use_bias", True),
            init_method=data.get("init_method", "he"),
        )
        layer.weights = np.array(data["weights"], dtype=np.float64)
        if data.get("bias") is not None:
            layer.bias = np.array(data["bias"], dtype=np.float64)
        return layer
