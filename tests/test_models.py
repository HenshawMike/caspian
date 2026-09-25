"""Unit tests for neural network primitives and PredictiveMLP architecture."""

import unittest
import numpy as np
import tempfile
import os

from models.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Identity, get_activation
from models.layers import LinearLayer
from models.optimizers import SGD, Adam, MSELoss, HuberLoss
from models.mlp import PredictiveMLP


class TestActivations(unittest.TestCase):
    """Test suite for mathematical activation functions and derivatives."""

    def test_relu(self):
        act = ReLU()
        x = np.array([-2.0, 0.0, 3.0])
        out = act.forward(x)
        np.testing.assert_allclose(out, [0.0, 0.0, 3.0])
        deriv = act.derivative(x, out)
        np.testing.assert_allclose(deriv, [0.0, 0.0, 1.0])

    def test_leaky_relu(self):
        act = LeakyReLU(alpha=0.1)
        x = np.array([-2.0, 3.0])
        out = act.forward(x)
        np.testing.assert_allclose(out, [-0.2, 3.0])
        deriv = act.derivative(x, out)
        np.testing.assert_allclose(deriv, [0.1, 1.0])

    def test_sigmoid(self):
        act = Sigmoid()
        x = np.array([0.0, 100.0, -100.0])
        out = act.forward(x)
        np.testing.assert_allclose(out[0], 0.5)
        self.assertGreater(out[1], 0.999)
        self.assertLess(out[2], 0.001)
        deriv = act.derivative(x, out)
        np.testing.assert_allclose(deriv[0], 0.25)

    def test_tanh(self):
        act = Tanh()
        x = np.array([0.0])
        out = act.forward(x)
        np.testing.assert_allclose(out, [0.0])
        deriv = act.derivative(x, out)
        np.testing.assert_allclose(deriv, [1.0])

    def test_get_activation(self):
        self.assertIsInstance(get_activation("relu"), ReLU)
        self.assertIsInstance(get_activation("sigmoid"), Sigmoid)
        self.assertIsInstance(get_activation("tanh"), Tanh)
        self.assertIsInstance(get_activation("linear"), Identity)


class TestLinearLayer(unittest.TestCase):
    """Test suite for LinearLayer transformations and gradient backpropagation."""

    def test_linear_forward_shape(self):
        layer = LinearLayer(in_features=10, out_features=5, seed=42)
        x = np.random.randn(8, 10)
        out = layer.forward(x)
        self.assertEqual(out.shape, (8, 5))

    def test_linear_1d_input(self):
        layer = LinearLayer(in_features=10, out_features=5, seed=42)
        x = np.random.randn(10)
        out = layer.forward(x)
        self.assertEqual(out.shape, (5,))

    def test_linear_numerical_gradient_check(self):
        """Verify analytical gradients against finite differences (numerical gradient check)."""
        rng = np.random.default_rng(42)
        layer = LinearLayer(in_features=4, out_features=3, seed=42)
        x = rng.normal(0, 1, (2, 4))
        target = rng.normal(0, 1, (2, 3))

        # Forward & analytical backward
        out = layer.forward(x)
        loss = np.sum((out - target) ** 2)
        grad_out = 2.0 * (out - target)

        layer.zero_grad()
        dx = layer.backward(grad_out)

        # Numerical check on weights
        eps = 1e-5
        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                orig = layer.weights[i, j]

                layer.weights[i, j] = orig + eps
                l_plus = np.sum((layer.forward(x) - target) ** 2)

                layer.weights[i, j] = orig - eps
                l_minus = np.sum((layer.forward(x) - target) ** 2)

                layer.weights[i, j] = orig
                num_grad = (l_plus - l_minus) / (2.0 * eps)
                ana_grad = layer.grad_weights[i, j]

                rel_err = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-8)
                self.assertLess(rel_err, 1e-4, f"Weight gradient mismatch at ({i}, {j})")


class TestPredictiveMLP(unittest.TestCase):
    """Test suite for PredictiveMLP end-to-end forward/backward passes and serialization."""

    def test_mlp_forward_and_latent(self):
        mlp = PredictiveMLP(
            input_dim=12,
            hidden_dims=(16, 8),
            output_dim=1,
            hidden_activation="relu",
            seed=42,
        )
        x = np.random.randn(5, 12)
        out = mlp.forward(x)
        latent = mlp.get_latent_state(x)

        self.assertEqual(out.shape, (5, 1))
        self.assertEqual(latent.shape, (5, 8))
        self.assertGreater(mlp.num_parameters, 0)

    def test_mlp_numerical_gradient_check(self):
        """Verify full MLP backpropagation against finite difference approximation."""
        rng = np.random.default_rng(123)
        mlp = PredictiveMLP(
            input_dim=6,
            hidden_dims=(8, 4),
            output_dim=1,
            hidden_activation="tanh",
            seed=123,
        )
        x = rng.normal(0, 1, (3, 6))
        target = rng.normal(0, 1, (3, 1))

        # Analytical backward
        out = mlp.forward(x)
        grad_out = 2.0 * (out - target) / 3.0
        mlp.zero_grad()
        mlp.backward(grad_out)

        # Check weights of first layer
        eps = 1e-5
        l0 = mlp.layers[0]
        for i in range(min(3, l0.weights.shape[0])):
            for j in range(min(3, l0.weights.shape[1])):
                orig = l0.weights[i, j]

                l0.weights[i, j] = orig + eps
                out_plus = mlp.forward(x)
                l_plus = np.mean((out_plus - target) ** 2)

                l0.weights[i, j] = orig - eps
                out_minus = mlp.forward(x)
                l_minus = np.mean((out_minus - target) ** 2)

                l0.weights[i, j] = orig
                num_grad = (l_plus - l_minus) / (2.0 * eps)
                ana_grad = l0.grad_weights[i, j]

                rel_err = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-7)
                self.assertLess(rel_err, 1e-3, f"MLP gradient mismatch at Layer 0 ({i}, {j})")

    def test_mlp_serialization(self):
        mlp = PredictiveMLP(input_dim=8, hidden_dims=(16, 8), output_dim=1, seed=42)
        x = np.random.randn(3, 8)
        out1 = mlp.forward(x)

        d = mlp.to_dict()
        mlp_restored = PredictiveMLP.from_dict(d)
        out2 = mlp_restored.forward(x)

        np.testing.assert_allclose(out1, out2)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file = os.path.join(tmpdir, "test_model.json")
            mlp.save_checkpoint(ckpt_file, metadata={"phase": 1})
            loaded_model, meta = PredictiveMLP.load_checkpoint(ckpt_file)
            out3 = loaded_model.forward(x)
            np.testing.assert_allclose(out1, out3)
            self.assertEqual(meta.get("phase"), 1)


if __name__ == "__main__":
    unittest.main()
