"""Unit tests and analytical BPTT gradient verification for Recurrent neural models."""

import unittest
import numpy as np

from models.recurrent import GRULayer, RecurrentPredictor
from models.optimizers import MSELoss


class TestRecurrentModels(unittest.TestCase):
    """Test suite for GRULayer and RecurrentPredictor."""

    def test_gru_layer_forward_shape(self):
        """Verify GRULayer forward output shape on 3D sequence inputs."""
        layer = GRULayer(in_features=10, hidden_dim=8, seed=42)
        x_seq = np.random.randn(4, 7, 10)  # (Batch=4, Time=7, Dim=10)
        h_seq = layer.forward(x_seq)

        self.assertEqual(h_seq.shape, (4, 7, 8))

    def test_recurrent_predictor_forward_shape(self):
        """Verify RecurrentPredictor forward output shape."""
        model = RecurrentPredictor(input_dim=12, hidden_dim=16, output_dim=1, seed=42)
        x_seq = np.random.randn(3, 5, 12)
        y_pred = model.forward(x_seq)

        self.assertEqual(y_pred.shape, (3, 5, 1))

    def test_recurrent_predictor_online_step(self):
        """Verify step-by-step sequential inference updates hidden state."""
        model = RecurrentPredictor(input_dim=6, hidden_dim=8, output_dim=1, seed=42)
        model.reset_state()

        h0 = model.get_hidden_state()
        np.testing.assert_array_equal(h0, np.zeros(8))

        x1 = np.ones(6)
        y1 = model.step(x1)
        h1 = model.get_hidden_state()

        self.assertEqual(y1.shape, (1,))
        self.assertFalse(np.all(h1 == 0.0))

        # Reset state should return to zeros
        model.reset_state()
        np.testing.assert_array_equal(model.get_hidden_state(), np.zeros(8))

    def test_analytical_bptt_gradient_check(self):
        """CRITICAL MATHEMATICAL TEST:
        Verify analytical BPTT gradients against finite-difference numerical gradients.
        """
        rng = np.random.default_rng(42)
        in_dim, hidden_dim, out_dim = 4, 4, 1
        batch_size, seq_len = 2, 3

        model = RecurrentPredictor(
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            output_dim=out_dim,
            seed=42,
        )
        loss_fn = MSELoss()

        X = rng.normal(0.0, 1.0, (batch_size, seq_len, in_dim))
        Y = rng.normal(0.0, 1.0, (batch_size, seq_len, out_dim))

        # 1. Analytical backward pass
        model.zero_grad()
        y_pred = model.forward(X)
        grad_out = loss_fn.gradient(y_pred, Y)
        model.backward(grad_out)

        params = model.get_params()
        eps = 1e-5

        # 2. Check each parameter against numerical gradient
        for name, param, grad_analytical in params:
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                orig_val = param[idx]

                # f(theta + eps)
                param[idx] = orig_val + eps
                y_plus = model.forward(X)
                loss_plus = loss_fn.forward(y_plus, Y)

                # f(theta - eps)
                param[idx] = orig_val - eps
                y_minus = model.forward(X)
                loss_minus = loss_fn.forward(y_minus, Y)

                # Restore original parameter
                param[idx] = orig_val

                grad_numerical = (loss_plus - loss_minus) / (2.0 * eps)
                grad_ana = grad_analytical[idx]

                # Relative error: |ana - num| / max(|ana|, |num|, 1e-7)
                diff = abs(grad_ana - grad_numerical)
                denom = max(abs(grad_ana), abs(grad_numerical), 1e-7)
                rel_error = diff / denom

                self.assertLess(
                    rel_error,
                    1e-3,
                    f"Gradient check failed for parameter '{name}' at index {idx}: "
                    f"analytical={grad_ana:.6e}, numerical={grad_numerical:.6e}, rel_error={rel_error:.6e}"
                )
                it.iternext()

    def test_serialization_and_checkpointing(self):
        """Verify model serialization and bitwise restoration."""
        model = RecurrentPredictor(input_dim=5, hidden_dim=6, output_dim=1, seed=123)
        x_seq = np.random.randn(2, 4, 5)
        out_orig = model.forward(x_seq)

        state_dict = model.to_dict()
        restored = RecurrentPredictor.from_dict(state_dict)
        out_restored = restored.forward(x_seq)

        np.testing.assert_allclose(out_orig, out_restored, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
