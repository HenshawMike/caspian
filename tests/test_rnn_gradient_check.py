import unittest
import numpy as np
from models.rnn import RecurrentPredictiveModel


class TestRNNGradientCheck(unittest.TestCase):
    """Verify analytical BPTT gradients against numerical (finite-difference) gradients.

    Mirrors the rigor of the Phase 1 report's gradient-check claim
    (relative error < 1e-4) for the new recurrent architecture's math.
    """

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.model = RecurrentPredictiveModel(input_dim=5, hidden_dim=4, output_dim=1, seed=0)
        self.batch, self.T = 3, 6
        self.X = self.rng.normal(size=(self.batch, self.T, 5))
        self.Y = self.rng.normal(size=(self.batch, self.T, 1))

    def _loss(self) -> float:
        Y_pred = self.model.forward_sequence(self.X)
        return float(np.mean(np.square(Y_pred - self.Y)))

    def _analytical_grads(self):
        self.model.zero_grad()
        Y_pred = self.model.forward_sequence(self.X)
        grad_Y = (2.0 / Y_pred.size) * (Y_pred - self.Y)
        self.model.backward_sequence(grad_Y)
        return {
            "W_xh": self.model.grad_W_xh.copy(),
            "W_hh": self.model.grad_W_hh.copy(),
            "b_h": self.model.grad_b_h.copy(),
            "W_hy": self.model.grad_W_hy.copy(),
            "b_y": self.model.grad_b_y.copy(),
        }

    def _numerical_grad(self, param: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]

            param[idx] = orig + eps
            loss_plus = self._loss()
            param[idx] = orig - eps
            loss_minus = self._loss()
            param[idx] = orig

            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        return grad

    def test_gradient_check_all_params(self):
        analytical = self._analytical_grads()
        for name, param in [
            ("W_xh", self.model.W_xh), ("W_hh", self.model.W_hh), ("b_h", self.model.b_h),
            ("W_hy", self.model.W_hy), ("b_y", self.model.b_y),
        ]:
            numerical = self._numerical_grad(param)
            rel_error = np.linalg.norm(analytical[name] - numerical) / (
                np.linalg.norm(analytical[name]) + np.linalg.norm(numerical) + 1e-10
            )
            self.assertLess(rel_error, 1e-4, msg=f"BPTT gradient check failed for {name}: rel_error={rel_error}")


if __name__ == "__main__":
    unittest.main()
