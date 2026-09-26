"""Unit tests for Caspian Neural Models."""

import unittest
import numpy as np
from models.mlp import PredictiveMLP
from models.recurrent import RecurrentPredictor
from models.optimizers import MSELoss, EventWeightedMSELoss


class TestModels(unittest.TestCase):
    def test_mlp_forward_and_params(self):
        mlp = PredictiveMLP(input_dim=37, hidden_dims=(173,), output_dim=1, seed=42)
        x = np.random.randn(10, 37)
        out = mlp.forward(x)
        self.assertEqual(out.shape, (10, 1))
        self.assertAlmostEqual(mlp.num_parameters, 6748, delta=10)

    def test_recurrent_forward_and_params(self):
        gru = RecurrentPredictor(input_dim=37, hidden_dim=32, output_dim=1, seed=42)
        x = np.random.randn(4, 10, 37)  # (Batch, Time, Dim)
        out = gru.forward(x)
        self.assertEqual(out.shape, (4, 10, 1))
        self.assertEqual(gru.num_parameters, 6753)

    def test_losses(self):
        mse = MSELoss()
        y_pred = np.array([[1.0], [2.0]])
        y_true = np.array([[1.0], [0.0]])
        loss = mse.forward(y_pred, y_true)
        self.assertAlmostEqual(loss, 2.0)

        ew_mse = EventWeightedMSELoss(w_event=3.0)
        loss_ew = ew_mse.forward(y_pred, y_true)
        # step 0: diff=0; step 1: diff=2.0, sq=4.0, y_true=0.0 -> w=1.0 -> weighted sum = 4.0 / 2 = 2.0
        self.assertAlmostEqual(loss_ew, 2.0)


if __name__ == "__main__":
    unittest.main()
