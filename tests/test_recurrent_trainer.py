import unittest
import numpy as np
from models.rnn import RecurrentPredictiveModel
from learning.recurrent_trainer import RecurrentTrainer


class TestRecurrentTrainerSmoke(unittest.TestCase):
    def test_loss_decreases_on_learnable_synthetic_task(self):
        """A trivially learnable synthetic memory task: predict x_t[0] seen k steps ago."""
        rng = np.random.default_rng(0)
        batch, T, dim = 20, 10, 4
        X = rng.normal(size=(batch, T, dim))
        # Target at time t = value of feature 0 at time t-2 (requires 2-step memory).
        Y = np.zeros((batch, T, 1))
        for t in range(T):
            src_t = max(0, t - 2)
            Y[:, t, 0] = X[:, src_t, 0]

        model = RecurrentPredictiveModel(input_dim=dim, hidden_dim=8, output_dim=1, seed=0)
        trainer = RecurrentTrainer(model=model, batch_size=10, epochs=60, early_stopping_patience=None, seed=0)
        history = trainer.fit((X, Y))

        self.assertGreater(len(history["train_losses"]), 0)
        self.assertLess(history["train_losses"][-1], history["train_losses"][0])

    def test_predict_step_matches_forward_sequence(self):
        model = RecurrentPredictiveModel(input_dim=5, hidden_dim=3, output_dim=1, seed=1)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(1, 4, 5))

        seq_out = model.forward_sequence(X)

        h = None
        step_outs = []
        for t in range(4):
            y_t, h = model.predict_step(X[0, t, :], h)
            step_outs.append(y_t)
        step_outs = np.stack(step_outs, axis=0)[None, :, :]

        np.testing.assert_allclose(seq_out, step_outs, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
