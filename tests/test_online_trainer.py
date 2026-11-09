import unittest
import numpy as np
from models.mlp import PredictiveMLP
from models.optimizers import Adam
from learning.online_trainer import OnlineAdaptationTrainer, detect_and_measure_adaptation


class TestOnlineAdaptationTrainer(unittest.TestCase):
    def _make_trainer(self, seed=1, buffer_size=8):
        model = PredictiveMLP(input_dim=4, hidden_dims=(8,), output_dim=1, hidden_activation="relu", seed=seed)
        opt = Adam(params=model.get_params(), lr=0.05, weight_decay=0.0)
        return OnlineAdaptationTrainer(model, optimizer=opt, buffer_size=buffer_size), model

    def test_observe_returns_nonnegative_squared_error(self):
        online, model = self._make_trainer()
        rng = np.random.default_rng(0)
        x = rng.normal(size=4)
        y = np.array([1.0])
        err = online.observe(x, y)
        self.assertGreaterEqual(err, 0.0)

    def test_error_decreases_as_model_learns_a_fixed_mapping(self):
        """Repeatedly observing the SAME (x, y) pair should drive the pre-update error
        toward zero -- the simplest possible sanity check that gradient updates are
        actually happening and pointed the right way.
        """
        online, model = self._make_trainer(seed=2, buffer_size=4)
        x = np.array([1.0, 0.5, -0.5, 0.2])
        y = np.array([3.0])
        errors = [online.observe(x, y) for _ in range(200)]
        self.assertLess(np.mean(errors[-10:]), np.mean(errors[:10]))
        self.assertLess(np.mean(errors[-10:]), 0.5)

    def test_buffer_is_bounded(self):
        online, model = self._make_trainer(buffer_size=5)
        rng = np.random.default_rng(3)
        for _ in range(20):
            online.observe(rng.normal(size=4), np.array([rng.normal()]))
        self.assertEqual(len(online._buffer_x), 5)
        self.assertEqual(len(online._buffer_y), 5)

    def test_frozen_checkpoint_unaffected_by_online_updates(self):
        """A checkpoint taken via to_dict()/from_dict() before any online updates must
        stay numerically identical afterward -- this is exactly what CSP-P6-E004's
        persistence control depends on (the frozen model must be a genuinely separate
        object, not a view onto the same mutable weights).
        """
        model = PredictiveMLP(input_dim=4, hidden_dims=(8,), output_dim=1, hidden_activation="relu", seed=5)
        frozen = PredictiveMLP.from_dict(model.to_dict())
        frozen_weights_before = [layer.weights.copy() for layer in frozen.layers]

        opt = Adam(params=model.get_params(), lr=0.1, weight_decay=0.0)
        online = OnlineAdaptationTrainer(model, optimizer=opt, buffer_size=4)
        rng = np.random.default_rng(6)
        for _ in range(50):
            online.observe(rng.normal(size=4), np.array([rng.normal() * 5]))

        for before, layer in zip(frozen_weights_before, frozen.layers):
            np.testing.assert_array_equal(before, layer.weights)

        # Sanity: the LIVE model's weights, by contrast, should have moved.
        moved = any(
            not np.array_equal(w_before, layer.weights)
            for w_before, layer in zip(frozen_weights_before, model.layers)
        )
        self.assertTrue(moved)


class TestDetectAndMeasureAdaptation(unittest.TestCase):
    def _records(self, pre_errors, post_errors):
        recs = [{"regime": "pre", "squared_error": e} for e in pre_errors]
        recs += [{"regime": "post", "squared_error": e} for e in post_errors]
        return recs

    def test_insufficient_data_flagged(self):
        result = detect_and_measure_adaptation(self._records([1.0], [1.0]))
        self.assertTrue(result["insufficient_data"])

    def test_detects_a_clear_spike_and_recovery(self):
        pre = [1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 1.1] * 2
        post = [50.0, 48.0, 45.0, 40.0, 1.0, 1.1, 0.9, 1.0, 1.0, 1.1]
        result = detect_and_measure_adaptation(self._records(pre, post), sustain=2, rolling_window=1)
        self.assertFalse(result["insufficient_data"])
        self.assertIsNotNone(result["detection_event_index"])
        self.assertEqual(result["detection_event_index"], 0)
        self.assertIsNotNone(result["adaptation_event_index"])
        self.assertTrue(result["recovered_to_baseline_by_end"])

    def test_no_detection_when_nothing_changes(self):
        pre = [1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 1.1] * 2
        post = [1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 1.0, 1.1, 0.9]
        result = detect_and_measure_adaptation(self._records(pre, post))
        self.assertIsNone(result["detection_event_index"])
        self.assertIsNone(result["adaptation_event_index"])


if __name__ == "__main__":
    unittest.main()
