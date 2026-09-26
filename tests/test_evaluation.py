"""Unit tests for evaluation metrics, baselines, and ModelEvaluator."""

import unittest
import numpy as np

from evaluation.metrics import (
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_r2,
    compute_baseline_gap,
    compute_cosine_similarity,
)
from evaluation.baselines import PersistencePredictor, RandomPredictor, ReactivePredictor
from evaluation.evaluator import ModelEvaluator
from models.mlp import PredictiveMLP


class TestEvaluation(unittest.TestCase):
    """Test suite for metrics and baseline predictors."""

    def test_metrics_calculation(self):
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[1.0], [2.0], [3.0], [4.0]])

        self.assertEqual(compute_mse(y_pred, y_true), 0.0)
        self.assertEqual(compute_rmse(y_pred, y_true), 0.0)
        self.assertEqual(compute_mae(y_pred, y_true), 0.0)
        self.assertEqual(compute_r2(y_pred, y_true), 1.0)

        y_noisy = np.array([[2.0], [2.0], [3.0], [4.0]])
        self.assertAlmostEqual(compute_mse(y_noisy, y_true), 0.25)
        self.assertAlmostEqual(compute_mae(y_noisy, y_true), 0.25)

    def test_baseline_gap(self):
        # Baseline MSE = 10.0, Model MSE = 2.0 -> Gap = 80.0%
        gap = compute_baseline_gap(2.0, 10.0)
        self.assertEqual(gap, 80.0)

    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        v3 = np.array([2.0, 0.0])

        self.assertAlmostEqual(compute_cosine_similarity(v1, v2), 0.0)
        self.assertAlmostEqual(compute_cosine_similarity(v1, v3), 1.0)

    def test_persistence_predictor(self):
        pred = PersistencePredictor(default_value=0.0)
        X = np.ones((5, 10))
        out = pred.predict(X)
        self.assertEqual(out.shape, (5, 1))
        np.testing.assert_array_equal(out, np.zeros((5, 1)))

    def test_reactive_predictor(self):
        reactive = ReactivePredictor()
        Y_train = np.array([[-1.0], [-1.0], [9.0], [-1.0]])
        reactive.fit(Y_train)
        X = np.ones((3, 10))
        out = reactive.predict(X)
        self.assertEqual(out.shape, (3, 1))
        self.assertAlmostEqual(out[0, 0], 1.5)

    def test_evaluator_benchmark(self):
        model = PredictiveMLP(input_dim=8, hidden_dims=(8,), output_dim=1, seed=42)
        X_test = np.random.randn(20, 8)
        Y_test = np.random.randn(20, 1)

        evaluator = ModelEvaluator(seed=42)
        bench = evaluator.benchmark_against_baselines(model, X_test, Y_test)

        self.assertIn("caspian_model", bench)
        self.assertIn("persistence_baseline", bench)
        self.assertIn("random_baseline", bench)
        self.assertIn("reactive_baseline", bench)
        self.assertIn("baseline_gap_persistence_pct", bench)


if __name__ == "__main__":
    unittest.main()
