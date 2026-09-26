"""Integration tests for Phase 1 experimental pipeline."""

import unittest

from experiments.phase1_experiments import (
    run_experiment_e001,
    run_experiment_e002,
    run_experiment_e003,
    run_experiment_e004,
    run_experiment_e005,
    run_experiment_e006,
)


class TestPhase1Pipeline(unittest.TestCase):
    """Test suite executing quick integration runs of all Phase 1 experiments."""

    def test_e001_quick_run(self):
        res = run_experiment_e001(num_train_steps=100, num_test_steps=50, seed=42)
        self.assertEqual(res["experiment_id"], "CSP-P1-E001")
        self.assertIn("benchmark", res)
        self.assertIn("caspian_model", res["benchmark"])

    def test_e002_quick_run(self):
        res = run_experiment_e002(seeds=[42, 101], num_train_steps=100, num_test_steps=50)
        self.assertEqual(res["experiment_id"], "CSP-P1-E002")
        self.assertEqual(len(res["per_seed_results"]), 2)

    def test_e003_quick_run(self):
        res = run_experiment_e003(seed=42, num_train_steps=100)
        self.assertEqual(res["experiment_id"], "CSP-P1-E003")
        self.assertIn("spatial_results", res)

    def test_e004_quick_run(self):
        res = run_experiment_e004(experience_budgets=[20, 50], seed=42)
        self.assertEqual(res["experiment_id"], "CSP-P1-E004")
        self.assertEqual(len(res["learning_curve"]), 2)

    def test_e005_quick_run(self):
        res = run_experiment_e005(seeds=[1, 2])
        self.assertEqual(res["experiment_id"], "CSP-P1-E005")
        self.assertEqual(len(res["seed_records"]), 2)

    def test_e006_quick_run(self):
        res = run_experiment_e006(seed=42)
        self.assertEqual(res["experiment_id"], "CSP-P1-E006")
        self.assertEqual(res["semantic_audit"]["leakages_found"], 0)


if __name__ == "__main__":
    unittest.main()
