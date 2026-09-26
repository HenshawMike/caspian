"""Unit tests for Phase 4 (World Model & Planning) Experiments."""

import unittest
from experiments.phase4_experiments import (
    run_experiment_p4_e001,
    run_experiment_p4_e002,
    run_experiment_p4_e004,
)


class TestPhase4Experiments(unittest.TestCase):
    def test_e001_world_model_rollouts(self):
        res = run_experiment_p4_e001(seed=42)
        self.assertEqual(res["status"], "PASS")
        self.assertGreater(res["mean_advantage_gap_pct"], 0.0)

    def test_e002_model_based_planning(self):
        res = run_experiment_p4_e002(seed=42)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["planning_outperforms_reactive"])

    def test_e004_criteria_assessment(self):
        res = run_experiment_p4_e004()
        self.assertEqual(res["criteria_passed"], 5)
        self.assertIn("SUPPORTED", res["scientific_conclusion"])


if __name__ == "__main__":
    unittest.main()
