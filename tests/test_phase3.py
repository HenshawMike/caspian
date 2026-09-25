"""Unit tests for Phase 3 (Generalization) Experiments."""

import unittest
from experiments.phase3_experiments import (
    run_experiment_p3_e001,
    run_experiment_p3_e002,
    run_experiment_p3_e003,
    run_experiment_p3_e005,
)


class TestPhase3Experiments(unittest.TestCase):
    def test_e001_spatial_generalization(self):
        res = run_experiment_p3_e001(seed=42)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["all_positions_passed"])

    def test_e002_layout_generalization(self):
        res = run_experiment_p3_e002(seed=42)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["all_layouts_passed"])

    def test_e003_multientity(self):
        res = run_experiment_p3_e003(seed=42)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["gru_beats_mlp"])

    def test_e005_criteria_assessment(self):
        res = run_experiment_p3_e005()
        self.assertEqual(res["criteria_passed"], 6)
        self.assertIn("SUPPORTED", res["scientific_conclusion"])


if __name__ == "__main__":
    unittest.main()
