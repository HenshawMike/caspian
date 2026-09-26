"""Unit tests for Phase 5 (Emergent Categories) Experiments."""

import unittest
from experiments.phase5_experiments import (
    run_experiment_p5_e001,
    run_experiment_p5_e002,
    run_experiment_p5_e003,
    run_experiment_p5_e005,
)


class TestPhase5Experiments(unittest.TestCase):
    def test_e001_latent_clustering(self):
        res = run_experiment_p5_e001(seed=42)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["distinct_category_clusters_formed"])

    def test_e002_probing_quality(self):
        res = run_experiment_p5_e002(seed=42)
        self.assertEqual(res["status"], "PASS")

    def test_e003_category_generalization(self):
        res = run_experiment_p5_e003(seed=42)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["zero_shot_category_transfer_successful"])

    def test_e005_criteria_assessment(self):
        res = run_experiment_p5_e005()
        self.assertEqual(res["criteria_passed"], 6)
        self.assertIn("SUPPORTED", res["scientific_conclusion"])


if __name__ == "__main__":
    unittest.main()
