"""Unit tests for Phase 1 and Phase 2 pipelines."""

import unittest
from experiments.phase2_e009_experiments import run_experiment_p2_e009d, run_experiment_p2_e009e


class TestPipelines(unittest.TestCase):
    def test_e009d_safety_audits(self):
        res = run_experiment_p2_e009d(seed=42)
        self.assertTrue(res["boundary_isolation"]["isolation_perfect"])
        self.assertTrue(res["semantic_firewall"]["firewall_passed"])

    def test_e009e_criteria_assessment(self):
        # Run e009e with a single seed for fast unit testing
        res = run_experiment_p2_e009e(seeds=[1])
        self.assertEqual(res["scientific_conclusion"], "SUPPORTED")
        self.assertEqual(res["criteria_passed"], 7)


if __name__ == "__main__":
    unittest.main()
