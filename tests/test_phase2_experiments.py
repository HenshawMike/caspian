"""Integration tests for Phase 2 experimental suite (CSP-P2-E001 to CSP-P2-E006)."""

import unittest
from experiments.phase2_experiments import (
    run_experiment_p2_e001,
    run_experiment_p2_e002,
    run_experiment_p2_e003,
    run_experiment_p2_e004,
    run_experiment_p2_e005,
    run_experiment_p2_e006,
)


class TestPhase2Experiments(unittest.TestCase):
    """Test suite ensuring all Phase 2 experiments execute successfully."""

    def test_e001_temporal_memory_execution(self):
        """Verify CSP-P2-E001 runs and returns valid benchmark results."""
        res = run_experiment_p2_e001(delay=1, num_train_episodes=8, num_test_episodes=4, steps_per_episode=15, seed=42)
        self.assertEqual(res["experiment_id"], "CSP-P2-E001")
        self.assertIn("benchmark", res)
        bench = res["benchmark"]
        self.assertIn("memory_model", bench)
        self.assertIn("no_memory_model", bench)

    def test_e002_information_deficit_isolation(self):
        """Verify CSP-P2-E002 runs and partitions deficit steps."""
        res = run_experiment_p2_e002(delay=1, seed=42)
        self.assertEqual(res["experiment_id"], "CSP-P2-E002")
        self.assertIn("deficit_steps", res)
        self.assertIn("standard_steps", res)

    def test_e003_variable_delay_execution(self):
        """Verify CSP-P2-E003 executes across multiple delay conditions."""
        res = run_experiment_p2_e003(delays=[0, 1], seed=42)
        self.assertEqual(res["experiment_id"], "CSP-P2-E003")
        self.assertEqual(len(res["records"]), 2)

    def test_e004_generalization_execution(self):
        """Verify CSP-P2-E004 runs generalization tests."""
        res = run_experiment_p2_e004(train_delays=[1], test_delays=[2], seed=42)
        self.assertEqual(res["experiment_id"], "CSP-P2-E004")
        self.assertIn("unseen_delay_results", res)

    def test_e005_reproducibility_execution(self):
        """Verify CSP-P2-E005 runs across seeds."""
        res = run_experiment_p2_e005(seeds=[1, 2], delay=1)
        self.assertEqual(res["experiment_id"], "CSP-P2-E005")
        self.assertEqual(len(res["seed_records"]), 2)

    def test_e006_semantic_firewall_audit(self):
        """Verify CSP-P2-E006 audit runs and passes all checks."""
        res = run_experiment_p2_e006(seed=42)
        self.assertEqual(res["experiment_id"], "CSP-P2-E006")
        self.assertTrue(res["all_audits_passed"])


if __name__ == "__main__":
    unittest.main()
