"""Unit tests for CSP-P2-E008 experiment suite (reproducibility, ablations, final reconciliation)."""

import unittest
import numpy as np

from experiments.phase2_e008_experiments import (
    run_experiment_p2_e008a,
    run_experiment_p2_e008b,
    run_experiment_p2_e008c,
    run_experiment_p2_e008d,
    run_experiment_p2_e008e,
    ORIGINAL_PHASE2_NUMBERS,
)


class TestPhase2E008Experiments(unittest.TestCase):
    """Test suite for CSP-P2-E008."""

    def test_e008a_ablations_and_determinism(self):
        """Verify E008a executes ablations, identifies p, and proves triple-run determinism."""
        res = run_experiment_p2_e008a(seed=42)

        self.assertEqual(res["experiment_id"], "CSP-P2-E008a")
        self.assertIn("p_parameter_identification", res)
        self.assertEqual(res["p_parameter_identification"]["parameter_name"], "interaction_prob")

        ablations = res["ablations"]
        self.assertIn("baseline_e004_settings", ablations)
        self.assertIn("ablation_a_episode_count_30_vs_36", ablations)
        self.assertIn("ablation_b_interaction_density_030_vs_025", ablations)
        self.assertIn("ablation_c_early_stopping_vs_none", ablations)
        self.assertIn("combined_corrected_settings", ablations)

        det = res["determinism_proof"]
        self.assertTrue(det["triple_run_identical"], "Triple-run at seed=42 must produce bit-identical results")
        self.assertTrue(det["negative_control_divergent"], "Negative control with seed=43 must produce different results")

    def test_e008b_canonical_sweep_and_generalization(self):
        """Verify E008b produces canonical sweep and generalization with 3-way side-by-side tables."""
        res = run_experiment_p2_e008b(seed=42)

        self.assertEqual(res["experiment_id"], "CSP-P2-E008b")
        self.assertIn("canonical_delay_sweep", res)
        self.assertEqual(len(res["canonical_delay_sweep"]), 5)  # [0, 1, 2, 4, 8]

        self.assertIn("canonical_generalization", res)
        gen = res["canonical_generalization"]
        self.assertIn("delay_3", gen["results"])
        self.assertIn("delay_6", gen["results"])

        sbs = res["side_by_side_comparisons"]
        self.assertIn("delay_sweep", sbs)
        self.assertIn("generalization", sbs)
        self.assertIn("delay_3_interpolation", sbs["generalization"])
        self.assertIn("delay_6_extrapolation", sbs["generalization"])

    def test_e008c_criterion1_deficit_isolation(self):
        """Verify E008c re-measures deficit-step error reduction vs 70% bar and standard steps."""
        res = run_experiment_p2_e008c(delay=2, seed=42)

        self.assertEqual(res["experiment_id"], "CSP-P2-E008c")
        self.assertIn("deficit_step_results", res)
        self.assertIn("standard_step_results", res)
        self.assertIn("comparison_vs_original", res)

        def_res = res["deficit_step_results"]
        self.assertEqual(def_res["target_threshold_pct"], 70.0)
        self.assertIn("error_reduction_vs_matched_pct", def_res)
        self.assertIsInstance(def_res["cleared_70pct_threshold"], bool)

    def test_e008d_criterion3_8seed_reproducibility(self):
        """Verify E008d executes across 8 seeds and returns accurate win rate and consistent_superiority."""
        res = run_experiment_p2_e008d(seeds=[1, 2, 3, 4, 5, 6, 7, 8], delay=2)

        self.assertEqual(res["experiment_id"], "CSP-P2-E008d")
        self.assertEqual(len(res["records"]), 8)
        summary = res["summary"]
        self.assertIn("win_rate", summary)
        self.assertIn("consistent_superiority", summary)
        self.assertIsInstance(summary["consistent_superiority"], bool)

    def test_e008e_final_7criteria_reconciliation(self):
        """Verify E008e maps all 7 criteria and outputs a clear scientific conclusion and Phase 3 status."""
        res = run_experiment_p2_e008e(seed=42)

        self.assertEqual(res["experiment_id"], "CSP-P2-E008e")
        self.assertEqual(res["total_criteria"], 7)
        self.assertEqual(len(res["criteria_table"]), 7)

        # Check all 7 criterion IDs are present
        ids = [c["criterion_id"] for c in res["criteria_table"]]
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6, 7])

        self.assertIn(res["scientific_conclusion"], ["SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED"])
        self.assertIn("phase3_status", res)
        self.assertIn("is_justified", res["phase3_status"])


if __name__ == "__main__":
    unittest.main()
