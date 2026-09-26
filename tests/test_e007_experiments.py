"""Tests for CSP-P2-E007 diagnostic follow-up experiments.

Covers:
  - E007a: early-stopped re-benchmark output shape/keys
  - E007b: parameter-matched MLP (count within ±10%, three-way benchmark present)
  - E007c: representation-metric transparency fields
  - decide_part_b(): OR logic with partial-failure cases and capacity condition
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mlp import PredictiveMLP
from models.recurrent import RecurrentPredictor
from experiments.phase2_e007_experiments import (
    run_experiment_p2_e007a,
    run_experiment_p2_e007b,
    run_experiment_p2_e007c,
    decide_part_b,
    ORIGINAL_E001_NUMBERS,
)


# ---------------------------------------------------------------------------
# Helpers — minimal mock result dicts to drive decide_part_b() in isolation
# ---------------------------------------------------------------------------

def _mock_e007a(d0_persists: bool = False, gen_persists: bool = False) -> Dict[str, Any]:
    return {
        "d0_anomaly_persists_after_early_stopping": d0_persists,
        "generalization_failure_persists_after_early_stopping": gen_persists,
    }


def _mock_e007b(
    d0_persists: bool = False,
    gen_persists: bool = False,
    capacity_gap_pct: float = 0.0,
) -> Dict[str, Any]:
    return {
        "d0_anomaly_persists_after_param_matching": d0_persists,
        "generalization_failure_persists_after_param_matching": gen_persists,
        "capacity_gap_pct": capacity_gap_pct,
    }


# ---------------------------------------------------------------------------
# Test: decide_part_b() logic
# ---------------------------------------------------------------------------

class TestDecidePartBLogic(unittest.TestCase):
    """Verify decide_part_b() uses OR logic and returns a dict."""

    def test_returns_dict_not_bool(self):
        """decide_part_b must return a dict, not a bare bool."""
        result = decide_part_b(_mock_e007a(), _mock_e007b())
        self.assertIsInstance(result, dict, "decide_part_b must return a dict")

    def test_required_keys_present(self):
        """Result dict must include all documented keys."""
        result = decide_part_b(_mock_e007a(), _mock_e007b())
        required = {
            "trigger_part_b", "d0_anomaly_persists", "generalization_failure_persists",
            "capacity_gap_material", "capacity_gap_pct", "rationale",
        }
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")

    # --- Partial-failure cases (the contract's Acceptance Test 5 scenarios) ---

    def test_triggers_on_d0_alone(self):
        """d=0 anomaly persists after BOTH fixes (gen fixed) → must trigger."""
        result = decide_part_b(
            _mock_e007a(d0_persists=True),
            _mock_e007b(d0_persists=True, gen_persists=False),
        )
        self.assertTrue(result["trigger_part_b"], "Should trigger on d=0 alone")
        self.assertTrue(result["d0_anomaly_persists"])
        self.assertFalse(result["generalization_failure_persists"])

    def test_triggers_on_generalization_alone(self):
        """Generalization failure persists after BOTH fixes (d=0 fixed) → must trigger."""
        result = decide_part_b(
            _mock_e007a(d0_persists=False, gen_persists=True),
            _mock_e007b(d0_persists=False, gen_persists=True),
        )
        self.assertTrue(result["trigger_part_b"], "Should trigger on generalization alone")
        self.assertFalse(result["d0_anomaly_persists"])
        self.assertTrue(result["generalization_failure_persists"])

    def test_triggers_on_capacity_gap_alone(self):
        """Memory-vs-matched-MLP gap ≥ 3% (both anomalies fixed) → must trigger."""
        result = decide_part_b(
            _mock_e007a(d0_persists=False, gen_persists=False),
            _mock_e007b(d0_persists=False, gen_persists=False, capacity_gap_pct=4.5),
        )
        self.assertTrue(result["trigger_part_b"], "Should trigger on capacity gap alone")
        self.assertFalse(result["d0_anomaly_persists"])
        self.assertFalse(result["generalization_failure_persists"])
        self.assertTrue(result["capacity_gap_material"])
        self.assertAlmostEqual(result["capacity_gap_pct"], 4.5, places=1)

    def test_does_not_trigger_when_all_fixed(self):
        """Both anomalies fixed AND capacity gap < 3% → must NOT trigger."""
        result = decide_part_b(
            _mock_e007a(d0_persists=False, gen_persists=False),
            _mock_e007b(d0_persists=False, gen_persists=False, capacity_gap_pct=1.0),
        )
        self.assertFalse(result["trigger_part_b"], "Should NOT trigger when all fixed")
        self.assertFalse(result["d0_anomaly_persists"])
        self.assertFalse(result["generalization_failure_persists"])
        self.assertFalse(result["capacity_gap_material"])

    def test_does_not_trigger_on_partial_persistence_in_one_fix(self):
        """d=0 anomaly present after ES but fixed after param-match → NOT persistent."""
        # Condition requires BOTH fixes to still show the anomaly
        result = decide_part_b(
            _mock_e007a(d0_persists=True, gen_persists=False),  # ES shows d=0 anomaly
            _mock_e007b(d0_persists=False, gen_persists=False),  # PM fixed it
        )
        # d0_persists requires both flags True
        self.assertFalse(result["d0_anomaly_persists"])
        # trigger only if something else fires
        self.assertFalse(result["trigger_part_b"])

    def test_trigger_part_b_is_bool(self):
        """trigger_part_b must be a plain bool, not a truthy dict."""
        result = decide_part_b(_mock_e007a(), _mock_e007b())
        self.assertIsInstance(result["trigger_part_b"], bool)

    def test_rationale_is_string(self):
        """rationale must be a non-empty string."""
        result = decide_part_b(_mock_e007a(), _mock_e007b())
        self.assertIsInstance(result["rationale"], str)
        self.assertGreater(len(result["rationale"]), 10)

    def test_capacity_gap_threshold_documented(self):
        """capacity_gap_threshold_used key must be present and be 3.0."""
        result = decide_part_b(_mock_e007a(), _mock_e007b())
        self.assertIn("capacity_gap_threshold_used", result)
        self.assertAlmostEqual(result["capacity_gap_threshold_used"], 3.0, places=1)


# ---------------------------------------------------------------------------
# Test: E007a output structure
# ---------------------------------------------------------------------------

class TestE007aStructure(unittest.TestCase):
    """Verify E007a returns correctly structured results (fast, minimal run)."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_experiment_p2_e007a(
            delay=2,
            num_train_episodes=8,
            num_test_episodes=4,
            steps_per_episode=10,
            seed=42,
            early_stopping_patience=5,
        )

    def test_experiment_id(self):
        self.assertEqual(self.result["experiment_id"], "CSP-P2-E007a")

    def test_required_top_level_keys(self):
        required = {
            "benchmark", "comparison_vs_original", "delay_sweep",
            "generalization_results", "memory_model", "no_memory_model",
            "d0_anomaly_persists_after_early_stopping",
            "generalization_failure_persists_after_early_stopping",
            "_model_b_state", "_model_a_state",
        }
        for key in required:
            self.assertIn(key, self.result, f"Missing key: {key}")

    def test_model_training_metadata(self):
        """best_epoch and trainval_gap_ratio must be present and sane."""
        mem = self.result["memory_model"]
        self.assertIn("best_epoch", mem)
        self.assertIn("trainval_gap_ratio", mem)
        self.assertGreater(mem["trainval_gap_ratio"], 0.0)

    def test_comparison_vs_original_keys(self):
        cmp = self.result["comparison_vs_original"]
        required = {
            "original_memory_mse", "corrected_memory_mse",
            "original_gap_pct", "corrected_gap_pct", "delta_gap_pct",
        }
        for key in required:
            self.assertIn(key, cmp)

    def test_delay_sweep_has_five_records(self):
        self.assertEqual(len(self.result["delay_sweep"]), 5)

    def test_delay_sweep_covers_all_delays(self):
        delays = {r["delay"] for r in self.result["delay_sweep"]}
        self.assertEqual(delays, {0, 1, 2, 4, 8})

    def test_generalization_results_structure(self):
        gen = self.result["generalization_results"]
        self.assertIn("delay_3", gen)
        self.assertIn("delay_6", gen)
        for key in ["mse", "persistence_mse", "generalization_successful"]:
            self.assertIn(key, gen["delay_3"])

    def test_anomaly_flags_are_bool(self):
        self.assertIsInstance(
            self.result["d0_anomaly_persists_after_early_stopping"], bool
        )
        self.assertIsInstance(
            self.result["generalization_failure_persists_after_early_stopping"], bool
        )

    def test_model_state_dicts_are_serializable(self):
        """Model states stored in result must reconstruct valid models."""
        model_b = RecurrentPredictor.from_dict(self.result["_model_b_state"])
        self.assertEqual(model_b.hidden_dim, 32)
        model_a = PredictiveMLP.from_dict(self.result["_model_a_state"])
        self.assertIn(32, model_a.hidden_dims)


# ---------------------------------------------------------------------------
# Test: E007b — parameter-matched MLP
# ---------------------------------------------------------------------------

class TestE007bMatchedMLP(unittest.TestCase):
    """Verify the matched MLP has the right parameter count and the output has three-way benchmark."""

    def test_matched_mlp_parameter_count(self):
        """Matched MLP (173 hidden units) must be within ±10% of target 6753."""
        model = PredictiveMLP(input_dim=37, hidden_dims=(173,), output_dim=1)
        target = ORIGINAL_E001_NUMBERS["memory_params"]
        self.assertGreaterEqual(model.num_parameters, target * 0.9,
                                f"Matched MLP too small: {model.num_parameters}")
        self.assertLessEqual(model.num_parameters, target * 1.1,
                             f"Matched MLP too large: {model.num_parameters}")

    def test_matched_mlp_exact_count(self):
        """Verify the exact documented count: hidden_dims=(173,) → 6748 params."""
        model = PredictiveMLP(input_dim=37, hidden_dims=(173,), output_dim=1)
        # 37*173 + 173 + 173*1 + 1 = 6401 + 173 + 173 + 1 = 6748
        self.assertEqual(model.num_parameters, 6748)

    @classmethod
    def setUpClass(cls):
        cls.result = run_experiment_p2_e007b(
            delay=2,
            num_train_episodes=8,
            num_test_episodes=4,
            steps_per_episode=10,
            seed=42,
            early_stopping_patience=5,
        )

    def test_experiment_id(self):
        self.assertEqual(self.result["experiment_id"], "CSP-P2-E007b")

    def test_three_way_benchmark_present(self):
        """All three models must appear in the benchmark output."""
        bench = self.result["three_way_benchmark"]
        self.assertIn("memory_model_gru", bench)
        self.assertIn("matched_mlp_173", bench)
        self.assertIn("small_mlp_32_16", bench)

    def test_three_way_benchmark_has_mse(self):
        bench = self.result["three_way_benchmark"]
        for model_key in ["memory_model_gru", "matched_mlp_173", "small_mlp_32_16"]:
            self.assertIn("mse", bench[model_key], f"Missing mse in {model_key}")
            self.assertIn("num_parameters", bench[model_key])

    def test_capacity_analysis_present(self):
        cap = self.result["capacity_analysis"]
        self.assertIn("fraction_of_gap_surviving_capacity_match", cap)
        self.assertIn("corrected_memory_vs_matched_gap_pct", cap)
        self.assertIn("interpretation", cap)

    def test_capacity_gap_pct_key_present(self):
        """capacity_gap_pct must be a top-level float — it feeds decide_part_b()."""
        self.assertIn("capacity_gap_pct", self.result)
        self.assertIsInstance(self.result["capacity_gap_pct"], float)

    def test_matched_mlp_param_error_within_10pct(self):
        arch = self.result["matched_mlp_architecture"]
        self.assertLessEqual(arch["parameter_match_pct_error"], 10.0)

    def test_delay_sweep_records(self):
        self.assertEqual(len(self.result["delay_sweep"]), 5)

    def test_anomaly_flags_are_bool(self):
        self.assertIsInstance(
            self.result["d0_anomaly_persists_after_param_matching"], bool
        )
        self.assertIsInstance(
            self.result["generalization_failure_persists_after_param_matching"], bool
        )


# ---------------------------------------------------------------------------
# Test: E007c — representation metric transparency
# ---------------------------------------------------------------------------

class TestE007cRepresentationMetric(unittest.TestCase):
    """Verify the sanity-check fields are present and logically consistent."""

    @classmethod
    def setUpClass(cls):
        # Build a minimal E007a stub to pass in pre-trained model state
        e007a = run_experiment_p2_e007a(
            delay=2,
            num_train_episodes=8,
            num_test_episodes=4,
            steps_per_episode=10,
            seed=42,
            early_stopping_patience=5,
        )
        cls.result = run_experiment_p2_e007c(seed=42, e007a_results=e007a)

    def test_experiment_id(self):
        self.assertEqual(self.result["experiment_id"], "CSP-P2-E007c")

    def test_representation_analysis_keys(self):
        ra = self.result["representation_analysis"]
        required = {
            "inter_class_cosine_similarity",
            "cosine_similarity_threshold",
            "cosine_sim_margin_from_threshold",
            "distinct_memory_representations_formed",
            "representation_distinctness_confidence",
            "delayed_steps_intraclass_variance",
            "normal_steps_intraclass_variance",
            "pca_top2_explained_variance_ratio",
        }
        for key in required:
            self.assertIn(key, ra, f"Missing key in representation_analysis: {key}")

    def test_threshold_audit_keys(self):
        ta = self.result["threshold_audit"]
        required = {
            "documented_threshold", "threshold_logic",
            "original_cos_sim", "corrected_cos_sim",
            "original_distinct_flag", "corrected_distinct_flag",
            "contradiction_resolved",
        }
        for key in required:
            self.assertIn(key, ta, f"Missing key in threshold_audit: {key}")

    def test_threshold_is_documented_value(self):
        """The threshold must be the documented 0.95 value."""
        ra = self.result["representation_analysis"]
        self.assertAlmostEqual(ra["cosine_similarity_threshold"], 0.95, places=5)

    def test_margin_sign_consistent_with_flag(self):
        """If margin > 0 (cos_sim < threshold), distinct must be True."""
        ra = self.result["representation_analysis"]
        margin = ra["cosine_sim_margin_from_threshold"]
        distinct = ra["distinct_memory_representations_formed"]
        if margin > 0:
            self.assertTrue(distinct, "Positive margin → distinct should be True")
        else:
            self.assertFalse(distinct, "Negative margin → distinct should be False")

    def test_confidence_field_valid_values(self):
        ra = self.result["representation_analysis"]
        valid = {"borderline", "clear_distinct", "not_distinct"}
        self.assertIn(ra["representation_distinctness_confidence"], valid)

    def test_contradiction_explanation_is_string(self):
        ta = self.result["threshold_audit"]
        self.assertIsInstance(ta["contradiction_resolved"], str)
        self.assertGreater(len(ta["contradiction_resolved"]), 20)

    def test_pca_ratios_sum_leq_one(self):
        ra = self.result["representation_analysis"]
        pca = ra["pca_top2_explained_variance_ratio"]
        self.assertLessEqual(len(pca), 2)
        total = sum(pca)
        self.assertLessEqual(total, 1.01)  # small float tolerance


# ---------------------------------------------------------------------------
# Test: decide_part_b call sites read dict["trigger_part_b"]
# ---------------------------------------------------------------------------

class TestDecidePartBCallSiteSafety(unittest.TestCase):
    """Verify that a plain dict is not accidentally used as a truthy bool."""

    def test_empty_dict_would_be_truthy_but_trigger_key_is_false(self):
        """
        Guard against the old pattern: `if decide_part_b(...):` where a dict
        is always truthy. The correct pattern is: `if result['trigger_part_b']:`.
        """
        result = decide_part_b(
            _mock_e007a(d0_persists=False, gen_persists=False),
            _mock_e007b(d0_persists=False, gen_persists=False, capacity_gap_pct=0.5),
        )
        # As a dict, result is truthy even when nothing should fire:
        self.assertTrue(bool(result))  # dict is always truthy
        # But trigger_part_b is explicitly False:
        self.assertFalse(result["trigger_part_b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
