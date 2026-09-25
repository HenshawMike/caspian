"""Unit tests for CSP-P2-E009 experiment suite (EventWeightedMSELoss, Balanced Collector, 2x2 Factorial)."""

import unittest
import numpy as np

from models.optimizers import EventWeightedMSELoss, MSELoss
from learning.sequential_dataset import (
    collect_sequential_trajectories,
    collect_balanced_sequential_trajectories,
)
from environment.world import WorldConfig, Entity
from experiments.phase2_e009_experiments import (
    _train_and_evaluate_condition,
    run_experiment_p2_e009a,
    run_experiment_p2_e009b,
    run_experiment_p2_e009c,
    run_experiment_p2_e009d,
    run_experiment_p2_e009e,
)


class TestPhase2E009Experiments(unittest.TestCase):
    """Test suite for CSP-P2-E009."""

    def test_event_weighted_mse_loss_forward_and_gradient(self):
        """Verify EventWeightedMSELoss forward loss and analytical gradient against finite difference."""
        loss_fn = EventWeightedMSELoss(w_event=3.0)

        y_pred = np.array([[1.0], [2.0], [0.5], [4.0]], dtype=np.float64)
        y_true = np.array([[0.0], [10.0], [0.0], [10.0]], dtype=np.float64)

        # Forward value calculation
        # y_true != 0 at idx 1 (diff: -8, sq: 64, w: 3 -> 192) and idx 3 (diff: -6, sq: 36, w: 3 -> 108)
        # y_true == 0 at idx 0 (diff: 1, sq: 1, w: 1 -> 1) and idx 2 (diff: 0.5, sq: 0.25, w: 1 -> 0.25)
        # Total = (1 + 192 + 0.25 + 108) / 4 = 301.25 / 4 = 75.3125
        loss_val = loss_fn.forward(y_pred, y_true)
        self.assertAlmostEqual(loss_val, 75.3125, places=5)

        # Analytical gradient
        analytical_grad = loss_fn.gradient(y_pred, y_true)

        # Finite difference numerical gradient
        eps = 1e-6
        numerical_grad = np.zeros_like(y_pred)
        for i in range(y_pred.size):
            y_plus = y_pred.copy()
            y_minus = y_pred.copy()
            y_plus.flat[i] += eps
            y_minus.flat[i] -= eps
            l_plus = loss_fn.forward(y_plus, y_true)
            l_minus = loss_fn.forward(y_minus, y_true)
            numerical_grad.flat[i] = (l_plus - l_minus) / (2.0 * eps)

        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-5, atol=1e-5)

    def test_balanced_sequential_trajectory_collection(self):
        """Verify collect_balanced_sequential_trajectories guarantees minimum interactions and logs audit."""
        cfg = WorldConfig(
            grid_width=5,
            grid_height=5,
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(2, 2),
                    is_interactive=True,
                    hidden_state_delta=10.0,
                    interaction_delay=2,
                )
            ],
            default_interaction_delay=2,
            seed=42,
        )

        dataset, audit = collect_balanced_sequential_trajectories(
            world_config=cfg,
            num_episodes=5,
            steps_per_episode=25,
            interaction_prob=0.30,
            min_interactions_per_episode=2,
            max_resample_attempts=20,
            seed=42,
        )

        self.assertEqual(len(dataset.episodes), 5)
        self.assertIn("mean_interactions_per_episode", audit)
        self.assertIn("total_consequence_events", audit)
        self.assertTrue(audit["all_episodes_met_requirement"])

        for ep in dataset.episodes:
            int_count = sum(1 for t in ep.transitions if t.info.get("interaction_occurred", False))
            self.assertGreaterEqual(int_count, 2)

    def test_e009a_2x2_factorial_execution(self):
        """Verify E009a runs 2x2 conditions and computes main/interaction effects."""
        res = run_experiment_p2_e009a(seeds=[42], delay=2)

        self.assertEqual(res["experiment_id"], "CSP-P2-E009a")
        self.assertIn("summary_by_condition", res)
        summary = res["summary_by_condition"]
        self.assertIn("Condition_A_Control", summary)
        self.assertIn("Condition_B_BalancedOnly", summary)
        self.assertIn("Condition_C_WeightedOnly", summary)
        self.assertIn("Condition_D_FullE009", summary)
        self.assertIn("factorial_effects_on_deficit_reduction_pp", res)

    def test_e009d_safety_audits(self):
        """Verify E009d passes boundary isolation and semantic firewall checks."""
        res = run_experiment_p2_e009d(seed=42)
        self.assertTrue(res["boundary_isolation"]["isolation_perfect"])
        self.assertTrue(res["semantic_firewall"]["firewall_passed"])

    def test_e009e_frozen_criteria_assessment(self):
        """Verify E009e evaluates all 7 criteria and outputs paired t-test metrics."""
        res = run_experiment_p2_e009e(seeds=[1, 2])
        self.assertEqual(res["total_criteria"], 7)
        self.assertIn("paired_statistics_8_seeds", res)
        self.assertIn("p_value", res["paired_statistics_8_seeds"])


if __name__ == "__main__":
    unittest.main()
