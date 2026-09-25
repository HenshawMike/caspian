"""Unit tests for Phase 1 Predictive Agent lifecycle and Semantic Firewall compliance."""

import unittest
import numpy as np

from agent.predictive_agent import Phase1PredictiveAgent
from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from environment.observations import FORBIDDEN_SEMANTIC_TOKENS, SemanticFirewallViolation
from models.mlp import PredictiveMLP


class TestPhase1Agent(unittest.TestCase):
    """Test suite for Phase1PredictiveAgent."""

    def test_agent_initialization_and_action_selection(self):
        world = GridWorld()
        obs = world.get_observation()

        agent = Phase1PredictiveAgent(exploration_rate=0.0, seed=42)
        act = agent.act(obs)

        self.assertIsInstance(act, Action)
        self.assertIn(act, list(Action))

    def test_agent_prediction_and_latent_state(self):
        world = GridWorld()
        obs = world.get_observation()

        agent = Phase1PredictiveAgent(seed=42)
        pred_delta = agent.predict(obs, Action.INTERACT)
        self.assertIsInstance(pred_delta, float)

        latent = agent.get_internal_state(obs, Action.INTERACT)
        self.assertIsInstance(latent, np.ndarray)
        self.assertEqual(latent.shape, (16,))  # default hidden_dims=(32, 16)

    def test_agent_semantic_firewall(self):
        """Verify that agent class and dictionary outputs contain zero forbidden semantic tokens."""
        agent = Phase1PredictiveAgent(seed=42)
        world = GridWorld()
        obs = world.get_observation()

        all_preds = agent.evaluate_all_actions(obs)
        pred_repr = str(all_preds).lower()

        for token in FORBIDDEN_SEMANTIC_TOKENS:
            self.assertNotIn(
                token,
                pred_repr,
                f"Forbidden semantic token '{token}' found in agent predictions dictionary.",
            )

    def test_agent_reset(self):
        agent = Phase1PredictiveAgent(seed=42)
        world = GridWorld()
        obs = world.get_observation()

        a1 = agent.act(obs)
        agent.reset(seed=42)
        a2 = agent.act(obs)

        self.assertEqual(a1, a2)


if __name__ == "__main__":
    unittest.main()
