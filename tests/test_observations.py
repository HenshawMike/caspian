"""Unit tests and Semantic Firewall audit for Caspian observations."""

import unittest
import math

from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from environment.entities import Entity
from environment.observations import (
    Observation,
    EntityObservation,
    SemanticFirewallViolation,
    FORBIDDEN_SEMANTIC_TOKENS,
    _check_semantic_purity_recursive,
)


class TestObservations(unittest.TestCase):
    """Test suite for neutral observation generation and Semantic Firewall enforcement."""

    def setUp(self):
        self.config = WorldConfig(
            grid_width=5,
            grid_height=5,
            initial_agent_pos=(1, 1),
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=101,
                    position=(4, 5),
                    is_interactive=True,
                    hidden_state_delta=10.0,
                )
            ],
            initial_internal_state=75.0,
        )
        self.world = GridWorld(config=self.config)

    def test_neutral_observation_properties(self):
        """Verify observation contains only permitted neutral physical measurements."""
        obs = self.world.get_observation()

        self.assertEqual(obs.agent_position, (1, 1))
        self.assertEqual(obs.internal_state, 75.0)
        self.assertEqual(obs.timestep, 0)
        self.assertFalse(obs.collision)
        self.assertIsNone(obs.previous_action)
        self.assertEqual(len(obs.entities), 1)

        ent_obs = obs.entities[0]
        self.assertEqual(ent_obs.entity_id, 1)
        self.assertEqual(ent_obs.entity_type, 101)
        self.assertEqual(ent_obs.relative_x, 3)  # 4 - 1
        self.assertEqual(ent_obs.relative_y, 4)  # 5 - 1
        expected_dist = math.sqrt(3**2 + 4**2)  # 5.0
        self.assertAlmostEqual(ent_obs.distance, expected_dist, places=3)

    def test_observation_vectorization(self):
        """Verify observation converts to standard flat numerical vector."""
        obs = self.world.get_observation()
        vec = obs.to_vector(max_entities=2)

        # Vector format: [x, y, collision, internal_state, timestep, prev_action, (id, type, rx, ry, dist)*2]
        self.assertEqual(len(vec), 6 + 2 * 5)
        self.assertEqual(vec[0], 1.0)
        self.assertEqual(vec[1], 1.0)
        self.assertEqual(vec[2], 0.0)
        self.assertEqual(vec[3], 75.0)
        self.assertEqual(vec[4], 0.0)
        self.assertEqual(vec[5], -1.0)  # None represented as -1.0
        self.assertEqual(vec[6], 1.0)   # entity_id
        self.assertEqual(vec[7], 101.0) # entity_type
        self.assertEqual(vec[8], 3.0)   # rel_x
        self.assertEqual(vec[9], 4.0)   # rel_y
        self.assertEqual(vec[10], 5.0)  # distance

    def test_observation_dictionary_serialization(self):
        """Verify dictionary serialization preserves neutral structure."""
        obs = self.world.get_observation()
        obs_dict = obs.to_dict()

        self.assertIn("agent_position", obs_dict)
        self.assertIn("entities", obs_dict)
        self.assertIn("collision", obs_dict)
        self.assertIn("internal_state", obs_dict)
        self.assertIn("timestep", obs_dict)
        self.assertIn("previous_action", obs_dict)

    def test_semantic_firewall_audit_passes_on_environment(self):
        """Comprehensive audit: verify no forbidden semantic tokens exist in observations during a full trajectory."""
        world = GridWorld()
        actions = [Action.UP, Action.RIGHT, Action.INTERACT, Action.DOWN, Action.LEFT, Action.NOOP]

        for action in actions:
            obs, _, _, _ = world.step(action)
            obs_dict = obs.to_dict()

            # 1. Purity assertion passes
            obs.assert_semantic_purity()

            # 2. Stringified dictionary check for all forbidden tokens
            obs_str = str(obs_dict).lower()
            for token in FORBIDDEN_SEMANTIC_TOKENS:
                self.assertNotIn(
                    token,
                    obs_str,
                    f"Forbidden semantic token '{token}' leaked into observation representation: {obs_str}"
                )

    def test_semantic_firewall_violation_detection(self):
        """Verify the semantic firewall detector actively raises SemanticFirewallViolation when tokens leak."""
        contaminated_dict_key = {"food_level": 10.0}
        with self.assertRaises(SemanticFirewallViolation):
            _check_semantic_purity_recursive(contaminated_dict_key)

        contaminated_dict_val = {"label": "danger_zone"}
        with self.assertRaises(SemanticFirewallViolation):
            _check_semantic_purity_recursive(contaminated_dict_val)

        contaminated_list = [{"agent_pos": [0, 0]}, "reward_signal"]
        with self.assertRaises(SemanticFirewallViolation):
            _check_semantic_purity_recursive(contaminated_list)


if __name__ == "__main__":
    unittest.main()
