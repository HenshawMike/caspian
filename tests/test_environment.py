"""Unit tests for Caspian Environment module."""

import unittest
import numpy as np
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.actions import Action
from environment.observations import FORBIDDEN_SEMANTIC_TOKENS, _check_semantic_purity_recursive


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.config = WorldConfig(
            grid_width=5,
            grid_height=5,
            initial_agent_pos=(0, 0),
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(2, 2),
                    is_interactive=True,
                    is_blocking=False,
                    hidden_state_delta=10.0,
                    interaction_delay=2,
                )
            ],
            initial_internal_state=50.0,
            step_penalty=1.0,
            default_interaction_delay=2,
            seed=42,
        )
        self.world = GridWorld(self.config)

    def test_reset_and_determinism(self):
        obs1 = self.world.reset(seed=42)
        obs2 = self.world.reset(seed=42)
        np.testing.assert_array_equal(obs1.to_vector(), obs2.to_vector())

    def test_movement_and_boundaries(self):
        self.world.reset(seed=42)
        # Agent at (0, 0). LEFT should hit boundary.
        obs, reward, done, info = self.world.step(Action.LEFT)
        self.assertEqual(self.world._agent_pos, (0, 0))

        # Move RIGHT
        obs, reward, done, info = self.world.step(Action.RIGHT)
        self.assertEqual(self.world._agent_pos, (1, 0))

    def test_semantic_firewall_purity(self):
        obs = self.world.reset(seed=42)
        obs_dict = obs.to_dict()
        # Should not raise any Exception
        _check_semantic_purity_recursive(obs_dict)

    def test_reset_clears_pending_events(self):
        self.world.reset(seed=42)
        # Move to entity at (2, 2)
        for act in [Action.RIGHT, Action.RIGHT, Action.UP, Action.UP]:
            self.world.step(act)
        # Interact
        self.world.step(Action.INTERACT)
        self.assertGreater(len(self.world._pending_events), 0)
        # Reset
        self.world.reset(seed=100)
        self.assertEqual(len(self.world._pending_events), 0)


if __name__ == "__main__":
    unittest.main()
