"""Unit tests for Caspian transition dynamics, collision mechanics, and action handling."""

import unittest

from environment.world import GridWorld, WorldConfig
from environment.actions import Action, InvalidActionError
from environment.entities import Entity


class TestDynamics(unittest.TestCase):
    """Test suite for physical transitions, boundaries, collisions, and actions."""

    def setUp(self):
        self.config = WorldConfig(
            grid_width=5,
            grid_height=5,
            initial_agent_pos=(2, 2),
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(2, 3),  # Interactive entity 1 step UP
                    is_interactive=True,
                    is_blocking=False,
                    hidden_state_delta=15.0,
                ),
                Entity(
                    entity_id=2,
                    entity_type=2,
                    position=(3, 2),  # Blocking entity 1 step RIGHT
                    is_interactive=False,
                    is_blocking=True,
                    hidden_state_delta=0.0,
                ),
            ],
            initial_internal_state=50.0,
            step_penalty=1.0,
            min_internal_state=0.0,
            max_internal_state=100.0,
        )
        self.world = GridWorld(config=self.config)

    def test_cardinal_movement(self):
        """Verify standard 4-way cardinal movement."""
        # Start at (2, 2)
        obs, delta, done, info = self.world.step(Action.LEFT)
        self.assertEqual(obs.agent_position, (1, 2))
        self.assertFalse(obs.collision)

        obs, delta, done, info = self.world.step(Action.DOWN)
        self.assertEqual(obs.agent_position, (1, 1))
        self.assertFalse(obs.collision)

        obs, delta, done, info = self.world.step(Action.RIGHT)
        self.assertEqual(obs.agent_position, (2, 1))
        self.assertFalse(obs.collision)

        obs, delta, done, info = self.world.step(Action.UP)
        self.assertEqual(obs.agent_position, (2, 2))
        self.assertFalse(obs.collision)

    def test_boundary_containment_and_collision(self):
        """Verify Caspian cannot move outside the world boundaries and collision is flagged."""
        # Move to top-left corner (0, 4)
        config = WorldConfig(grid_width=3, grid_height=3, initial_agent_pos=(0, 2))
        world = GridWorld(config=config)

        # Attempt to move UP past y=2 boundary
        obs, delta, done, info = world.step(Action.UP)
        self.assertEqual(obs.agent_position, (0, 2))
        self.assertTrue(obs.collision)
        self.assertTrue(info["collision"])

        # Attempt to move LEFT past x=0 boundary
        obs, delta, done, info = world.step(Action.LEFT)
        self.assertEqual(obs.agent_position, (0, 2))
        self.assertTrue(obs.collision)

        # Move DOWN to (0, 0)
        world.step(Action.DOWN)
        world.step(Action.DOWN)
        self.assertEqual(world.get_observation().agent_position, (0, 0))

        # Attempt to move DOWN past y=0 boundary
        obs, delta, done, info = world.step(Action.DOWN)
        self.assertEqual(obs.agent_position, (0, 0))
        self.assertTrue(obs.collision)

    def test_blocking_entity_collision(self):
        """Verify agent cannot walk into a blocking entity."""
        # Initial pos (2, 2). Blocking entity at (3, 2).
        obs, delta, done, info = self.world.step(Action.RIGHT)
        self.assertEqual(obs.agent_position, (2, 2))
        self.assertTrue(obs.collision)

    def test_passable_entity_movement(self):
        """Verify agent can walk into a non-blocking entity."""
        # Initial pos (2, 2). Non-blocking entity at (2, 3).
        obs, delta, done, info = self.world.step(Action.UP)
        self.assertEqual(obs.agent_position, (2, 3))
        self.assertFalse(obs.collision)

    def test_interaction_mechanics_and_hidden_rule(self):
        """Verify interaction with entity changes internal state according to hidden rule."""
        # Initial energy = 50.0, step_penalty = 1.0.
        # Entity 1 at (2, 3) with hidden_state_delta = 15.0 is within interaction radius=1 of (2, 2).
        obs, delta, done, info = self.world.step(Action.INTERACT)
        # Expected new energy = 50.0 - 1.0 + 15.0 = 64.0
        self.assertEqual(obs.internal_state, 64.0)
        self.assertEqual(delta, 14.0)
        self.assertTrue(info["interaction"])

    def test_interaction_out_of_range(self):
        """Verify INTERACT does not trigger entity rule when out of range."""
        # Move away from entities: from (2, 2) move LEFT to (1, 2) then LEFT to (0, 2)
        self.world.step(Action.LEFT)  # pos (1, 2)
        self.world.step(Action.LEFT)  # pos (0, 2)
        # Entity 1 is at (2, 3), Manhattan distance is |2-0| + |3-2| = 3 > 1.
        obs, delta, done, info = self.world.step(Action.INTERACT)
        # Only step penalty applies: previous state 48.0 - 1.0 = 47.0
        self.assertEqual(obs.internal_state, 47.0)
        self.assertFalse(info["interaction"])

    def test_internal_state_clamping(self):
        """Verify internal state is strictly clamped between min and max bounds."""
        # Test max clamping
        config = WorldConfig(
            initial_internal_state=98.0,
            step_penalty=0.0,
            max_internal_state=100.0,
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(0, 0),
                    is_interactive=True,
                    hidden_state_delta=50.0,
                )
            ],
            initial_agent_pos=(0, 0),
        )
        world = GridWorld(config=config)
        obs, delta, done, info = world.step(Action.INTERACT)
        self.assertEqual(obs.internal_state, 100.0)

        # Test min clamping
        config_min = WorldConfig(
            initial_internal_state=2.0,
            step_penalty=5.0,
            min_internal_state=0.0,
            initial_agent_pos=(0, 0),
        )
        world_min = GridWorld(config=config_min)
        obs, delta, done, info = world_min.step(Action.NOOP)
        self.assertEqual(obs.internal_state, 0.0)

    def test_valid_actions_accepted(self):
        """Verify all valid action types (enum, int, str) are accepted."""
        world = GridWorld()
        obs, _, _, _ = world.step(Action.UP)
        self.assertEqual(obs.previous_action, int(Action.UP))

        obs, _, _, _ = world.step(1)  # DOWN
        self.assertEqual(obs.previous_action, int(Action.DOWN))

        obs, _, _, _ = world.step("left")
        self.assertEqual(obs.previous_action, int(Action.LEFT))

        obs, _, _, _ = world.step("RIGHT")
        self.assertEqual(obs.previous_action, int(Action.RIGHT))

        obs, _, _, _ = world.step(Action.INTERACT)
        self.assertEqual(obs.previous_action, int(Action.INTERACT))

        obs, _, _, _ = world.step(Action.NOOP)
        self.assertEqual(obs.previous_action, int(Action.NOOP))

    def test_invalid_actions_rejected(self):
        """Verify invalid actions are cleanly rejected with InvalidActionError."""
        world = GridWorld()

        with self.assertRaises(InvalidActionError):
            world.step(999)

        with self.assertRaises(InvalidActionError):
            world.step(-1)

        with self.assertRaises(InvalidActionError):
            world.step("JUMP")

        with self.assertRaises(InvalidActionError):
            world.step(3.14)  # type: ignore

        with self.assertRaises(InvalidActionError):
            world.step(None)  # type: ignore


if __name__ == "__main__":
    unittest.main()
