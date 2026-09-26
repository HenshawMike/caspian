"""Unit tests for Caspian GridWorld initialization, config, and lifecycle."""

import unittest
import json
from pathlib import Path

from environment.world import GridWorld, WorldConfig, EnvironmentState
from environment.actions import Action
from environment.entities import Entity


class TestWorld(unittest.TestCase):
    """Test suite for GridWorld lifecycle and configuration."""

    def test_default_initialization(self):
        """Verify default initialization produces valid initial state."""
        world = GridWorld()
        state = world.get_state()
        self.assertEqual(state.agent_position, (0, 0))
        self.assertEqual(state.internal_state, 100.0)
        self.assertEqual(state.timestep, 0)
        self.assertFalse(state.collision)
        self.assertFalse(state.interaction_occurred)
        self.assertIsNone(state.last_action)
        self.assertFalse(state.is_terminated)
        self.assertEqual(len(state.entities), 1)

    def test_config_from_dict_and_file(self):
        """Verify WorldConfig serializes and deserializes cleanly."""
        config_path = Path(__file__).parent.parent / "configs" / "default_world.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = WorldConfig.from_dict(data)
        self.assertEqual(config.grid_width, 5)
        self.assertEqual(config.grid_height, 5)
        self.assertEqual(config.initial_agent_pos, (0, 0))
        self.assertEqual(len(config.initial_entities), 1)
        self.assertEqual(config.seed, 42)

        config_dict = config.to_dict()
        self.assertEqual(config_dict["grid_width"], 5)
        self.assertEqual(config_dict["initial_agent_pos"], [0, 0])

    def test_reset_lifecycle(self):
        """Verify reset restores all state variables to initial state."""
        world = GridWorld()
        world.step(Action.UP)
        world.step(Action.RIGHT)
        world.step(Action.INTERACT)

        state_after_steps = world.get_state()
        self.assertEqual(state_after_steps.timestep, 3)
        self.assertNotEqual(state_after_steps.agent_position, (0, 0))

        obs = world.reset()
        state_after_reset = world.get_state()

        self.assertEqual(state_after_reset.timestep, 0)
        self.assertEqual(state_after_reset.agent_position, (0, 0))
        self.assertEqual(state_after_reset.internal_state, 100.0)
        self.assertFalse(state_after_reset.collision)
        self.assertFalse(state_after_reset.interaction_occurred)
        self.assertIsNone(state_after_reset.last_action)
        self.assertFalse(state_after_reset.is_terminated)
        self.assertEqual(obs.timestep, 0)
        self.assertEqual(obs.agent_position, (0, 0))

    def test_max_timestep_termination(self):
        """Verify environment terminates cleanly upon reaching max_timesteps."""
        config = WorldConfig(max_timesteps=3)
        world = GridWorld(config=config)

        obs, delta, done, info = world.step(Action.NOOP)
        self.assertFalse(done)
        obs, delta, done, info = world.step(Action.NOOP)
        self.assertFalse(done)
        obs, delta, done, info = world.step(Action.NOOP)
        self.assertTrue(done)
        self.assertTrue(world.get_state().is_terminated)

        # Subsequent step without reset must raise RuntimeError
        with self.assertRaises(RuntimeError):
            world.step(Action.NOOP)


if __name__ == "__main__":
    unittest.main()
