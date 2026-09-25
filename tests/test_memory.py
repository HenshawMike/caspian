"""Unit tests for Phase 2 Memory & Persistence and Episode Isolation."""

import unittest
import numpy as np

from agent.memory import InternalMemory
from agent.memory_agent import Phase2MemoryAgent
from models.recurrent import RecurrentPredictor
from environment.world import GridWorld, WorldConfig
from environment.actions import Action


class TestInternalMemory(unittest.TestCase):
    """Test suite for InternalMemory container."""

    def test_memory_initialization(self):
        """Verify memory initializes to empty zero vector."""
        mem = InternalMemory(hidden_dim=16)
        self.assertTrue(mem.is_empty)
        self.assertEqual(mem.step_count, 0)
        np.testing.assert_array_equal(mem.get_state(), np.zeros(16))

    def test_memory_update_and_history(self):
        """Verify state updates and history recording."""
        mem = InternalMemory(hidden_dim=4)
        v1 = np.array([1.0, 2.0, 3.0, 4.0])
        mem.update(v1)

        self.assertFalse(mem.is_empty)
        self.assertEqual(mem.step_count, 1)
        np.testing.assert_array_equal(mem.get_state(), v1)

        v2 = np.array([0.5, -0.5, 1.5, -1.5])
        mem.update(v2)
        self.assertEqual(mem.step_count, 2)
        np.testing.assert_array_equal(mem.get_state(), v2)

        history = mem.get_history()
        self.assertEqual(len(history), 2)
        np.testing.assert_array_equal(history[0], v1)
        np.testing.assert_array_equal(history[1], v2)

    def test_critical_episode_isolation(self):
        """CRITICAL EPISODE TEST:
        Episode A -> memory contains information -> reset -> Episode B -> memory must be empty/reinitialized.
        """
        mem = InternalMemory(hidden_dim=8)

        # Episode A
        mem.update(np.ones(8) * 42.0)
        self.assertFalse(mem.is_empty)
        self.assertEqual(mem.step_count, 1)
        self.assertTrue(np.all(mem.get_state() == 42.0))

        # Reset between episodes
        mem.reset()

        # Episode B Verification
        self.assertTrue(mem.is_empty, "Memory must be empty immediately after reset.")
        self.assertEqual(mem.step_count, 0, "Step count must be 0 after reset.")
        self.assertEqual(len(mem.get_history()), 0, "History buffer must be cleared after reset.")
        np.testing.assert_array_equal(
            mem.get_state(),
            np.zeros(8),
            err_msg="Memory state must be strictly all zeros after reset."
        )


class TestMemoryAgentEpisodeIsolation(unittest.TestCase):
    """Test suite for Phase2MemoryAgent closed-loop episode boundary isolation."""

    def test_agent_episode_boundary_reset(self):
        """Verify agent memory is completely cleared when reset between episodes."""
        model = RecurrentPredictor(input_dim=37, hidden_dim=16, output_dim=1, seed=42)
        agent = Phase2MemoryAgent(model=model, hidden_dim=16, seed=42)
        world = GridWorld(WorldConfig(seed=42))

        # Run Episode A
        obs = world.reset(seed=42)
        agent.reset(seed=42)

        for _ in range(5):
            act = agent.act(obs)
            next_obs, delta, done, info = world.step(act)
            agent.observe(next_obs, delta, done, info)
            obs = next_obs

        # Memory in Episode A must now be populated
        self.assertFalse(agent.memory.is_empty)
        self.assertGreater(agent.memory.step_count, 0)
        self.assertFalse(np.all(agent.memory.get_state() == 0.0))

        # Reset for Episode B
        agent.reset(seed=101)
        world.reset(seed=101)

        # Verify Episode B starts with pristine memory state
        self.assertTrue(agent.memory.is_empty, "Agent memory must be empty at start of Episode B.")
        self.assertEqual(agent.memory.step_count, 0)
        np.testing.assert_array_equal(
            agent.memory.get_state(),
            np.zeros(16),
            err_msg="Episode B must have zero internal memory vector at initialization."
        )


if __name__ == "__main__":
    unittest.main()
