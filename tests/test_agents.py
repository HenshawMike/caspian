"""Unit tests for Caspian baseline agents (RandomAgent and OracleAgent)."""

import unittest

from agent.random_agent import RandomAgent
from agent.oracle_agent import OracleAgent, ScriptedAgent
from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from environment.entities import Entity
from environment.observations import FORBIDDEN_SEMANTIC_TOKENS


class TestAgents(unittest.TestCase):
    """Test suite for Random and Oracle baseline agents."""

    def test_random_agent_determinism(self):
        """Verify RandomAgent produces identical action streams with identical seeds."""
        world = GridWorld()
        obs = world.get_observation()

        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)
        agent3 = RandomAgent(seed=999)

        actions1 = [agent1.act(obs) for _ in range(50)]
        actions2 = [agent2.act(obs) for _ in range(50)]
        actions3 = [agent3.act(obs) for _ in range(50)]

        self.assertEqual(actions1, actions2)
        self.assertNotEqual(actions1, actions3)

    def test_random_agent_valid_actions(self):
        """Verify RandomAgent selects only valid Action enum instances."""
        agent = RandomAgent(seed=123)
        world = GridWorld()
        obs = world.get_observation()

        for _ in range(100):
            act = agent.act(obs)
            self.assertIsInstance(act, Action)
            self.assertIn(act, list(Action))

    def test_oracle_agent_navigation_and_interaction(self):
        """Verify OracleAgent navigates directly to entity and repeatedly interacts."""
        # Agent at (0, 0), Entity 1 at (2, 2), radius=1
        config = WorldConfig(
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
                )
            ],
            initial_internal_state=50.0,
            step_penalty=1.0,
            max_internal_state=100.0,
        )
        world = GridWorld(config=config)
        agent = OracleAgent(interaction_radius=1)

        obs = world.get_observation()
        # Step 1: rel_x = 2, rel_y = 2 -> RIGHT -> pos (1, 0)
        a1 = agent.act(obs)
        obs, d1, _, _ = world.step(a1)
        self.assertEqual(a1, Action.RIGHT)
        self.assertEqual(obs.agent_position, (1, 0))

        # Step 2: rel_x = 1, rel_y = 2 -> UP -> pos (1, 1)
        a2 = agent.act(obs)
        obs, d2, _, _ = world.step(a2)
        self.assertEqual(a2, Action.UP)
        self.assertEqual(obs.agent_position, (1, 1))

        # Step 3: pos (1, 1) to (2, 2) has Manhattan dist = 2 > 1 -> RIGHT -> pos (2, 1)
        a3 = agent.act(obs)
        obs, d3, _, _ = world.step(a3)
        self.assertEqual(a3, Action.RIGHT)
        self.assertEqual(obs.agent_position, (2, 1))

        # Step 4: pos (2, 1) to (2, 2) has Manhattan dist = 1 <= 1 -> INTERACT!
        a4 = agent.act(obs)
        self.assertEqual(a4, Action.INTERACT)
        obs, d4, _, info4 = world.step(a4)
        self.assertTrue(info4["interaction"])
        # Energy: 50.0 - 1(step1) - 1(step2) - 1(step3) - 1(step4) + 10 = 56.0
        self.assertEqual(obs.internal_state, 56.0)

        # Step 5: Remains in range -> INTERACT!
        a5 = agent.act(obs)
        self.assertEqual(a5, Action.INTERACT)
        obs, d5, _, info5 = world.step(a5)
        self.assertTrue(info5["interaction"])
        self.assertEqual(obs.internal_state, 65.0)

    def test_oracle_agent_empty_entities(self):
        """Verify OracleAgent defaults to NOOP when no entities exist."""
        config = WorldConfig(initial_entities=[])
        world = GridWorld(config=config)
        agent = OracleAgent()
        act = agent.act(world.get_observation())
        self.assertEqual(act, Action.NOOP)

    def test_oracle_agent_outperforms_random(self):
        """Verify OracleAgent achieves significantly higher internal state and interactions than RandomAgent."""
        config = WorldConfig(
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
                )
            ],
            initial_internal_state=50.0,
            step_penalty=1.0,
            max_timesteps=40,
            seed=42,
        )

        # Evaluate RandomAgent
        world_random = GridWorld(config=config)
        agent_random = RandomAgent(seed=42)
        random_interactions = 0
        obs = world_random.get_observation()
        for _ in range(30):
            act = agent_random.act(obs)
            obs, delta, done, info = world_random.step(act)
            if info["interaction"]:
                random_interactions += 1
            if done:
                break
        random_final_energy = obs.internal_state

        # Evaluate OracleAgent
        world_oracle = GridWorld(config=config)
        agent_oracle = OracleAgent(interaction_radius=1)
        oracle_interactions = 0
        obs = world_oracle.get_observation()
        for _ in range(30):
            act = agent_oracle.act(obs)
            obs, delta, done, info = world_oracle.step(act)
            if info["interaction"]:
                oracle_interactions += 1
            if done:
                break
        oracle_final_energy = obs.internal_state

        self.assertGreater(oracle_interactions, random_interactions)
        self.assertGreater(oracle_final_energy, random_final_energy)

    def test_agent_semantic_firewall(self):
        """Verify agents do not expose or require semantic tokens."""
        agent_r = RandomAgent()
        agent_o = OracleAgent()

        for agent in [agent_r, agent_o]:
            cls_name = agent.__class__.__name__.lower()
            for token in FORBIDDEN_SEMANTIC_TOKENS:
                self.assertNotIn(token, cls_name)


if __name__ == "__main__":
    unittest.main()
