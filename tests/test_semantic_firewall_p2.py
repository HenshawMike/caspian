"""Semantic firewall audit tests for Phase 2 components."""

import unittest
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.observations import (
    FORBIDDEN_SEMANTIC_TOKENS,
    _check_semantic_purity_recursive,
    SemanticFirewallViolation,
)
from agent.memory import InternalMemory
from agent.memory_agent import Phase2MemoryAgent
from models.recurrent import RecurrentPredictor


class TestSemanticFirewallP2(unittest.TestCase):
    """Audits Phase 2 memory components, observations, and serialized models."""

    def test_observation_purity_under_delays(self):
        """Verify observations during delayed transitions never leak semantic tokens."""
        config = WorldConfig(
            grid_width=5,
            grid_height=5,
            initial_agent_pos=(2, 1),
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(2, 2),
                    is_interactive=True,
                    is_blocking=False,
                    hidden_state_delta=10.0,
                    interaction_delay=3,
                )
            ],
            seed=42,
        )
        world = GridWorld(config)

        for act in [1, 2, 5, 0, 0, 0]:  # includes INTERACT (5)
            obs, delta, done, info = world.step(act)
            # Must not raise SemanticFirewallViolation
            obs.assert_semantic_purity()
            _check_semantic_purity_recursive(obs.to_dict())
            _check_semantic_purity_recursive(info)

    def test_memory_state_serialization_purity(self):
        """Verify InternalMemory serialization contains zero semantic tokens."""
        mem = InternalMemory(hidden_dim=16)
        mem.update([0.1] * 16)
        data = mem.to_dict()
        _check_semantic_purity_recursive(data)

    def test_recurrent_model_serialization_purity(self):
        """Verify RecurrentPredictor dictionary serialization contains zero semantic tokens."""
        model = RecurrentPredictor(input_dim=37, hidden_dim=16, output_dim=1, seed=42)
        state_dict = model.to_dict()
        _check_semantic_purity_recursive(state_dict)

    def test_firewall_raises_on_forbidden_token(self):
        """Verify the semantic firewall scanner actively detects injected forbidden tokens."""
        dirty_dict = {"agent_status": "searching_for_food", "energy": 50.0}
        with self.assertRaises(SemanticFirewallViolation):
            _check_semantic_purity_recursive(dirty_dict)


if __name__ == "__main__":
    unittest.main()
