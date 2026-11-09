import unittest
import numpy as np
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.actions import Action
from experiments.phase5_experiments import _encode_property_type


class TestEntityPropertiesPlumbing(unittest.TestCase):
    """Regression coverage for the `properties` field added to Entity/EntityObservation/
    WorldConfig for CSP-P5-E007 (property-based encoding). Mirrors the flicker-field
    regression pattern from test_entities_flicker.py: the two places that previously
    dropped new Entity fields during reset() (plain clone, and randomize_entity_positions
    clone) are exactly where a `properties` regression would hide.
    """

    def test_entity_roundtrip_preserves_properties(self):
        e = Entity(entity_id=1, entity_type=5, position=(0, 0), properties=(0.3, 0.7))
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        self.assertEqual(e2.properties, (0.3, 0.7))

    def test_entity_default_properties_is_empty_tuple(self):
        e = Entity(entity_id=1, entity_type=1, position=(0, 0))
        self.assertEqual(e.properties, ())

    def test_reset_preserves_properties_on_plain_clone(self):
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2), properties=(0.9,))],
            max_timesteps=10, seed=1,
        )
        world = GridWorld(config=config)
        world.reset(seed=1)
        self.assertEqual(world._entities[0].properties, (0.9,))

    def test_reset_preserves_properties_with_randomized_positions(self):
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2), properties=(0.9,))],
            max_timesteps=10, seed=1, randomize_entity_positions=True,
        )
        world = GridWorld(config=config)
        world.reset(seed=1)
        self.assertEqual(world._entities[0].properties, (0.9,))

    def test_entity_type_pool_assigns_properties_from_template(self):
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2))],
            max_timesteps=10, seed=7,
            entity_type_pool=[
                {"entity_type": 1, "hidden_state_delta": 10.0, "is_interactive": True, "is_blocking": False, "properties": [0.9]},
                {"entity_type": 2, "hidden_state_delta": -10.0, "is_interactive": True, "is_blocking": False, "properties": [0.1]},
            ],
        )
        world = GridWorld(config=config)
        for seed in range(20):
            world.reset(seed=seed)
            ent = world._entities[0]
            expected = (0.9,) if ent.entity_type == 1 else (0.1,)
            self.assertEqual(ent.properties, expected)

    def test_observation_carries_properties_through(self):
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2), properties=(0.42,))],
            max_timesteps=10, seed=1,
        )
        world = GridWorld(config=config)
        obs = world.reset(seed=1)
        self.assertEqual(obs.entities[0].properties, (0.42,))

    def test_properties_do_not_trip_semantic_firewall(self):
        """Adding a numeric properties field must not raise SemanticFirewallViolation."""
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2), properties=(0.5, 0.25))],
            max_timesteps=10, seed=1,
        )
        world = GridWorld(config=config)
        obs = world.reset(seed=1)  # would raise if assert_semantic_purity() failed
        self.assertEqual(obs.entities[0].properties, (0.5, 0.25))


class TestPropertyTypeEncoding(unittest.TestCase):
    def setUp(self):
        self.config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=9, position=(2, 2), properties=(0.73,))],
            max_timesteps=20, seed=1,
        )
        self.world = GridWorld(config=self.config)

    def test_output_dimension(self):
        obs = self.world.reset(seed=1)
        vec = _encode_property_type(obs, Action.NOOP, num_properties=1)
        # global(6) + 5 slots * (id + 1 property + rel_x + rel_y + dist) + action onehot(6)
        expected_dim = 6 + 5 * (1 + 1 + 3) + 6
        self.assertEqual(vec.shape[0], expected_dim)

    def test_property_value_present_in_slot0(self):
        obs = self.world.reset(seed=1)
        vec = _encode_property_type(obs, Action.NOOP, num_properties=1)
        slot0_start = 6
        # layout: [entity_id, property_0, rel_x, rel_y, distance]
        self.assertAlmostEqual(vec[slot0_start + 1], 0.73, places=6)

    def test_entity_type_code_not_present_anywhere_in_vector(self):
        """The whole point of E007: the raw entity_type code (9, an unusual value chosen
        specifically so it can't coincide with position/distance/property values below) must
        not leak into the encoding anywhere -- only property_0 should carry type information.
        """
        obs = self.world.reset(seed=1)
        vec = _encode_property_type(obs, Action.NOOP, num_properties=1)
        self.assertNotIn(9.0, list(vec))

    def test_absent_slot_is_zero(self):
        obs = self.world.reset(seed=1)
        vec = _encode_property_type(obs, Action.NOOP, num_properties=1)
        slot_width = 1 + 1 + 3
        slot1_start = 6 + slot_width
        np.testing.assert_array_equal(vec[slot1_start:slot1_start + slot_width], np.zeros(slot_width))


if __name__ == "__main__":
    unittest.main()
