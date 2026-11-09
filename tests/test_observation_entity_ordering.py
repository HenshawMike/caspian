import unittest
from environment.observations import build_observation
from environment.entities import Entity


class TestEntityObservationOrdering(unittest.TestCase):
    def test_nearest_entity_occupies_slot_zero_regardless_of_id(self):
        # entity_id=5 is far away; entity_id=1 is adjacent to the agent.
        # A naive entity_id sort would put the far entity in slot 0.
        far_entity = Entity(entity_id=5, entity_type=1, position=(4, 4))
        near_entity = Entity(entity_id=1, entity_type=1, position=(0, 1))

        obs = build_observation(
            agent_pos=(0, 0), entities=[far_entity, near_entity],
            collision=False, internal_state=100.0, timestep=0, previous_action=None,
        )

        self.assertEqual(obs.entities[0].entity_id, 1, "nearest entity should occupy slot 0")
        self.assertEqual(obs.entities[1].entity_id, 5)
        self.assertLess(obs.entities[0].distance, obs.entities[1].distance)

    def test_tie_broken_by_entity_id(self):
        e_a = Entity(entity_id=9, entity_type=1, position=(1, 0))
        e_b = Entity(entity_id=2, entity_type=1, position=(0, 1))  # same distance from (0,0)

        obs = build_observation(
            agent_pos=(0, 0), entities=[e_a, e_b],
            collision=False, internal_state=100.0, timestep=0, previous_action=None,
        )
        self.assertEqual(obs.entities[0].entity_id, 2)
        self.assertEqual(obs.entities[1].entity_id, 9)


if __name__ == "__main__":
    unittest.main()
