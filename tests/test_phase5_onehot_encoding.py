import unittest
import numpy as np
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.actions import Action
from experiments.phase5_experiments import _encode_onehot_type, TYPE_VOCAB


class TestOnehotTypeEncoding(unittest.TestCase):
    def setUp(self):
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=3, position=(2, 2))],
            max_timesteps=20, seed=1,
        )
        self.world = GridWorld(config=config)

    def test_output_dimension(self):
        obs = self.world.reset(seed=1)
        vec = _encode_onehot_type(obs, Action.NOOP)
        v = len(TYPE_VOCAB)
        expected_dim = 6 + 5 * (1 + v + 3) + 6  # global + 5 slots*(id+onehot+relpos) + action onehot
        self.assertEqual(vec.shape[0], expected_dim)

    def test_correct_onehot_slot_set_for_known_type(self):
        obs = self.world.reset(seed=1)
        vec = _encode_onehot_type(obs, Action.NOOP)
        # slot 0 starts right after the 6 global features; layout is [id, onehot(4), rel_x, rel_y, dist]
        slot0_start = 6
        onehot = vec[slot0_start + 1: slot0_start + 1 + len(TYPE_VOCAB)]
        expected = np.zeros(len(TYPE_VOCAB))
        expected[TYPE_VOCAB.index(3)] = 1.0  # entity_type=3 in this world
        np.testing.assert_array_equal(onehot, expected)

    def test_no_ordinal_relationship_unlike_raw_scalar(self):
        """The whole point: type 1 and type 4's onehot vectors should be exactly as
        different as type 1 and type 2's (orthogonal), unlike a raw scalar where
        |1-4| > |1-2|.
        """
        def onehot_for_type(t):
            cfg = WorldConfig(
                grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
                initial_entities=[Entity(entity_id=1, entity_type=t, position=(2, 2))],
                max_timesteps=20, seed=1,
            )
            w = GridWorld(config=cfg)
            obs = w.reset(seed=1)
            vec = _encode_onehot_type(obs, Action.NOOP)
            return vec[7:7 + len(TYPE_VOCAB)]  # just the onehot block

        d_1_2 = np.linalg.norm(onehot_for_type(1) - onehot_for_type(2))
        d_1_4 = np.linalg.norm(onehot_for_type(1) - onehot_for_type(4))
        self.assertAlmostEqual(d_1_2, d_1_4)  # both equally (dis)similar -- no ordinal structure

    def test_absent_slot_is_zero(self):
        obs = self.world.reset(seed=1)
        vec = _encode_onehot_type(obs, Action.NOOP)
        slot1_start = 6 + (1 + len(TYPE_VOCAB) + 3)  # second entity slot (absent in this world)
        slot_width = 1 + len(TYPE_VOCAB) + 3
        np.testing.assert_array_equal(vec[slot1_start:slot1_start + slot_width], np.zeros(slot_width))


if __name__ == "__main__":
    unittest.main()
