import unittest
import numpy as np
from learning.sequence_dataset import Episode, EpisodeDataset
from learning.dataset import Transition
from environment.observations import Observation, EntityObservation


def make_dummy_transition(delta=1.0):
    obs = Observation(
        agent_position=(0, 0), internal_state=100.0, timestep=0, collision=False,
        previous_action=None, entities=(),
    )
    next_obs = Observation(
        agent_position=(0, 0), internal_state=100.0 + delta, timestep=1, collision=False,
        previous_action=1, entities=(),
    )
    return Transition(obs=obs, action=1, state_delta=delta, next_obs=next_obs, done=False, info={})


class TestEpisodeDataset(unittest.TestCase):
    def test_steps_since_last_observable(self):
        ep = Episode(
            transitions=[make_dummy_transition() for _ in range(6)],
            entity_observable=[True, False, False, True, False, False],
        )
        self.assertEqual(ep.steps_since_last_observable(), [0, 1, 2, 0, 1, 2])

    def test_never_observed_yields_none(self):
        ep = Episode(
            transitions=[make_dummy_transition() for _ in range(3)],
            entity_observable=[False, False, True],
        )
        self.assertEqual(ep.steps_since_last_observable(), [None, None, 0])

    def test_to_numpy_arrays_shape(self):
        ds = EpisodeDataset()
        for _ in range(4):
            ds.add_episode(Episode(
                transitions=[make_dummy_transition() for _ in range(5)],
                entity_observable=[True] * 5,
            ))
        X, Y = ds.to_numpy_arrays()
        self.assertEqual(X.shape[0], 4)
        self.assertEqual(X.shape[1], 5)
        self.assertEqual(Y.shape, (4, 5, 1))

    def test_split_is_by_episode_not_timestep(self):
        ds = EpisodeDataset()
        for _ in range(10):
            ds.add_episode(Episode(
                transitions=[make_dummy_transition() for _ in range(4)],
                entity_observable=[True] * 4,
            ))
        train, val = ds.split(train_ratio=0.7, seed=0)
        self.assertEqual(len(train) + len(val), 10)
        self.assertEqual(len(train), 7)

    def test_mismatched_episode_lengths_raise(self):
        ds = EpisodeDataset()
        ds.add_episode(Episode(transitions=[make_dummy_transition()] * 3, entity_observable=[True] * 3))
        ds.add_episode(Episode(transitions=[make_dummy_transition()] * 5, entity_observable=[True] * 5))
        with self.assertRaises(ValueError):
            ds.to_numpy_arrays()


if __name__ == "__main__":
    unittest.main()
