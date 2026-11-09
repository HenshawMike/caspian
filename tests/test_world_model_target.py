import unittest
import numpy as np
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.actions import Action
from learning.dataset import encode_world_model_target, Transition, ExperienceDataset


def make_world():
    config = WorldConfig(
        grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
        initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2), is_blocking=True)],
        max_timesteps=50, seed=1,
    )
    return GridWorld(config=config)


class TestWorldModelTarget(unittest.TestCase):
    def test_movement_delta_matches_actual_position_change(self):
        world = make_world()
        obs = world.reset(seed=1)
        next_obs, delta, done, info = world.step(Action.RIGHT)

        target = encode_world_model_target(obs, next_obs)
        expected_dx = next_obs.agent_position[0] - obs.agent_position[0]
        expected_dy = next_obs.agent_position[1] - obs.agent_position[1]

        self.assertEqual(target[0], expected_dx)
        self.assertEqual(target[1], expected_dy)
        self.assertAlmostEqual(target[2], delta)

    def test_blocked_move_yields_zero_position_delta(self):
        # entity at (2,2) is_blocking=True; walk the agent to (1,2) then try to move
        # into (2,2) -- should be blocked (position unchanged) even though the action
        # was a movement action.
        world = make_world()
        obs = world.reset(seed=1)
        for act in [Action.RIGHT, Action.RIGHT, Action.UP, Action.UP]:
            obs, _, _, _ = world.step(act)
        self.assertEqual(obs.agent_position, (2, 1))

        next_obs, delta, done, info = world.step(Action.UP)  # would move into (2,2), blocked
        target = encode_world_model_target(obs, next_obs)
        self.assertEqual(target[0], 0.0)
        self.assertEqual(target[1], 0.0)
        self.assertTrue(next_obs.collision)

    def test_world_model_arrays_shape(self):
        world = make_world()
        obs = world.reset(seed=1)
        dataset = ExperienceDataset()
        for act in [Action.RIGHT, Action.UP, Action.NOOP, Action.INTERACT]:
            next_obs, delta, done, info = world.step(act)
            dataset.add(Transition(obs=obs, action=act, state_delta=delta, next_obs=next_obs, done=done, info=info))
            obs = next_obs

        X, Y = dataset.to_world_model_arrays()
        self.assertEqual(X.shape, (4, 37))
        self.assertEqual(Y.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
