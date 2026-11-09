import unittest
import numpy as np
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.actions import Action
from evaluation.world_model_rollout import simulate_rollout_open_loop, plan_action


class PerfectMockModel:
    """A stand-in 'world model' that always predicts the exact ground-truth delta for a
    known, simple scenario: every movement action moves by exactly its get_delta(), with
    no blocking, and every step costs exactly -1 internal state. Used to test that the
    rollout/planning harness's reconstruction math is correct, independent of learning.
    """
    def forward(self, X: np.ndarray) -> np.ndarray:
        # The last 6 columns of the 37-dim feature vector are the one-hot action.
        n = X.shape[0]
        out = np.zeros((n, 3))
        action_onehot = X[:, -6:]
        action_codes = np.argmax(action_onehot, axis=1)
        deltas = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (0, 0), 5: (0, 0)}  # UP,DOWN,LEFT,RIGHT,INTERACT,NOOP
        for i, code in enumerate(action_codes):
            dx, dy = deltas[int(code)]
            out[i] = [dx, dy, -1.0]
        return out


def make_world():
    config = WorldConfig(
        grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
        initial_entities=[Entity(entity_id=1, entity_type=1, position=(4, 4), is_blocking=False)],
        max_timesteps=50, seed=1,
    )
    return GridWorld(config=config)


class TestRolloutSimulation(unittest.TestCase):
    def test_perfect_model_reproduces_ground_truth_trajectory(self):
        world = make_world()
        obs = world.reset(seed=1)
        entities = list(world._entities)

        actions = [Action.RIGHT, Action.RIGHT, Action.UP, Action.UP]
        model = PerfectMockModel()
        rollout = simulate_rollout_open_loop(
            model, obs, actions, entities, grid_width=5, grid_height=5, min_state=0.0, max_state=100.0,
        )

        # Compare against actually stepping the real world with the same actions.
        true_positions = [obs.agent_position]
        true_states = [obs.internal_state]
        for a in actions:
            o, _, _, _ = world.step(a)
            true_positions.append(o.agent_position)
            true_states.append(o.internal_state)

        for h, (sim_obs, true_pos, true_state) in enumerate(zip(rollout, true_positions, true_states)):
            self.assertEqual(sim_obs.agent_position, true_pos, msg=f"horizon step {h}")
            self.assertAlmostEqual(sim_obs.internal_state, true_state, msg=f"horizon step {h}")

    def test_rollout_length_matches_action_sequence(self):
        world = make_world()
        obs = world.reset(seed=1)
        entities = list(world._entities)
        actions = [Action.NOOP] * 5
        rollout = simulate_rollout_open_loop(
            PerfectMockModel(), obs, actions, entities, grid_width=5, grid_height=5, min_state=0.0, max_state=100.0,
        )
        self.assertEqual(len(rollout), 6)  # initial + 5 steps


class TestPlanner(unittest.TestCase):
    def test_planner_picks_action_with_highest_predicted_reward(self):
        """A mock model that scores RIGHT far higher than everything else should be planned for."""
        class BiasedMockModel:
            def forward(self, X):
                action_codes = np.argmax(X[:, -6:], axis=1)
                out = np.zeros((X.shape[0], 3))
                for i, code in enumerate(action_codes):
                    out[i, 2] = 10.0 if int(code) == int(Action.RIGHT) else -1.0
                return out

        world = make_world()
        obs = world.reset(seed=1)
        entities = list(world._entities)

        chosen = plan_action(
            BiasedMockModel(), obs, entities, grid_width=5, grid_height=5,
            min_state=0.0, max_state=100.0, horizon=2,
        )
        self.assertEqual(chosen, Action.RIGHT)


if __name__ == "__main__":
    unittest.main()
