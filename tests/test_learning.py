"""Unit tests for dataset containers, collectors, and neural training loops."""

import unittest
import numpy as np

from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from agent.random_agent import RandomAgent
from learning.dataset import (
    ExperienceDataset,
    Transition,
    encode_state_action,
    one_hot_encode_action,
)
from learning.collector import TrajectoryCollector
from learning.trainer import PredictiveTrainer
from models.mlp import PredictiveMLP


class TestLearning(unittest.TestCase):
    """Test suite for data collection, encoding, and model training."""

    def test_one_hot_action(self):
        vec = one_hot_encode_action(Action.INTERACT, num_actions=6)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(vec, expected)

    def test_encode_state_action(self):
        world = GridWorld()
        obs = world.get_observation()
        encoded = encode_state_action(obs, Action.UP, max_entities=5, num_actions=6)
        # obs_vec (31) + action_one_hot (6) = 37
        self.assertEqual(encoded.shape, (37,))

    def test_trajectory_collector(self):
        collector = TrajectoryCollector()
        agent = RandomAgent(seed=42)
        dataset = collector.collect_with_agent(
            agent=agent,
            num_episodes=2,
            steps_per_episode=10,
            base_seed=100,
        )
        self.assertEqual(len(dataset), 20)

        X, Y = dataset.to_numpy_arrays()
        self.assertEqual(X.shape, (20, 37))
        self.assertEqual(Y.shape, (20, 1))

    def test_exploratory_collector(self):
        collector = TrajectoryCollector()
        dataset = collector.collect_exploratory_dataset(num_steps=50, interaction_prob=0.3, seed=42)
        self.assertEqual(len(dataset), 50)

    def test_trainer_convergence_on_synthetic_mapping(self):
        """Verify PredictiveTrainer reliably converges on a target relationship."""
        rng = np.random.default_rng(42)
        N = 200
        input_dim = 10

        # Synthetic relationship: y = 2.0 * x_0 - 3.0 * x_1 + noise
        X = rng.normal(0, 1, (N, input_dim))
        Y = (2.0 * X[:, 0:1] - 3.0 * X[:, 1:2])

        model = PredictiveMLP(
            input_dim=input_dim,
            hidden_dims=(16, 8),
            output_dim=1,
            hidden_activation="relu",
            seed=42,
        )

        trainer = PredictiveTrainer(
            model=model,
            batch_size=16,
            epochs=80,
            seed=42,
        )

        history = trainer.fit((X, Y))
        final_loss = history["final_train_loss"]

        # Assert loss decreased significantly
        self.assertLess(final_loss, history["train_losses"][0] * 0.1)
        self.assertLess(final_loss, 0.15)


if __name__ == "__main__":
    unittest.main()
