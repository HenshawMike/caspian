"""Trajectory collection utilities for Project Caspian.

Collects environmental transitions across episodes and seeds while enforcing
the Semantic Firewall.
"""

from typing import List, Optional, Union
import random

from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from agent.base import BaseAgent
from agent.random_agent import RandomAgent
from learning.dataset import ExperienceDataset, Transition


class TrajectoryCollector:
    """Collects environmental interaction trajectories into ExperienceDataset."""

    def __init__(self, world_config: Optional[WorldConfig] = None):
        self.world_config = world_config if world_config is not None else WorldConfig()

    def collect_with_agent(
        self,
        agent: BaseAgent,
        num_episodes: int = 10,
        steps_per_episode: int = 50,
        base_seed: int = 42,
    ) -> ExperienceDataset:
        """Collect transitions using a given agent policy across multiple episodes.

        Args:
            agent: BaseAgent instance to select actions.
            num_episodes: Number of episodes to simulate.
            steps_per_episode: Maximum steps per episode.
            base_seed: Base seed for reproducible environment runs.

        Returns:
            ExperienceDataset: Populated dataset of transitions.
        """
        dataset = ExperienceDataset()

        for ep in range(num_episodes):
            ep_seed = base_seed + ep
            world = GridWorld(config=self.world_config)
            agent.reset(seed=ep_seed)
            world.reset(seed=ep_seed)

            obs = world.get_observation()
            for _ in range(steps_per_episode):
                action = agent.act(obs)
                next_obs, delta, done, info = world.step(action)
                agent.observe(next_obs, delta, done, info)

                transition = Transition(
                    obs=obs,
                    action=action,
                    state_delta=delta,
                    next_obs=next_obs,
                    done=done,
                    info=info,
                )
                dataset.add(transition)

                obs = next_obs
                if done:
                    break

        return dataset

    def collect_exploratory_dataset(
        self,
        num_steps: int = 500,
        interaction_prob: float = 0.25,
        seed: int = 42,
    ) -> ExperienceDataset:
        """Collect balanced exploratory dataset with controlled interaction frequency.

        Args:
            num_steps: Total number of environmental interaction steps.
            interaction_prob: Probability of choosing INTERACT when near an entity or randomly.
            seed: Seed for reproducibility.

        Returns:
            ExperienceDataset: Balanced dataset of movements, collisions, and interactions.
        """
        dataset = ExperienceDataset()
        rng = random.Random(seed)
        world = GridWorld(config=self.world_config)
        world.reset(seed=seed)

        obs = world.get_observation()
        movement_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.NOOP]

        for _ in range(num_steps):
            # Select action: with interaction_prob choose INTERACT, otherwise random movement
            if rng.random() < interaction_prob:
                act = Action.INTERACT
            else:
                act = rng.choice(movement_actions)

            next_obs, delta, done, info = world.step(act)
            transition = Transition(
                obs=obs,
                action=act,
                state_delta=delta,
                next_obs=next_obs,
                done=done,
                info=info,
            )
            dataset.add(transition)

            if done:
                world.reset(seed=rng.randint(0, 100000))
                obs = world.get_observation()
            else:
                obs = next_obs

        return dataset
