"""Random baseline agent for Project Caspian.

Uniformly samples actions from the permitted action space using an isolated,
deterministic pseudo-random number generator.
"""

import random
from typing import Optional, List

from agent.base import BaseAgent
from environment.actions import Action
from environment.observations import Observation


class RandomAgent(BaseAgent):
    """Agent that chooses actions uniformly at random from the action space."""

    def __init__(
        self,
        seed: Optional[int] = 42,
        action_space: Optional[List[Action]] = None,
    ):
        super().__init__(seed=seed)
        self.action_space = action_space if action_space is not None else list(Action)
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal RNG state deterministically."""
        super().reset(seed=seed)
        self._rng.seed(self.seed)

    def act(self, observation: Observation) -> Action:
        """Sample a uniform random action from the configured action space.

        Args:
            observation: Current neutral observation (unused by random policy).

        Returns:
            Action: Uniformly chosen Action enum member.
        """
        return self._rng.choice(self.action_space)
