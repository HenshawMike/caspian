"""Base agent interface for Project Caspian.

Defines the abstract lifecycle and action selection protocol for all Caspian agents.
Agents interact with the environment strictly through neutral Observation and Action
interfaces, preserving the Semantic Firewall.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from environment.actions import Action
from environment.observations import Observation


class BaseAgent(ABC):
    """Abstract base class for all Caspian agents."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Select an action given the current neutral observation.

        Args:
            observation: Current neutral Observation object.

        Returns:
            Action: Selected Action enum member.
        """
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal agent state and random number generator.

        Args:
            seed: Optional random seed for reproducible behavior.
        """
        if seed is not None:
            self.seed = seed

    def observe(
        self,
        observation: Observation,
        state_delta: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Receive post-transition environment feedback.

        Phase 0 baselines do not train on feedback, but this hook establishes
        the standard lifecycle interface for future phases.

        Args:
            observation: Resulting neutral Observation object.
            state_delta: Change in internal measurable state (Delta E).
            done: Episode termination flag.
            info: Diagnostic metadata dictionary.
        """
        pass
