"""Scripted / Oracle baseline agent for Project Caspian.

Provides an optimal rule-based baseline that navigates to entities using neutral
observation geometry and executes interactions to maintain internal state.
"""

from typing import Optional

from agent.base import BaseAgent
from environment.actions import Action
from environment.observations import Observation


class OracleAgent(BaseAgent):
    """Rule-based optimal navigation and interaction baseline agent."""

    def __init__(
        self,
        target_entity_type: Optional[int] = None,
        interaction_radius: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.target_entity_type = target_entity_type
        self.interaction_radius = interaction_radius

    def act(self, observation: Observation) -> Action:
        """Select optimal navigation or interaction action.

        Policy:
        1. Identify closest matching entity from observation entities list.
        2. If within Manhattan interaction radius (|rel_x| + |rel_y| <= radius):
           Execute Action.INTERACT.
        3. Else navigate along axis with largest displacement towards entity.

        Args:
            observation: Current neutral Observation object.

        Returns:
            Action: Selected Action enum member.
        """
        if not observation.entities:
            return Action.NOOP

        # Find target entity
        target = None
        if self.target_entity_type is not None:
            for ent in observation.entities:
                if ent.entity_type == self.target_entity_type:
                    target = ent
                    break

        if target is None:
            # Default to closest entity by distance
            target = min(observation.entities, key=lambda e: e.distance)

        manhattan_dist = abs(target.relative_x) + abs(target.relative_y)

        # In interaction range -> INTERACT
        if manhattan_dist <= self.interaction_radius:
            return Action.INTERACT

        # Navigation: prioritize larger coordinate offset
        dx = target.relative_x
        dy = target.relative_y

        if abs(dx) >= abs(dy) and dx != 0:
            return Action.RIGHT if dx > 0 else Action.LEFT
        elif dy != 0:
            return Action.UP if dy > 0 else Action.DOWN
        elif dx != 0:
            return Action.RIGHT if dx > 0 else Action.LEFT

        return Action.NOOP


# Alias for clarity
ScriptedAgent = OracleAgent
