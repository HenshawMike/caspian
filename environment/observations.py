"""Neutral observation interfaces and semantic firewall validation for Project Caspian.

Observations expose strictly neutral, measurable physical quantities.
Under the Semantic Firewall principle, no human semantic labels (such as 'food',
'reward', 'hazard', 'beneficial', 'danger') are permitted in the observation space.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math


FORBIDDEN_SEMANTIC_TOKENS = {
    "food",
    "hazard",
    "danger",
    "reward",
    "beneficial",
    "harmful",
    "resource",
    "target",
    "enemy",
    "good",
    "bad",
    "positive",
    "negative",
    "expected_outcome",
    "goal",
    "threat",
}


class SemanticFirewallViolation(ValueError):
    """Raised when an observation or representation leaks forbidden semantic terminology."""
    pass


@dataclass(frozen=True)
class EntityObservation:
    """Neutral perceptual observation of an environmental entity.

    Attributes:
        entity_id: Numeric instance identifier.
        entity_type: Numeric category code.
        relative_x: Relative x-coordinate from agent (entity_x - agent_x).
        relative_y: Relative y-coordinate from agent (entity_y - agent_y).
        distance: Euclidean or Manhattan distance from agent.
    """
    entity_id: int
    entity_type: int
    relative_x: int
    relative_y: int
    distance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity observation to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "relative_x": self.relative_x,
            "relative_y": self.relative_y,
            "distance": round(self.distance, 4),
        }

    def to_vector(self) -> List[float]:
        """Convert entity observation to numeric vector."""
        return [
            float(self.entity_id),
            float(self.entity_type),
            float(self.relative_x),
            float(self.relative_y),
            float(self.distance),
        ]


@dataclass(frozen=True)
class Observation:
    """Neutral observation object returned to the agent at each timestep.

    Attributes:
        agent_position: Discrete coordinates (x, y) of the agent.
        entities: List of neutral entity observations.
        collision: Boolean indicating if a movement collision occurred on the last step.
        internal_state: Current value of measurable state variable (e.g., energy level).
        timestep: Current discrete world timestep.
        previous_action: Integer code of previous action executed, or None at t=0.
    """
    agent_position: Tuple[int, int]
    entities: Tuple[EntityObservation, ...]
    collision: bool
    internal_state: float
    timestep: int
    previous_action: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to a clean dictionary structure."""
        return {
            "agent_position": list(self.agent_position),
            "entities": [e.to_dict() for e in self.entities],
            "collision": self.collision,
            "internal_state": float(self.internal_state),
            "timestep": self.timestep,
            "previous_action": self.previous_action,
        }

    def to_vector(self, max_entities: int = 5) -> List[float]:
        """Convert observation to a fixed-size numeric flat vector suitable for numeric models.

        Vector format:
        [agent_x, agent_y, collision_flag, internal_state, timestep, prev_action,
         (entity_id, entity_type, rel_x, rel_y, dist) * max_entities]
        """
        vec = [
            float(self.agent_position[0]),
            float(self.agent_position[1]),
            1.0 if self.collision else 0.0,
            float(self.internal_state),
            float(self.timestep),
            float(self.previous_action if self.previous_action is not None else -1.0),
        ]
        
        for i in range(max_entities):
            if i < len(self.entities):
                vec.extend(self.entities[i].to_vector())
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        return vec

    def assert_semantic_purity(self) -> None:
        """Audit the observation instance and verify zero semantic terminology exists."""
        obs_dict = self.to_dict()
        _check_semantic_purity_recursive(obs_dict)


def _check_semantic_purity_recursive(data: Any) -> None:
    """Recursively scan data structures for forbidden semantic tokens."""
    if isinstance(data, dict):
        for k, v in data.items():
            k_lower = str(k).lower()
            for token in FORBIDDEN_SEMANTIC_TOKENS:
                if token in k_lower:
                    raise SemanticFirewallViolation(
                        f"Semantic leakage detected in observation dictionary key: '{k}' contains '{token}'."
                    )
            _check_semantic_purity_recursive(v)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            _check_semantic_purity_recursive(item)
    elif isinstance(data, str):
        data_lower = data.lower()
        for token in FORBIDDEN_SEMANTIC_TOKENS:
            if token in data_lower:
                raise SemanticFirewallViolation(
                  f"Semantic leakage detected in observation string value: '{data}' contains '{token}'."
                )


def build_observation(
    agent_pos: Tuple[int, int],
    entities: List[Any],
    collision: bool,
    internal_state: float,
    timestep: int,
    previous_action: Optional[int],
) -> Observation:
    """Construct a clean, validated Observation instance.

    Args:
        agent_pos: (x, y) agent coordinates.
        entities: List of Entity instances present in the environment.
        collision: Collision flag from last transition.
        internal_state: Current measurable state value.
        timestep: Current discrete world timestep.
        previous_action: Integer code of previous action, or None.

    Returns:
        Observation: Strictly neutral observation object.
    """
    entity_obs_list: List[EntityObservation] = []
    for ent in entities:
        rel_x = ent.position[0] - agent_pos[0]
        rel_y = ent.position[1] - agent_pos[1]
        dist = math.sqrt(rel_x ** 2 + rel_y ** 2)
        entity_obs_list.append(
            EntityObservation(
                entity_id=ent.entity_id,
                entity_type=ent.entity_type,
                relative_x=rel_x,
                relative_y=rel_y,
                distance=dist,
            )
        )
    # Sort entities by entity_id for deterministic ordering
    entity_obs_list.sort(key=lambda e: e.entity_id)

    obs = Observation(
        agent_position=agent_pos,
        entities=tuple(entity_obs_list),
        collision=collision,
        internal_state=internal_state,
        timestep=timestep,
        previous_action=previous_action,
    )
    obs.assert_semantic_purity()
    return obs
