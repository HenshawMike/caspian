"""Entity representations and properties for Project Caspian.

Entities in the world are defined using purely neutral identifiers and physical
attributes. No human semantic categories (such as 'food', 'danger', 'resource')
are exposed or embedded in entity representations.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass(frozen=True)
class Entity:
    """Neutral environmental entity representation.

    Attributes:
        entity_id: Unique integer identifier for this specific entity instance.
        entity_type: Numeric code identifying the entity category/type.
        position: Discrete 2D coordinates (x, y) on the grid.
        is_interactive: Whether the entity responds to the INTERACT action.
        is_blocking: Whether the entity blocks agent movement into its cell.
        hidden_state_delta: Magnitude of change to internal state upon interaction (hidden rule).
    """
    entity_id: int
    entity_type: int
    position: Tuple[int, int]
    is_interactive: bool = True
    is_blocking: bool = False
    hidden_state_delta: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity properties to dictionary (internal state)."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "position": self.position,
            "is_interactive": self.is_interactive,
            "is_blocking": self.is_blocking,
            "hidden_state_delta": self.hidden_state_delta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Deserialize entity from dictionary."""
        return cls(
            entity_id=int(data["entity_id"]),
            entity_type=int(data["entity_type"]),
            position=tuple(data["position"]),
            is_interactive=bool(data.get("is_interactive", True)),
            is_blocking=bool(data.get("is_blocking", False)),
            hidden_state_delta=float(data.get("hidden_state_delta", 0.0)),
        )
