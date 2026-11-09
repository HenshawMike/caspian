"""Entity representations and properties for Project Caspian.

Entities in the world are defined using purely neutral identifiers and physical
attributes. No human semantic categories (such as 'food', 'danger', 'resource')
are exposed or embedded in entity representations.

Phase 2 (Memory & Persistence) adds an optional periodic-observability rule to
entities: an entity may be physically present and fully affect world dynamics
(collision, interaction) at all times, while being included in the agent's
*observation* only during specific timesteps. This is the mechanism used to
construct situations where "current observations alone are insufficient"
(Development Plan, Section 29) — the agent must retain information about an
entity across the timesteps during which it is not observable. The rule itself
(period / visible-steps) is never exposed to the agent; only its consequence
(the entity's absence from the observed entity list) is.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional


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
        flicker_period: If set, the entity's observability cycles with this period
            (in timesteps). None means the entity is always observable (Phase 0/1
            default — fully backward compatible).
        flicker_visible_steps: Number of timesteps per period during which the
            entity is observable (observable when ``timestep % flicker_period <
            flicker_visible_steps``). Ignored when flicker_period is None.
        properties: Optional tuple of neutral continuous numbers describing the
            entity's physical attributes (Phase 5 — Emergent Categories, property-based
            encoding). These are NOT a semantic label and NOT the category code: they are
            arbitrary measurable quantities (e.g. a texture/appearance reading) that may or
            may not relate to the entity's hidden_state_delta. Empty tuple by default
            (fully backward compatible with every world defined before this field existed).
    """
    entity_id: int
    entity_type: int
    position: Tuple[int, int]
    is_interactive: bool = True
    is_blocking: bool = False
    hidden_state_delta: float = 10.0
    flicker_period: Optional[int] = None
    flicker_visible_steps: int = 0
    properties: Tuple[float, ...] = ()

    def is_observable(self, timestep: int) -> bool:
        """Determine whether this entity appears in the observation at a given timestep.

        Note: observability governs only what is *reported* to the agent. It has
        no effect on collision or interaction dynamics, which always use the
        entity's true (ground-truth) state.

        Args:
            timestep: Current discrete world timestep.

        Returns:
            bool: True if the entity should be included in the neutral observation.
        """
        if self.flicker_period is None or self.flicker_period <= 0:
            return True
        return (timestep % self.flicker_period) < self.flicker_visible_steps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity properties to dictionary (internal state)."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "position": self.position,
            "is_interactive": self.is_interactive,
            "is_blocking": self.is_blocking,
            "hidden_state_delta": self.hidden_state_delta,
            "flicker_period": self.flicker_period,
            "flicker_visible_steps": self.flicker_visible_steps,
            "properties": list(self.properties),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Deserialize entity from dictionary."""
        flicker_period = data.get("flicker_period", None)
        return cls(
            entity_id=int(data["entity_id"]),
            entity_type=int(data["entity_type"]),
            position=tuple(data["position"]),
            is_interactive=bool(data.get("is_interactive", True)),
            is_blocking=bool(data.get("is_blocking", False)),
            hidden_state_delta=float(data.get("hidden_state_delta", 0.0)),
            flicker_period=int(flicker_period) if flicker_period is not None else None,
            flicker_visible_steps=int(data.get("flicker_visible_steps", 0)),
            properties=tuple(float(p) for p in data.get("properties", ())),
        )
