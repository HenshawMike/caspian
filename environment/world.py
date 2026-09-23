"""Core GridWorld environment implementation for Project Caspian.

Provides a deterministic 2D discrete grid world supporting:
- Agent position & movement
- Environmental entities with hidden interaction rules
- Boundaries & collision detection
- Measurable internal state variable (energy)
- Strictly neutral observations adhering to the Semantic Firewall
- Deterministic reset and reproducibility under fixed seeds
"""

from dataclasses import dataclass, field
import random
from typing import Tuple, List, Optional, Dict, Any, Union

from environment.actions import Action, InvalidActionError
from environment.entities import Entity
from environment.observations import Observation, build_observation
from environment.dynamics import (
    compute_next_position,
    compute_interaction_delta,
    compute_interaction_events,
    update_internal_state,
)


@dataclass
class WorldConfig:
    """Configuration specification for the Caspian 2D Grid World."""
    grid_width: int = 5
    grid_height: int = 5
    initial_agent_pos: Optional[Tuple[int, int]] = (0, 0)
    initial_entities: List[Entity] = field(default_factory=lambda: [
        Entity(
            entity_id=1,
            entity_type=1,
            position=(2, 2),
            is_interactive=True,
            is_blocking=False,
            hidden_state_delta=10.0,
            interaction_delay=0,
        )
    ])
    initial_internal_state: float = 100.0
    step_penalty: float = 1.0
    min_internal_state: float = 0.0
    max_internal_state: float = 100.0
    max_timesteps: int = 100
    interaction_radius: int = 1
    terminate_on_depletion: bool = False
    default_interaction_delay: int = 0
    seed: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "initial_agent_pos": list(self.initial_agent_pos) if self.initial_agent_pos else None,
            "initial_entities": [e.to_dict() for e in self.initial_entities],
            "initial_internal_state": self.initial_internal_state,
            "step_penalty": self.step_penalty,
            "min_internal_state": self.min_internal_state,
            "max_internal_state": self.max_internal_state,
            "max_timesteps": self.max_timesteps,
            "interaction_radius": self.interaction_radius,
            "terminate_on_depletion": self.terminate_on_depletion,
            "default_interaction_delay": self.default_interaction_delay,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldConfig":
        """Load config from dictionary."""
        entities = [
            Entity.from_dict(e) for e in data.get("initial_entities", [])
        ]
        agent_pos = tuple(data["initial_agent_pos"]) if data.get("initial_agent_pos") else None
        return cls(
            grid_width=int(data.get("grid_width", 5)),
            grid_height=int(data.get("grid_height", 5)),
            initial_agent_pos=agent_pos,
            initial_entities=entities,
            initial_internal_state=float(data.get("initial_internal_state", 100.0)),
            step_penalty=float(data.get("step_penalty", 1.0)),
            min_internal_state=float(data.get("min_internal_state", 0.0)),
            max_internal_state=float(data.get("max_internal_state", 100.0)),
            max_timesteps=int(data.get("max_timesteps", 100)),
            interaction_radius=int(data.get("interaction_radius", 1)),
            terminate_on_depletion=bool(data.get("terminate_on_depletion", False)),
            default_interaction_delay=int(data.get("default_interaction_delay", 0)),
            seed=data.get("seed", 42),
        )


@dataclass(frozen=True)
class EnvironmentState:
    """Full internal ground-truth state of the environment."""
    agent_position: Tuple[int, int]
    internal_state: float
    timestep: int
    entities: Tuple[Entity, ...]
    collision: bool
    interaction_occurred: bool
    last_action: Optional[Action]
    is_terminated: bool
    pending_events: Tuple[Tuple[int, float], ...]
    seed: Optional[int]


class GridWorld:
    """Deterministic 2D Grid World Environment supporting temporal delayed interactions."""

    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config if config is not None else WorldConfig()
        self._rng = random.Random()
        self._seed: Optional[int] = None
        self._agent_pos: Tuple[int, int] = (0, 0)
        self._entities: List[Entity] = []
        self._internal_state: float = self.config.initial_internal_state
        self._timestep: int = 0
        self._collision: bool = False
        self._interaction_occurred: bool = False
        self._last_action: Optional[Action] = None
        self._is_terminated: bool = False
        self._pending_events: List[Tuple[int, float]] = []

        # Initialize environment state
        self.reset(seed=self.config.seed)

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the world to its initial state deterministically.

        Ensures 100% episode isolation (clears all pending queues and internal memory).

        Args:
            seed: Random seed for initialization. If None, uses config.seed.

        Returns:
            Observation: Initial neutral observation at timestep 0.
        """
        effective_seed = seed if seed is not None else self.config.seed
        self._seed = effective_seed
        self._rng.seed(effective_seed)

        # Initialize agent position
        if self.config.initial_agent_pos is not None:
            self._agent_pos = self.config.initial_agent_pos
        else:
            # Deterministic random placement if initial position not fixed
            rx = self._rng.randint(0, self.config.grid_width - 1)
            ry = self._rng.randint(0, self.config.grid_height - 1)
            self._agent_pos = (rx, ry)

        # Clone configured entities
        self._entities = [
            Entity(
                entity_id=e.entity_id,
                entity_type=e.entity_type,
                position=e.position,
                is_interactive=e.is_interactive,
                is_blocking=e.is_blocking,
                hidden_state_delta=e.hidden_state_delta,
                interaction_delay=e.interaction_delay if e.interaction_delay != 0 else self.config.default_interaction_delay,
            )
            for e in self.config.initial_entities
        ]

        self._internal_state = float(self.config.initial_internal_state)
        self._timestep = 0
        self._collision = False
        self._interaction_occurred = False
        self._last_action = None
        self._is_terminated = False
        self._pending_events = []

        return self.get_observation()

    def step(self, action: Union[Action, int, str]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute one discrete transition in the environment.

        Args:
            action: Action enum member, integer code, or string representation.

        Returns:
            Tuple containing:
            - Observation: Neutral observation of the new state.
            - float: Measurable change in internal state (Delta E).
            - bool: Termination flag (done).
            - Dict[str, Any]: Diagnostic info (non-semantic metadata).

        Raises:
            InvalidActionError: If action is invalid or unrecognized.
            RuntimeError: If step is called on a terminated environment without reset.
        """
        if self._is_terminated:
            raise RuntimeError(
                "Environment is terminated. Call reset() before taking further steps."
            )

        # Validate and parse action cleanly
        validated_action = Action.parse(action)

        # Compute next position and collision status
        new_pos, collision = compute_next_position(
            current_pos=self._agent_pos,
            action=validated_action,
            grid_width=self.config.grid_width,
            grid_height=self.config.grid_height,
            entities=self._entities,
        )

        target_timestep = self._timestep + 1

        # Check for immediate or delayed interaction events
        interaction_events = compute_interaction_events(
            agent_pos=new_pos,
            action=validated_action,
            entities=self._entities,
            interaction_radius=self.config.interaction_radius,
        )

        immediate_delta = 0.0
        interaction_occurred = len(interaction_events) > 0

        for entity, hidden_delta, delay in interaction_events:
            effective_delay = delay if delay != 0 else self.config.default_interaction_delay
            if effective_delay == 0:
                immediate_delta += hidden_delta
            else:
                # Schedule delayed consequence at target timestep
                delivery_time = target_timestep + effective_delay
                self._pending_events.append((delivery_time, hidden_delta))

        # Check pending events maturing at this target timestep
        delayed_delta = 0.0
        remaining_events: List[Tuple[int, float]] = []
        for delivery_time, delta in self._pending_events:
            if delivery_time == target_timestep:
                delayed_delta += delta
            else:
                remaining_events.append((delivery_time, delta))
        self._pending_events = remaining_events

        total_interaction_delta = immediate_delta + delayed_delta

        # Compute updated internal state
        prev_internal_state = self._internal_state
        new_internal_state = update_internal_state(
            current_state_val=self._internal_state,
            step_penalty=self.config.step_penalty,
            interaction_delta=total_interaction_delta,
            min_val=self.config.min_internal_state,
            max_val=self.config.max_internal_state,
        )
        state_delta = round(new_internal_state - prev_internal_state, 6)

        # Apply state updates
        self._agent_pos = new_pos
        self._collision = collision
        self._interaction_occurred = interaction_occurred
        self._internal_state = new_internal_state
        self._last_action = validated_action
        self._timestep = target_timestep

        # Check termination conditions
        terminated = False
        if self._timestep >= self.config.max_timesteps:
            terminated = True
        elif self.config.terminate_on_depletion and self._internal_state <= self.config.min_internal_state:
            terminated = True

        self._is_terminated = terminated

        obs = self.get_observation()
        info: Dict[str, Any] = {
            "timestep": self._timestep,
            "collision": self._collision,
            "interaction": self._interaction_occurred,
            "pending_events_count": len(self._pending_events),
        }

        return obs, state_delta, terminated, info

    def get_observation(self) -> Observation:
        """Construct the neutral observation object for current state."""
        return build_observation(
            agent_pos=self._agent_pos,
            entities=self._entities,
            collision=self._collision,
            internal_state=self._internal_state,
            timestep=self._timestep,
            previous_action=int(self._last_action) if self._last_action is not None else None,
        )

    def get_state(self) -> EnvironmentState:
        """Retrieve complete ground-truth internal environment state."""
        return EnvironmentState(
            agent_position=self._agent_pos,
            internal_state=self._internal_state,
            timestep=self._timestep,
            entities=tuple(self._entities),
            collision=self._collision,
            interaction_occurred=self._interaction_occurred,
            last_action=self._last_action,
            is_terminated=self._is_terminated,
            pending_events=tuple(self._pending_events),
            seed=self._seed,
        )
