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
        )
    ])
    initial_internal_state: float = 100.0
    step_penalty: float = 1.0
    min_internal_state: float = 0.0
    max_internal_state: float = 100.0
    max_timesteps: int = 100
    interaction_radius: int = 1
    terminate_on_depletion: bool = False
    seed: Optional[int] = 42
    randomize_entity_positions: bool = False
    allowed_entity_positions: Optional[List[Tuple[int, int]]] = None
    entity_type_pool: Optional[List[Dict[str, Any]]] = None
    interact_cost: float = 0.0  # extra internal-state cost of every INTERACT action (action-consequence variation; 0.0 = original behavior)
    outcome_noise_std: float = 0.0  # Gaussian noise added to interaction outcomes only (0.0 = deterministic, original behavior)

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
            "seed": self.seed,
            "randomize_entity_positions": self.randomize_entity_positions,
            "allowed_entity_positions": (
                [list(p) for p in self.allowed_entity_positions] if self.allowed_entity_positions else None
            ),
            "entity_type_pool": self.entity_type_pool,
            "outcome_noise_std": self.outcome_noise_std,
            "interact_cost": self.interact_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldConfig":
        """Load config from dictionary."""
        entities = [
            Entity.from_dict(e) for e in data.get("initial_entities", [])
        ]
        agent_pos = tuple(data["initial_agent_pos"]) if data.get("initial_agent_pos") else None
        allowed_positions = data.get("allowed_entity_positions")
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
            seed=data.get("seed", 42),
            randomize_entity_positions=bool(data.get("randomize_entity_positions", False)),
            allowed_entity_positions=(
                [tuple(p) for p in allowed_positions] if allowed_positions else None
            ),
            entity_type_pool=data.get("entity_type_pool"),
            outcome_noise_std=float(data.get("outcome_noise_std", 0.0)),
            interact_cost=float(data.get("interact_cost", 0.0)),
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
    seed: Optional[int]


class GridWorld:
    """Deterministic 2D Grid World Environment."""

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

        # Initialize environment state
        self.reset(seed=self.config.seed)

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the world to its initial state deterministically.

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
                flicker_period=e.flicker_period,
                flicker_visible_steps=e.flicker_visible_steps,
                properties=e.properties,
            )
            for e in self.config.initial_entities
        ]

        if self.config.randomize_entity_positions:
            # Re-place each entity at a fresh random cell each episode (distinct from the
            # agent's start cell), so a model cannot learn a fixed global position -> outcome
            # lookup across episodes and must instead track the entity from within-episode
            # observations (Phase 2 partial-observability requirement).
            #
            # When allowed_entity_positions is set (Phase 3 — Generalization), sampling is
            # restricted to that subset of cells, letting train/test configs use disjoint
            # position sets to test whether a learned relationship transfers to unseen cells.
            occupied = {self._agent_pos}
            new_entities = []
            candidate_cells = self.config.allowed_entity_positions
            for e in self._entities:
                while True:
                    if candidate_cells:
                        pos = candidate_cells[self._rng.randint(0, len(candidate_cells) - 1)]
                    else:
                        pos = (
                            self._rng.randint(0, self.config.grid_width - 1),
                            self._rng.randint(0, self.config.grid_height - 1),
                        )
                    if pos not in occupied:
                        occupied.add(pos)
                        break
                new_entities.append(
                    Entity(
                        entity_id=e.entity_id, entity_type=e.entity_type, position=pos,
                        is_interactive=e.is_interactive, is_blocking=e.is_blocking,
                        hidden_state_delta=e.hidden_state_delta,
                        flicker_period=e.flicker_period, flicker_visible_steps=e.flicker_visible_steps,
                        properties=e.properties,
                    )
                )
            self._entities = new_entities

        if self.config.entity_type_pool:
            # Resample each entity's FUNCTIONAL identity (type code + hidden_state_delta +
            # interactive/blocking flags) each episode from a pool of templates (Phase 5 --
            # Emergent Categories). This is independent of position randomization above: it
            # varies which *kind* of entity appears, not where. entity_id and position are
            # left untouched here so this composes cleanly with randomize_entity_positions.
            pool = self.config.entity_type_pool
            self._entities = [
                Entity(
                    entity_id=e.entity_id,
                    entity_type=int(template["entity_type"]),
                    position=e.position,
                    is_interactive=bool(template.get("is_interactive", True)),
                    is_blocking=bool(template.get("is_blocking", False)),
                    hidden_state_delta=float(template["hidden_state_delta"]),
                    flicker_period=e.flicker_period,
                    flicker_visible_steps=e.flicker_visible_steps,
                    properties=tuple(float(p) for p in template.get("properties", ())),
                )
                for e, template in (
                    (e, pool[self._rng.randint(0, len(pool) - 1)]) for e in self._entities
                )
            ]

        self._internal_state = float(self.config.initial_internal_state)
        self._timestep = 0
        self._collision = False
        self._interaction_occurred = False
        self._last_action = None
        self._is_terminated = False

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

        # Compute interaction delta and interaction flag
        interaction_delta, interaction_occurred = compute_interaction_delta(
            agent_pos=new_pos,
            action=validated_action,
            entities=self._entities,
            interaction_radius=self.config.interaction_radius,
        )

        if validated_action == Action.INTERACT and self.config.interact_cost != 0.0:
            interaction_delta -= self.config.interact_cost
        if interaction_occurred and self.config.outcome_noise_std > 0.0:
            interaction_delta += self._rng.gauss(0.0, self.config.outcome_noise_std)

        # Compute updated internal state
        prev_internal_state = self._internal_state
        new_internal_state = update_internal_state(
            current_state_val=self._internal_state,
            step_penalty=self.config.step_penalty,
            interaction_delta=interaction_delta,
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
        self._timestep += 1

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
            "observable_entity_ids": [
                e.entity_id for e in self._entities if e.is_observable(self._timestep)
            ],
        }

        return obs, state_delta, terminated, info

    def get_observation(self) -> Observation:
        """Construct the neutral observation object for current state.

        Only entities that are currently observable (``Entity.is_observable``)
        are included in the returned Observation's entity list; this is the sole
        mechanism by which Phase 2 partial-observability tasks are constructed.
        Ground-truth dynamics (collision, interaction) always use the full
        entity list regardless of observability — see step().
        """
        visible_entities = [e for e in self._entities if e.is_observable(self._timestep)]
        return build_observation(
            agent_pos=self._agent_pos,
            entities=visible_entities,
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
            seed=self._seed,
        )
