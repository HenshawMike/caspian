r"""Pure transition dynamics and state update equations for Project Caspian.

Mathematical formulation:
Let S_t = (p_t, E_t, t, \mathcal{E}, c_t, i_t, a_{t-1}) be the environment state at timestep t.
Let A_t be the action selected from {UP, DOWN, LEFT, RIGHT, INTERACT, NOOP}.

Transition equations:
1. Candidate position:
   p' = p_t + \Delta(A_t)
   where \Delta(UP)=(0,1), \Delta(DOWN)=(0,-1), \Delta(LEFT)=(-1,0), \Delta(RIGHT)=(1,0), \Delta(INTERACT/NOOP)=(0,0).

2. Boundary & collision check:
   If p' \notin [0, W-1] \times [0, H-1] or \exists e \in \mathcal{E}: pos(e)=p' \land blocking(e):
       p_{t+1} = p_t
       collision_{t+1} = True
   Else:
       p_{t+1} = p'
       collision_{t+1} = False

3. Interaction & internal state (energy) transition:
   \Delta E_{interaction} = \sum_{e \in \mathcal{E}_{interactable}(p_{t+1})} \delta(e) \quad \text{if } A_t = \text{INTERACT} \text{ else } 0
   E_{t+1} = \text{clamp}(E_t - \delta_{step} + \Delta E_{interaction}, E_{min}, E_{max})

4. Timestep update:
   t_{t+1} = t_t + 1
"""

from typing import Tuple, List, Optional
from environment.actions import Action
from environment.entities import Entity


def compute_next_position(
    current_pos: Tuple[int, int],
    action: Action,
    grid_width: int,
    grid_height: int,
    entities: List[Entity],
) -> Tuple[Tuple[int, int], bool]:
    """Calculate the next agent position and collision status.

    Args:
        current_pos: Current (x, y) coordinates of the agent.
        action: Action to be executed.
        grid_width: Width of the grid world (W).
        grid_height: Height of the grid world (H).
        entities: List of entities in the environment.

    Returns:
        Tuple[Tuple[int, int], bool]: (new_position, collision_occurred)
    """
    dx, dy = action.get_delta()
    candidate_x = current_pos[0] + dx
    candidate_y = current_pos[1] + dy

    # Check grid boundary collision
    if candidate_x < 0 or candidate_x >= grid_width or candidate_y < 0 or candidate_y >= grid_height:
        return current_pos, True

    # Check blocking entity collision
    candidate_pos = (candidate_x, candidate_y)
    for entity in entities:
        if entity.position == candidate_pos and entity.is_blocking:
            return current_pos, True

    return candidate_pos, False


def compute_interaction_events(
    agent_pos: Tuple[int, int],
    action: Action,
    entities: List[Entity],
    interaction_radius: int = 1,
) -> List[Tuple[Entity, float, int]]:
    """Compute triggered interaction events including hidden state deltas and delays.

    Args:
        agent_pos: Agent (x, y) coordinates.
        action: Executed action.
        entities: List of entities in the environment.
        interaction_radius: Maximum Manhattan distance to interact with an entity.

    Returns:
        List[Tuple[Entity, float, int]]: List of (entity, hidden_state_delta, interaction_delay).
    """
    if action != Action.INTERACT:
        return []

    events = []
    for entity in entities:
        if not entity.is_interactive:
            continue
        manhattan_dist = abs(entity.position[0] - agent_pos[0]) + abs(entity.position[1] - agent_pos[1])
        if manhattan_dist <= interaction_radius:
            events.append((entity, entity.hidden_state_delta, entity.interaction_delay))

    return events


def compute_interaction_delta(
    agent_pos: Tuple[int, int],
    action: Action,
    entities: List[Entity],
    interaction_radius: int = 1,
) -> Tuple[float, bool]:
    """Compute internal state change from interaction and interaction occurrence flag.

    Args:
        agent_pos: Agent (x, y) coordinates.
        action: Executed action.
        entities: List of entities in the environment.
        interaction_radius: Maximum Manhattan distance to interact with an entity.

    Returns:
        Tuple[float, bool]: (total_state_delta, interaction_occurred)
    """
    events = compute_interaction_events(agent_pos, action, entities, interaction_radius)
    if not events:
        return 0.0, False

    total_delta = sum(delta for _, delta, delay in events if delay == 0)
    interacted = len(events) > 0
    return total_delta, interacted


def update_internal_state(
    current_state_val: float,
    step_penalty: float,
    interaction_delta: float,
    min_val: float = 0.0,
    max_val: float = 100.0,
) -> float:
    """Apply step penalty, interaction delta, and clamp internal state variable.

    Args:
        current_state_val: Current internal state value (e.g. energy).
        step_penalty: Fixed baseline step decrement (delta_step >= 0).
        interaction_delta: Net change from interaction.
        min_val: Minimum permitted state value.
        max_val: Maximum permitted state value.

    Returns:
        float: Clamped updated internal state value.
    """
    updated = current_state_val - step_penalty + interaction_delta
    clamped = max(min_val, min(max_val, updated))
    return round(clamped, 6)
