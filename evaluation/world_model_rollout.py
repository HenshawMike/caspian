"""Multi-step rollout simulation and lookahead planning for Project Caspian (Phase 4).

Phase 4's defining capability (Section 31 of the Development Plan): chain a
single-step world model's own predictions into a simulated multi-step future
(open-loop rollout: Ê_{t+1} -> Ê_{t+2} -> Ê_{t+3} -> ...), and use that
simulated future to choose actions (planning), rather than only ever reacting
to the single next step as Phase 1-3's agents do.

Entity positions are carried forward unchanged during simulation: entities are
static in this environment (verified in CSP-P4-E001, not assumed), so there is
nothing to predict for them. Only agent position and internal_state are
predicted and fed back in as the next step's input.
"""

from itertools import product
from typing import List, Tuple, Optional, Sequence
import numpy as np

from environment.actions import Action
from environment.entities import Entity
from environment.observations import Observation, build_observation
from learning.dataset import encode_state_action


def _step_observation(
    model,
    obs: Observation,
    action: Action,
    entities: Sequence[Entity],
    grid_width: int,
    grid_height: int,
    min_state: float,
    max_state: float,
) -> Observation:
    """Advance one simulated step: predict [dx, dy, d_state] and reconstruct the next Observation."""
    feature = encode_state_action(obs, action)
    pred = model.forward(feature[None, :])[0]  # [dx, dy, d_state]
    dx, dy, d_state = float(pred[0]), float(pred[1]), float(pred[2])

    new_x = int(np.clip(round(obs.agent_position[0] + dx), 0, grid_width - 1))
    new_y = int(np.clip(round(obs.agent_position[1] + dy), 0, grid_height - 1))
    new_internal_state = float(np.clip(obs.internal_state + d_state, min_state, max_state))

    is_move_action = action.get_delta() != (0, 0)
    collision = bool(is_move_action and (new_x, new_y) == obs.agent_position)

    return build_observation(
        agent_pos=(new_x, new_y), entities=list(entities), collision=collision,
        internal_state=new_internal_state, timestep=obs.timestep + 1, previous_action=int(action),
    )


def simulate_rollout_open_loop(
    model,
    initial_obs: Observation,
    actions: Sequence[Action],
    entities: Sequence[Entity],
    grid_width: int,
    grid_height: int,
    min_state: float,
    max_state: float,
) -> List[Observation]:
    """Simulate a fixed action sequence open-loop: each step's input is the MODEL'S OWN
    previous prediction, never the true environment. Returns [obs_0, obs_1, ..., obs_H].
    """
    trajectory = [initial_obs]
    obs = initial_obs
    for action in actions:
        obs = _step_observation(model, obs, action, entities, grid_width, grid_height, min_state, max_state)
        trajectory.append(obs)
    return trajectory


def plan_action(
    model,
    current_obs: Observation,
    entities: Sequence[Entity],
    grid_width: int,
    grid_height: int,
    min_state: float,
    max_state: float,
    horizon: int = 2,
) -> Action:
    """MPC-style lookahead: enumerate every action sequence of length `horizon`, simulate
    each open-loop with the world model, and return the first action of whichever sequence
    the model predicts accumulates the most internal_state. Called again (receding-horizon)
    after each real step, since the model's own drift makes longer open-loop commitment
    unreliable (see CSP-P4-E002).
    """
    action_space = list(Action)
    sequences = list(product(action_space, repeat=horizon))
    n = len(sequences)

    agent_x = np.full(n, current_obs.agent_position[0], dtype=np.float64)
    agent_y = np.full(n, current_obs.agent_position[1], dtype=np.float64)
    internal_state = np.full(n, current_obs.internal_state, dtype=np.float64)
    timestep = np.full(n, current_obs.timestep, dtype=np.int64)
    prev_action = np.full(n, int(current_obs.previous_action) if current_obs.previous_action is not None else -1, dtype=np.int64)
    collision = np.zeros(n, dtype=bool)
    cumulative_reward = np.zeros(n, dtype=np.float64)

    for depth in range(horizon):
        step_actions = [seq[depth] for seq in sequences]
        features = np.stack([
            encode_state_action(
                build_observation(
                    agent_pos=(int(agent_x[i]), int(agent_y[i])), entities=list(entities), collision=bool(collision[i]),
                    internal_state=float(internal_state[i]), timestep=int(timestep[i]),
                    previous_action=int(prev_action[i]) if prev_action[i] >= 0 else None,
                ),
                step_actions[i],
            )
            for i in range(n)
        ])
        preds = model.forward(features)  # (n, 3)
        dx, dy, d_state = preds[:, 0], preds[:, 1], preds[:, 2]

        new_x = np.clip(np.round(agent_x + dx), 0, grid_width - 1)
        new_y = np.clip(np.round(agent_y + dy), 0, grid_height - 1)
        is_move = np.array([a.get_delta() != (0, 0) for a in step_actions])
        collision = is_move & (new_x == agent_x) & (new_y == agent_y)

        agent_x, agent_y = new_x, new_y
        internal_state = np.clip(internal_state + d_state, min_state, max_state)
        cumulative_reward += d_state  # planning objective: maximize predicted internal-state gain
        prev_action = np.array([int(a) for a in step_actions])
        timestep = timestep + 1

    best_idx = int(np.argmax(cumulative_reward))
    return sequences[best_idx][0]
