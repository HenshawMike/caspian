"""Experience datasets and transition containers for Project Caspian.

Provides neutral vectorized experience storage, batch sampling, and train/val splitting.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from environment.actions import Action
from environment.observations import Observation


def one_hot_encode_action(action: Union[Action, int], num_actions: int = 6) -> np.ndarray:
    """Encode an action code as a one-hot indicator vector.

    Args:
        action: Action enum or integer index.
        num_actions: Cardinality of the discrete action space (default 6).

    Returns:
        np.ndarray: Vector of shape (num_actions,) with 1.0 at action index.
    """
    idx = int(action)
    vec = np.zeros(num_actions, dtype=np.float64)
    if 0 <= idx < num_actions:
        vec[idx] = 1.0
    return vec


def encode_state_action(
    obs: Union[Observation, List[float], np.ndarray],
    action: Union[Action, int],
    max_entities: int = 5,
    num_actions: int = 6,
) -> np.ndarray:
    """Concatenate observation vector and one-hot action vector.

    Args:
        obs: Neutral Observation object or numeric vector.
        action: Action executed.
        max_entities: Max entity slots in vector representation.
        num_actions: Number of discrete action dimensions.

    Returns:
        np.ndarray: Unified state-action feature vector.
    """
    if isinstance(obs, Observation):
        obs_vec = np.array(obs.to_vector(max_entities=max_entities), dtype=np.float64)
    else:
        obs_vec = np.asarray(obs, dtype=np.float64)

    act_vec = one_hot_encode_action(action, num_actions=num_actions)
    return np.concatenate([obs_vec, act_vec])


def encode_state_action_deepsets(
    obs: Observation,
    action: Union[Action, int],
    max_entities: int = 5,
    num_actions: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structured, permutation-friendly encoding for DeepSetsPredictiveModel.

    Unlike encode_state_action() (which flattens entities into fixed input
    slots), this keeps entities as a set: a per-slot feature array plus an
    explicit presence mask, so a permutation-invariant model can pool over
    only the entities actually present, in any order. entity_id is
    deliberately excluded from entity_features (an arbitrary instance
    identifier is not something a reusable rule should condition on).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            entity_features: (max_entities, 4) -- [entity_type, rel_x, rel_y, distance] per slot.
            presence_mask: (max_entities,) -- 1.0 where a real entity occupies that slot.
            global_features: (6 + num_actions,) -- agent/env state concatenated with one-hot action.
    """
    if not isinstance(obs, Observation):
        raise TypeError("encode_state_action_deepsets requires an Observation instance")

    entity_features = np.zeros((max_entities, 4), dtype=np.float64)
    presence_mask = np.zeros((max_entities,), dtype=np.float64)
    for i, e in enumerate(obs.entities[:max_entities]):
        entity_features[i] = [float(e.entity_type), float(e.relative_x), float(e.relative_y), float(e.distance)]
        presence_mask[i] = 1.0

    global_state = np.array([
        float(obs.agent_position[0]),
        float(obs.agent_position[1]),
        1.0 if obs.collision else 0.0,
        float(obs.internal_state),
        float(obs.timestep),
        float(obs.previous_action if obs.previous_action is not None else -1.0),
    ], dtype=np.float64)
    act_vec = one_hot_encode_action(action, num_actions=num_actions)
    global_features = np.concatenate([global_state, act_vec])

    return entity_features, presence_mask, global_features


def encode_world_model_target(obs: Observation, next_obs: Observation) -> np.ndarray:
    """Full-state transition target for a Phase 4 world model: [dx, dy, d_internal_state].

    Phase 1-3 predict only the scalar internal-state delta. A world model that will be
    used for multi-step rollout (feeding its own predictions back in as the next input)
    needs to predict the *whole* next state, not just one scalar -- otherwise there is
    nothing to reconstruct a next Observation from. Entity positions are not part of this
    target: entities are static in this environment (verified, not assumed -- see
    CSP-P4-E001), so a rollout can carry them forward unchanged rather than predict a
    quantity that is always exactly zero.

    Returns:
        np.ndarray: [dx, dy, d_internal_state], shape (3,).
    """
    dx = float(next_obs.agent_position[0] - obs.agent_position[0])
    dy = float(next_obs.agent_position[1] - obs.agent_position[1])
    d_state = float(next_obs.internal_state - obs.internal_state)
    return np.array([dx, dy, d_state], dtype=np.float64)


@dataclass
class Transition:
    """Single discrete step transition record."""
    obs: Observation
    action: Action
    state_delta: float
    next_obs: Observation
    done: bool
    info: Dict[str, Any]
    max_entities: int = 5
    num_actions: int = 6

    def to_feature_vector(self) -> np.ndarray:
        """Construct the state-action input vector for this transition."""
        return encode_state_action(
            self.obs,
            self.action,
            max_entities=self.max_entities,
            num_actions=self.num_actions,
        )

    def to_target_vector(self) -> np.ndarray:
        """Return the target state delta [Delta E]."""
        return np.array([self.state_delta], dtype=np.float64)

    def to_deepsets_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Structured (entity_features, presence_mask, global_features, target) for DeepSetsPredictiveModel."""
        entity_features, presence_mask, global_features = encode_state_action_deepsets(
            self.obs, self.action, max_entities=self.max_entities, num_actions=self.num_actions,
        )
        return entity_features, presence_mask, global_features, self.to_target_vector()

    def to_world_model_target(self) -> np.ndarray:
        """Full-state target [dx, dy, d_internal_state] for a Phase 4 world model."""
        return encode_world_model_target(self.obs, self.next_obs)


class ExperienceDataset:
    """In-memory dataset of environmental transitions for neural training."""

    def __init__(self):
        self.transitions: List[Transition] = []

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: Transition) -> None:
        """Add a transition to the dataset."""
        self.transitions.append(transition)

    def add_transitions(self, transitions: List[Transition]) -> None:
        """Add multiple transitions to the dataset."""
        self.transitions.extend(transitions)

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.transitions.clear()

    def to_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert all stored transitions into feature matrix X and target matrix Y.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, Y) arrays.
                X: shape (N, input_dim)
                Y: shape (N, 1)
        """
        if not self.transitions:
            return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

        x_list = [t.to_feature_vector() for t in self.transitions]
        y_list = [t.to_target_vector() for t in self.transitions]

        X = np.stack(x_list, axis=0)
        Y = np.stack(y_list, axis=0)
        return X, Y

    def to_world_model_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Batched (X, Y) for a Phase 4 world model: same 37-dim input as to_numpy_arrays(),
        but Y is [dx, dy, d_internal_state] (shape (N, 3)) instead of [d_internal_state] alone.
        """
        if not self.transitions:
            return np.empty((0, 0)), np.empty((0, 0))
        x_list = [t.to_feature_vector() for t in self.transitions]
        y_list = [t.to_world_model_target() for t in self.transitions]
        return np.stack(x_list, axis=0), np.stack(y_list, axis=0)

    def to_deepsets_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batched structured arrays for DeepSetsPredictiveModel.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                entity_features (N, max_entities, 4), presence_mask (N, max_entities),
                global_features (N, G), Y (N, 1).
        """
        if not self.transitions:
            return (
                np.empty((0, 0, 4), dtype=np.float64), np.empty((0, 0), dtype=np.float64),
                np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64),
            )
        entity_list, mask_list, global_list, y_list = [], [], [], []
        for t in self.transitions:
            ef, pm, gf, y = t.to_deepsets_arrays()
            entity_list.append(ef); mask_list.append(pm); global_list.append(gf); y_list.append(y)

        return (
            np.stack(entity_list, axis=0), np.stack(mask_list, axis=0),
            np.stack(global_list, axis=0), np.stack(y_list, axis=0),
        )

    def split(
        self,
        train_ratio: float = 0.8,
        seed: Optional[int] = 42,
    ) -> Tuple["ExperienceDataset", "ExperienceDataset"]:
        """Split dataset into training and validation subsets.

        Args:
            train_ratio: Fraction of transitions assigned to train dataset.
            seed: Random seed for reproducible shuffling.

        Returns:
            Tuple[ExperienceDataset, ExperienceDataset]: (train_dataset, val_dataset)
        """
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.transitions))
        rng.shuffle(indices)

        split_idx = int(len(self.transitions) * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train_ds = ExperienceDataset()
        val_ds = ExperienceDataset()

        train_ds.transitions = [self.transitions[i] for i in train_idx]
        val_ds.transitions = [self.transitions[i] for i in val_idx]

        return train_ds, val_ds
