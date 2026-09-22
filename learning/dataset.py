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
