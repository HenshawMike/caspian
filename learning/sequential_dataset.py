"""Sequential experience datasets and trajectory episode containers for Project Caspian (Phase 2).

Provides episode-level trajectory containers, 3D tensor sequence generation (Batch, Seq_Len, Features),
episode-based train/test splitting, and conversion to flattened transition datasets.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import random

from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from environment.observations import Observation
from learning.dataset import Transition, ExperienceDataset, encode_state_action


@dataclass
class TrajectoryEpisode:
    """A sequence of transitions comprising a single discrete episode."""
    episode_id: int
    transitions: List[Transition] = field(default_factory=list)
    delay: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.transitions)

    def to_sequence_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert episode transitions to input matrix X (T, D_in) and target matrix Y (T, 1)."""
        if not self.transitions:
            return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

        x_list = [t.to_feature_vector() for t in self.transitions]
        y_list = [t.to_target_vector() for t in self.transitions]

        X = np.stack(x_list, axis=0)  # (T, D_in)
        Y = np.stack(y_list, axis=0)  # (T, 1)
        return X, Y


class SequentialExperienceDataset:
    """Dataset containing trajectory episodes for recurrent / memory model training."""

    def __init__(self):
        self.episodes: List[TrajectoryEpisode] = []

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def total_transitions(self) -> int:
        """Total number of transitions across all stored episodes."""
        return sum(len(ep) for ep in self.episodes)

    def add_episode(self, episode: TrajectoryEpisode) -> None:
        """Add a trajectory episode to the dataset."""
        self.episodes.append(episode)

    def clear(self) -> None:
        """Clear all stored episodes."""
        self.episodes.clear()

    def to_numpy_sequences(self, pad_to_max: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Convert episodes into 3D batch arrays (Batch, Seq_Len, Features).

        Args:
            pad_to_max: If True, pads shorter episodes with zeros to match max length.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                X: shape (num_episodes, max_seq_len, input_dim)
                Y: shape (num_episodes, max_seq_len, 1)
        """
        if not self.episodes:
            return np.empty((0, 0, 0), dtype=np.float64), np.empty((0, 0, 0), dtype=np.float64)

        max_len = max(len(ep) for ep in self.episodes)
        first_feat = self.episodes[0].transitions[0].to_feature_vector()
        feat_dim = first_feat.shape[0]

        X_batch = np.zeros((len(self.episodes), max_len, feat_dim), dtype=np.float64)
        Y_batch = np.zeros((len(self.episodes), max_len, 1), dtype=np.float64)

        for i, ep in enumerate(self.episodes):
            X_ep, Y_ep = ep.to_sequence_arrays()
            ep_len = len(ep)
            if ep_len > 0:
                X_batch[i, :ep_len, :] = X_ep
                Y_batch[i, :ep_len, :] = Y_ep

        return X_batch, Y_batch

    def to_flat_dataset(self) -> ExperienceDataset:
        """Flatten all episode transitions into a standard ExperienceDataset."""
        flat_ds = ExperienceDataset()
        for ep in self.episodes:
            flat_ds.add_transitions(ep.transitions)
        return flat_ds

    def split(
        self,
        train_ratio: float = 0.8,
        seed: Optional[int] = 42,
    ) -> Tuple["SequentialExperienceDataset", "SequentialExperienceDataset"]:
        """Split dataset into training and validation subsets at the episode boundary.

        Ensures full trajectory isolation between train and validation sets.

        Args:
            train_ratio: Fraction of episodes allocated to train set.
            seed: Random seed for shuffling.

        Returns:
            Tuple[SequentialExperienceDataset, SequentialExperienceDataset]: (train_ds, val_ds)
        """
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.episodes))
        rng.shuffle(indices)

        split_idx = int(len(self.episodes) * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train_ds = SequentialExperienceDataset()
        val_ds = SequentialExperienceDataset()

        train_ds.episodes = [self.episodes[i] for i in train_idx]
        val_ds.episodes = [self.episodes[i] for i in val_idx]

        return train_ds, val_ds


def collect_sequential_trajectories(
    world_config: WorldConfig,
    num_episodes: int = 20,
    steps_per_episode: int = 30,
    interaction_prob: float = 0.25,
    seed: int = 42,
) -> SequentialExperienceDataset:
    """Collect exploratory sequential trajectories under specified world configuration.

    Args:
        world_config: World configuration including delay parameters.
        num_episodes: Number of independent episodes to collect.
        steps_per_episode: Number of transitions per episode.
        interaction_prob: Probability of executing INTERACT action.
        seed: Base seed for reproducible data generation.

    Returns:
        SequentialExperienceDataset: Populated sequential dataset.
    """
    dataset = SequentialExperienceDataset()
    rng = random.Random(seed)
    movement_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.NOOP]

    for ep in range(num_episodes):
        ep_seed = seed + ep * 1000 + 7
        ep_rng = random.Random(ep_seed)

        world = GridWorld(config=world_config)
        obs = world.reset(seed=ep_seed)

        episode = TrajectoryEpisode(
            episode_id=ep,
            delay=world_config.default_interaction_delay,
            metadata={"seed": ep_seed},
        )

        for step_idx in range(steps_per_episode):
            if ep_rng.random() < interaction_prob:
                act = Action.INTERACT
            else:
                act = ep_rng.choice(movement_actions)

            next_obs, delta, done, info = world.step(act)

            transition = Transition(
                obs=obs,
                action=act,
                state_delta=delta,
                next_obs=next_obs,
                done=done,
                info=info,
            )
            episode.transitions.append(transition)
            obs = next_obs

            if done:
                break

        dataset.add_episode(episode)

    return dataset
