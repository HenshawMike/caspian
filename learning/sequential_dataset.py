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


def collect_balanced_sequential_trajectories(
    world_config: WorldConfig,
    num_episodes: int = 20,
    steps_per_episode: int = 30,
    interaction_prob: float = 0.25,
    min_interactions_per_episode: int = 2,
    max_resample_attempts: int = 20,
    seed: int = 42,
) -> Tuple[SequentialExperienceDataset, Dict[str, Any]]:
    """Collect exploratory sequential trajectories enforcing a minimum valid interaction count.

    Uses bounded deterministic resampling if an episode does not reach min_interactions_per_episode.
    Tracks and returns comprehensive collection statistics for distribution auditing.

    Args:
        world_config: World configuration including delay parameters.
        num_episodes: Number of independent episodes to collect.
        steps_per_episode: Number of transitions per episode.
        interaction_prob: Probability of sampling Action.INTERACT during random exploration.
        min_interactions_per_episode: Minimum valid interactions required per episode (default 2).
        max_resample_attempts: Maximum retry attempts per episode before accepting best effort (default 20).
        seed: Base seed for reproducible data generation.

    Returns:
        Tuple[SequentialExperienceDataset, Dict[str, Any]]:
            - Populated sequential dataset.
            - Audit dictionary with per-episode and aggregate distribution statistics.
    """
    dataset = SequentialExperienceDataset()
    movement_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.NOOP]
    episode_audit_records = []
    total_consequence_steps = 0
    total_interaction_events = 0
    episodes_interactions = []

    for ep in range(num_episodes):
        base_ep_seed = seed + ep * 1000 + 7
        best_episode = None
        best_interaction_count = -1
        best_attempt_record = None
        attempt_history = []
        accepted = False

        for attempt in range(max_resample_attempts):
            # Use a prime-stride to maximize seed diversity and avoid clustering
            attempt_seed = base_ep_seed if attempt == 0 else (base_ep_seed + attempt * 104729 + 37)
            attempt_rng = random.Random(attempt_seed)

            world = GridWorld(config=world_config)
            obs = world.reset(seed=attempt_seed)

            episode = TrajectoryEpisode(
                episode_id=ep,
                delay=world_config.default_interaction_delay,
                metadata={"seed": attempt_seed, "attempt": attempt},
            )

            ep_interactions = 0
            ep_consequences = 0

            for step_idx in range(steps_per_episode):
                if attempt_rng.random() < interaction_prob:
                    act = Action.INTERACT
                else:
                    act = attempt_rng.choice(movement_actions)

                next_obs, delta, done, info = world.step(act)

                if info.get("interaction_occurred", False):
                    ep_interactions += 1
                if delta > 0.0:
                    ep_consequences += 1

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

            attempt_record = {
                "episode_id": ep,
                "attempt_number": attempt + 1,
                "offset_seed": attempt_seed,
                "interaction_count": ep_interactions,
                "consequence_count": ep_consequences,
                "met_requirement": ep_interactions >= min_interactions_per_episode,
            }
            attempt_history.append(attempt_record)

            if ep_interactions > best_interaction_count:
                best_interaction_count = ep_interactions
                best_episode = episode
                best_attempt_record = attempt_record

            if ep_interactions >= min_interactions_per_episode:
                accepted = True
                break


        final_episode = best_episode
        episode_audit_records.append({
            "episode_id": ep,
            "base_seed": base_ep_seed,
            "final_seed": final_episode.metadata.get("seed"),
            "attempts_used": len(attempt_history),
            "accepted_target_met": accepted,
            "interaction_count": best_interaction_count,
            "consequence_count": best_attempt_record["consequence_count"],
            "attempt_history": attempt_history,
        })

        episodes_interactions.append(best_interaction_count)
        total_interaction_events += best_interaction_count
        total_consequence_steps += best_attempt_record["consequence_count"]
        dataset.add_episode(final_episode)

    total_timesteps = dataset.total_transitions
    audit_summary = {
        "seed": seed,
        "total_episodes": num_episodes,
        "total_timesteps": total_timesteps,
        "total_interaction_events": total_interaction_events,
        "interaction_rate": round(total_interaction_events / total_timesteps, 4) if total_timesteps > 0 else 0.0,
        "mean_interactions_per_episode": round(float(np.mean(episodes_interactions)), 2),
        "median_interactions_per_episode": round(float(np.median(episodes_interactions)), 2),
        "min_interactions_per_episode": int(np.min(episodes_interactions)) if episodes_interactions else 0,
        "max_interactions_per_episode": int(np.max(episodes_interactions)) if episodes_interactions else 0,
        "total_consequence_events": total_consequence_steps,
        "consequence_rate": round(total_consequence_steps / total_timesteps, 4) if total_timesteps > 0 else 0.0,
        "all_episodes_met_requirement": all(r["accepted_target_met"] for r in episode_audit_records),
        "episode_records": episode_audit_records,
    }

    return dataset, audit_summary
