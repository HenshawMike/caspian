"""Episode-sequence datasets for Project Caspian (Phase 2 — Memory & Persistence).

Phase 1's ExperienceDataset treats every transition as an independent, order-free
sample, which is correct for a stateless model. A recurrent model requires the
temporal order *within* an episode to be preserved (for BPTT) while still being
free to shuffle *across* episodes. EpisodeDataset provides that structure, plus
per-step diagnostic metadata (never fed to the model) used to test the "how long
is information retained?" question (Section 29).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from learning.dataset import Transition


@dataclass
class Episode:
    """A single, temporally-ordered sequence of transitions from one episode."""
    transitions: List[Transition]
    # Per-step diagnostic metadata (not part of the model input): whether the
    # single tracked entity was observable at that step, taken from the
    # environment's non-semantic `info["observable_entity_ids"]`.
    entity_observable: List[bool]

    def __len__(self) -> int:
        return len(self.transitions)

    def steps_since_last_observable(self) -> List[Optional[int]]:
        """For each step, how many steps have elapsed since the entity was last observable.

        0 means observable *this* step. None means the entity has never been
        observable yet in this episode (no memory available to draw on).
        """
        result: List[Optional[int]] = []
        last_seen: Optional[int] = None
        for t, visible in enumerate(self.entity_observable):
            if visible:
                last_seen = t
                result.append(0)
            else:
                result.append(None if last_seen is None else t - last_seen)
        return result

    def to_feature_sequence(self) -> np.ndarray:
        """Shape (T, input_dim) feature-vector sequence."""
        return np.stack([tr.to_feature_vector() for tr in self.transitions], axis=0)

    def to_target_sequence(self) -> np.ndarray:
        """Shape (T, output_dim) target sequence."""
        return np.stack([tr.to_target_vector() for tr in self.transitions], axis=0)


class EpisodeDataset:
    """In-memory collection of fixed-length episode sequences."""

    def __init__(self):
        self.episodes: List[Episode] = []

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def total_steps(self) -> int:
        return sum(len(ep) for ep in self.episodes)

    def to_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stack all episodes into batched arrays.

        Requires all episodes to share the same length T (guaranteed by the
        collector, which always runs a fixed number of steps per episode).

        Returns:
            Tuple[np.ndarray, np.ndarray]: X of shape (N, T, input_dim), Y of shape (N, T, output_dim).
        """
        if not self.episodes:
            return np.empty((0, 0, 0), dtype=np.float64), np.empty((0, 0, 0), dtype=np.float64)

        lengths = {len(ep) for ep in self.episodes}
        if len(lengths) != 1:
            raise ValueError(f"All episodes must share the same length for batched BPTT; got lengths {lengths}")

        X = np.stack([ep.to_feature_sequence() for ep in self.episodes], axis=0)
        Y = np.stack([ep.to_target_sequence() for ep in self.episodes], axis=0)
        return X, Y

    def split(self, train_ratio: float = 0.8, seed: Optional[int] = 42) -> Tuple["EpisodeDataset", "EpisodeDataset"]:
        """Split by *episode* (never by timestep, to avoid leaking future context)."""
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.episodes))
        rng.shuffle(indices)

        split_idx = int(len(self.episodes) * train_ratio)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        train_ds, val_ds = EpisodeDataset(), EpisodeDataset()
        train_ds.episodes = [self.episodes[i] for i in train_idx]
        val_ds.episodes = [self.episodes[i] for i in val_idx]
        return train_ds, val_ds

    def retention_breakdown(self) -> Dict[str, List[Any]]:
        """Flatten (steps_since_last_observable, squared_error-ready pair) across all episodes.

        Returns a dict of parallel lists suitable for grouping predictions by
        memory age: 'steps_since_observed' (Optional[int] per step, flattened)
        alongside 'episode_index' and 'timestep' so callers can align with
        model predictions computed via to_numpy_arrays().
        """
        steps_since: List[Optional[int]] = []
        episode_index: List[int] = []
        timestep: List[int] = []
        for ei, ep in enumerate(self.episodes):
            for t, s in enumerate(ep.steps_since_last_observable()):
                steps_since.append(s)
                episode_index.append(ei)
                timestep.append(t)
        return {
            "steps_since_observed": steps_since,
            "episode_index": episode_index,
            "timestep": timestep,
        }
