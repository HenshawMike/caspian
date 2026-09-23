"""Internal memory abstractions and episode isolation for Project Caspian (Phase 2).

Maintains the persistent internal state vector S_t across sequential timesteps,
guaranteeing strict episode boundary isolation and zero cross-episode state contamination.
"""

from typing import Optional, Dict, Any, List
import numpy as np


class InternalMemory:
    """Persistent internal recurrent memory container.

    Attributes:
        hidden_dim: Dimensionality of the latent memory vector S_t.
        state: Current latent memory vector of shape (hidden_dim,).
        step_count: Number of updates performed in the current episode.
    """

    def __init__(self, hidden_dim: int = 32):
        self.hidden_dim = int(hidden_dim)
        self._state: np.ndarray = np.zeros((self.hidden_dim,), dtype=np.float64)
        self._step_count: int = 0
        self._history: List[np.ndarray] = []
        self._is_empty: bool = True

    @property
    def is_empty(self) -> bool:
        """Check if memory is at its clean initial zero state."""
        return self._is_empty

    @property
    def step_count(self) -> int:
        """Return number of sequential updates in active episode."""
        return self._step_count

    def reset(self) -> None:
        """Completely reset the internal memory state (Episode Isolation Guarantee).

        Clears the latent memory vector to zeros, empties history buffers,
        and sets step count to 0.
        """
        self._state = np.zeros((self.hidden_dim,), dtype=np.float64)
        self._step_count = 0
        self._history.clear()
        self._is_empty = True

    def update(self, new_state: np.ndarray) -> None:
        """Update the latent internal memory state with a new vector S_t.

        Args:
            new_state: Updated latent state array of shape (hidden_dim,) or (1, hidden_dim).
        """
        s = np.asarray(new_state, dtype=np.float64).reshape(-1)
        if s.size != self.hidden_dim:
            raise ValueError(f"State dimension mismatch: expected {self.hidden_dim}, got {s.size}")

        self._state = s.copy()
        self._history.append(s.copy())
        self._step_count += 1
        self._is_empty = False

    def get_state(self) -> np.ndarray:
        """Retrieve a copy of the current latent internal memory vector S_t."""
        return self._state.copy()

    def get_history(self) -> List[np.ndarray]:
        """Retrieve the sequence of all latent memory states recorded in this episode."""
        return [h.copy() for h in self._history]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory state to dictionary."""
        return {
            "hidden_dim": self.hidden_dim,
            "step_count": self._step_count,
            "is_empty": self._is_empty,
            "current_state": self._state.tolist(),
        }
