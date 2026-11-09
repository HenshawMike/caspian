"""Phase 2 Recurrent (memory-capable) Predictive Agent for Project Caspian.

Structurally identical in role to Phase1PredictiveAgent (epsilon-greedy over
predicted state-deltas for each candidate action), but carries a persistent
hidden state h_t across timesteps within an episode, so it can retain
information about entities that are no longer observable (Section 29).
"""

from typing import Optional, Dict, Any
import random
import numpy as np

from agent.base import BaseAgent
from environment.actions import Action
from environment.observations import Observation
from models.rnn import RecurrentPredictiveModel
from learning.dataset import encode_state_action


class Phase2RecurrentAgent(BaseAgent):
    """Memory-capable predictive agent for Phase 2."""

    def __init__(
        self,
        model: Optional[RecurrentPredictiveModel] = None,
        exploration_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.model = model if model is not None else RecurrentPredictiveModel(seed=seed)
        self.exploration_rate = float(exploration_rate)
        self._rng = random.Random(seed)
        self._hidden_state: Optional[np.ndarray] = None  # persists across act() calls within an episode

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset RNG and clear the hidden state (start of a new episode)."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self._hidden_state = None

    def evaluate_all_actions(self, observation: Observation) -> Dict[Action, float]:
        """Predict Delta E for every candidate action, without committing the hidden state.

        Each candidate action is evaluated from the *current* hidden state (a
        hypothetical one-step lookahead); the hidden state is only actually
        advanced once, for the action that is chosen (see act()).
        """
        predictions = {}
        for act in Action:
            feature_vec = encode_state_action(observation, act)
            y_hat, _ = self.model.predict_step(feature_vec, self._hidden_state)
            predictions[act] = float(np.asarray(y_hat).item())
        return predictions

    def act(self, observation: Observation) -> Action:
        """Select an action using epsilon-greedy predictive optimization, then advance memory."""
        all_actions = list(Action)

        if self._rng.random() < self.exploration_rate:
            chosen = self._rng.choice(all_actions)
        else:
            predictions = self.evaluate_all_actions(observation)
            max_val = max(predictions.values())
            best_actions = [a for a, val in predictions.items() if val == max_val]
            chosen = self._rng.choice(best_actions)

        # Advance the persistent hidden state using the action actually taken.
        feature_vec = encode_state_action(observation, chosen)
        _, self._hidden_state = self.model.predict_step(feature_vec, self._hidden_state)

        return chosen

    def observe(self, observation: Observation, state_delta: float, done: bool, info: Dict[str, Any]) -> None:
        """Post-transition hook. Hidden state already advanced in act()."""
        pass
