"""Phase 1 Predictive Neural Agent for Project Caspian.

Implements a closed-loop agent that uses a trained PredictiveMLP to forecast the
consequences of candidate actions and maintain internal state representations.
"""

from typing import Optional, Dict, Any, List
import random
import numpy as np

from agent.base import BaseAgent
from environment.actions import Action
from environment.observations import Observation
from models.mlp import PredictiveMLP
from learning.dataset import encode_state_action


class Phase1PredictiveAgent(BaseAgent):
    """Predictive neural agent for Phase 1.

    Selects actions by forecasting environmental consequences using a learned
    predictive model, balanced with exploratory behavior.
    """

    def __init__(
        self,
        model: Optional[PredictiveMLP] = None,
        exploration_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.model = model if model is not None else PredictiveMLP(seed=seed)
        self.exploration_rate = float(exploration_rate)
        self._rng = random.Random(seed)
        self._step_count = 0
        self._last_observation: Optional[Observation] = None

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal RNG and episode step count."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self._step_count = 0
        self._last_observation = None

    def predict(self, observation: Observation, action: Action) -> float:
        """Predict the state variable delta (Delta E) for a hypothetical action.

        Args:
            observation: Current neutral Observation.
            action: Candidate Action.

        Returns:
            float: Predicted state delta.
        """
        feature_vec = encode_state_action(observation, action)
        pred = self.model.forward(feature_vec)
        return float(pred.item() if isinstance(pred, np.ndarray) else pred)

    def get_internal_state(self, observation: Observation, action: Action) -> np.ndarray:
        """Extract the latent internal state S_t for a state-action pair."""
        feature_vec = encode_state_action(observation, action)
        return self.model.get_latent_state(feature_vec)

    def evaluate_all_actions(self, observation: Observation) -> Dict[Action, float]:
        """Compute predicted state delta for all available discrete actions."""
        predictions = {}
        for act in Action:
            pred = self.predict(observation, act)
            predictions[act] = pred
        return predictions

    def act(self, observation: Observation) -> Action:
        """Select an action using epsilon-greedy predictive optimization.

        Args:
            observation: Current neutral Observation.

        Returns:
            Action: Selected Action enum member.
        """
        self._last_observation = observation
        self._step_count += 1

        all_actions = list(Action)

        # Exploration branch
        if self._rng.random() < self.exploration_rate:
            return self._rng.choice(all_actions)

        # Predictive greedy branch
        predictions = self.evaluate_all_actions(observation)
        max_val = max(predictions.values())
        best_actions = [a for a, val in predictions.items() if val == max_val]

        return self._rng.choice(best_actions)

    def observe(
        self,
        observation: Observation,
        state_delta: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Receive post-transition environment feedback."""
        self._last_observation = observation
