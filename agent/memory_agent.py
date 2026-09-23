"""Phase 2 Memory-augmented Predictive Neural Agent for Project Caspian.

Integrates a RecurrentPredictor (GRU) with InternalMemory to perform sequential
decision-making and temporal consequence forecasting across delayed dependencies.
"""

from typing import Optional, Dict, Any, List
import random
import numpy as np

from agent.base import BaseAgent
from agent.memory import InternalMemory
from environment.actions import Action
from environment.observations import Observation
from models.recurrent import RecurrentPredictor
from learning.dataset import encode_state_action


class Phase2MemoryAgent(BaseAgent):
    """Memory-augmented predictive neural agent (Model B — Memory Model).

    Maintains a persistent recurrent latent state S_t across timesteps, enabling
    the retention of historical environmental cues across arbitrary temporal delays.
    """

    def __init__(
        self,
        model: Optional[RecurrentPredictor] = None,
        exploration_rate: float = 0.1,
        hidden_dim: int = 32,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.model = model if model is not None else RecurrentPredictor(hidden_dim=hidden_dim, seed=seed)
        self.memory = InternalMemory(hidden_dim=self.model.hidden_dim)
        self.exploration_rate = float(exploration_rate)
        self._rng = random.Random(seed)
        self._step_count = 0
        self._last_observation: Optional[Observation] = None
        self._last_action: Optional[Action] = None

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset agent, internal recurrent state, and memory (Episode Isolation).

        Args:
            seed: Random seed for action selection.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self._step_count = 0
        self._last_observation = None
        self._last_action = None

        # Reset neural model online state and internal memory container
        self.model.reset_state()
        self.memory.reset()

    def predict(self, observation: Observation, action: Action) -> float:
        """Predict the state variable delta (Delta E) for candidate action conditioning on history.

        Evaluates hypothetical step without permanently mutating online memory state.

        Args:
            observation: Current neutral Observation.
            action: Candidate Action.

        Returns:
            float: Predicted state delta.
        """
        feature_vec = encode_state_action(observation, action)
        current_h = self.memory.get_state().reshape(1, -1)
        x_2d = feature_vec.reshape(1, -1)

        # Hypothetical GRU step
        h_candidate, _ = self.model.gru.step(x_2d, current_h)
        y_hat = self.model.head.forward(h_candidate)
        return float(y_hat.item() if isinstance(y_hat, np.ndarray) else y_hat)

    def evaluate_all_actions(self, observation: Observation) -> Dict[Action, float]:
        """Forecast predicted state delta for all candidate discrete actions."""
        predictions = {}
        for act in Action:
            pred = self.predict(observation, act)
            predictions[act] = pred
        return predictions

    def act(self, observation: Observation) -> Action:
        """Select action via epsilon-greedy policy informed by recurrent memory.

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
            selected_action = self._rng.choice(all_actions)
        else:
            # Predictive greedy branch
            predictions = self.evaluate_all_actions(observation)
            max_val = max(predictions.values())
            best_actions = [a for a, val in predictions.items() if val == max_val]
            selected_action = self._rng.choice(best_actions)

        self._last_action = selected_action
        return selected_action

    def observe(
        self,
        observation: Observation,
        state_delta: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Receive post-transition environment feedback and advance internal memory state S_t.

        Args:
            observation: New Observation resulting from transition.
            state_delta: Measured change in state variable (Delta E).
            done: Episode termination flag.
            info: Transition diagnostic info.
        """
        if self._last_observation is not None and self._last_action is not None:
            # Advance online GRU state with executed transition
            feature_vec = encode_state_action(self._last_observation, self._last_action)
            _ = self.model.step(feature_vec)
            # Synchronize InternalMemory with model's updated hidden state
            self.memory.update(self.model.get_hidden_state())

        self._last_observation = observation
        if done:
            # Clear internal state upon episode termination
            self.model.reset_state()
            self.memory.reset()
