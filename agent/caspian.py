"""Unified Agent Interface for Project Caspian.

Provides the primary Caspian entry point capable of instantiating Phase 1 (Reactive/MLP)
or Phase 2 (Persistent Memory/GRU) agent architectures.
"""

from typing import Optional, Union, Dict, Any
from agent.base import BaseAgent
from agent.predictive_agent import Phase1PredictiveAgent
from agent.memory_agent import Phase2MemoryAgent
from models.mlp import PredictiveMLP
from models.recurrent import RecurrentPredictor


class Caspian(BaseAgent):
    """Unified Caspian Agent wrapper.

    Configurable between Phase 1 (No-Memory Feedforward) and Phase 2 (Recurrent Memory) modes.
    """

    def __init__(
        self,
        mode: str = "memory",  # "memory" (Phase 2) or "reactive" (Phase 1)
        model: Optional[Union[PredictiveMLP, RecurrentPredictor]] = None,
        exploration_rate: float = 0.1,
        hidden_dim: int = 32,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.mode = mode.lower()
        self.seed = seed

        if self.mode == "memory" or self.mode == "phase2":
            if model is not None and not isinstance(model, RecurrentPredictor):
                raise TypeError(f"Phase 2 memory mode expects RecurrentPredictor, got {type(model)}")
            self._underlying_agent = Phase2MemoryAgent(
                model=model,
                exploration_rate=exploration_rate,
                hidden_dim=hidden_dim,
                seed=seed,
            )
        elif self.mode == "reactive" or self.mode == "phase1" or self.mode == "no_memory":
            if model is not None and not isinstance(model, PredictiveMLP):
                raise TypeError(f"Phase 1 reactive mode expects PredictiveMLP, got {type(model)}")
            self._underlying_agent = Phase1PredictiveAgent(
                model=model,
                exploration_rate=exploration_rate,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown Caspian mode '{mode}'. Choose 'memory' or 'reactive'.")

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset underlying agent and internal memory."""
        self._underlying_agent.reset(seed=seed)

    def act(self, observation) -> Any:
        """Select action using underlying agent policy."""
        return self._underlying_agent.act(observation)

    def predict(self, observation, action) -> float:
        """Predict consequence using underlying agent model."""
        return self._underlying_agent.predict(observation, action)

    def observe(self, observation, state_delta: float, done: bool, info: Dict[str, Any]) -> None:
        """Forward environment feedback to underlying agent."""
        self._underlying_agent.observe(observation, state_delta, done, info)

    @property
    def model(self):
        """Access underlying predictive model."""
        return self._underlying_agent.model

    @property
    def memory(self):
        """Access underlying internal memory (if memory mode)."""
        if hasattr(self._underlying_agent, "memory"):
            return self._underlying_agent.memory
        return None
