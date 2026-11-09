"""Streaming/online adaptation trainer for Project Caspian (Phase 6 — Rule Change &
Adaptation).

Every trainer elsewhere in this project (PredictiveTrainer, RecurrentTrainer,
DeepSetsTrainer) is a full-batch, offline fit: collect a dataset, then train to
convergence. Phase 6 asks a different question -- can Caspian notice, DURING a
continuous stream of interaction, that its existing model has become wrong, and
correct itself -- which requires a trainer that consumes one transition at a time
and updates incrementally, rather than one that is handed a finished dataset.

Design, deliberately the simplest thing that could work (earn complexity, Section
41): a bounded sliding-window replay buffer plus ordinary Adam gradient steps on
that buffer after every new sample, reusing the exact same forward/backward/
zero_grad/optimizer-step primitives every other trainer in this project already
uses (models/mlp.py, models/optimizers.py). No exotic continual-learning machinery
(no EWC, no replay prioritization, no separate fast/slow weights) -- if something
like that turns out to be needed, CSP-P6's experiments are what would supply the
evidence for earning it, not an assumption made up front.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from models.mlp import PredictiveMLP
from models.optimizers import Optimizer, Adam, MSELoss, LossFunction


class OnlineAdaptationTrainer:
    """Processes a stream of (x, y) transitions one at a time, performing a bounded
    sliding-window gradient update after each. Reports the PRE-update prediction error
    for every sample -- i.e. how surprised the model's current weights are by this
    observation, BEFORE learning from it -- which is the quantity Phase 6's detection/
    adaptation-time measurements are built on. Evaluating post-update error instead would
    leak the very sample being measured into the metric.
    """

    def __init__(
        self,
        model: PredictiveMLP,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[LossFunction] = None,
        buffer_size: int = 32,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.buffer_size = int(buffer_size)
        self.optimizer = optimizer if optimizer is not None else Adam(
            params=self.model.get_params(), lr=lr, weight_decay=weight_decay,
        )
        self._buffer_x: List[np.ndarray] = []
        self._buffer_y: List[np.ndarray] = []

    def observe(self, x: np.ndarray, y: np.ndarray) -> float:
        """Record one transition, return its pre-update squared prediction error, and
        perform one sliding-window gradient update including this sample.
        """
        pred = self.model.forward(x.reshape(1, -1))
        error = float(np.mean((pred.flatten() - y.flatten()) ** 2))

        self._buffer_x.append(x)
        self._buffer_y.append(y)
        if len(self._buffer_x) > self.buffer_size:
            self._buffer_x.pop(0)
            self._buffer_y.pop(0)

        X_b = np.stack(self._buffer_x, axis=0)
        Y_b = np.stack(self._buffer_y, axis=0)

        self.model.zero_grad()
        y_pred = self.model.forward(X_b)
        grad_out = self.loss_fn.gradient(y_pred, Y_b)
        self.model.backward(grad_out)
        self.optimizer.step()

        return error


def detect_and_measure_adaptation(
    records: List[Dict[str, Any]],
    pre_change_key: str = "pre",
    regime_field: str = "regime",
    error_field: str = "squared_error",
    detect_sigma: float = 3.0,
    recover_sigma: float = 1.0,
    sustain: int = 3,
    rolling_window: int = 5,
) -> Dict[str, Any]:
    """Shared detection/adaptation-time logic for Phase 6 experiments.

    `records` must already be filtered to the comparable event stream (e.g. only
    interaction events with the entity whose rule may have changed), in chronological
    order, each a dict with at least `regime_field` ("pre"/"post") and `error_field`.

    Detection time: index (within the post-change subsequence) of the first event
    whose rolling-window mean error exceeds `mean_pre + detect_sigma * std_pre`,
    sustained for `sustain` consecutive events (to reject single-sample noise).

    Adaptation time: index (within the post-change subsequence, counted from the
    change point, not from detection) of the first event AFTER detection whose
    rolling-window mean error drops back below `mean_pre + recover_sigma * std_pre`,
    sustained for `sustain` consecutive events.
    """
    pre = [r[error_field] for r in records if r[regime_field] == pre_change_key]
    post = [r[error_field] for r in records if r[regime_field] != pre_change_key]

    if len(pre) < 2 or len(post) < rolling_window:
        return {
            "insufficient_data": True,
            "num_pre_events": len(pre),
            "num_post_events": len(post),
        }

    mean_pre = float(np.mean(pre))
    std_pre = float(np.std(pre)) + 1e-8
    detect_threshold = mean_pre + detect_sigma * std_pre
    recover_threshold = mean_pre + recover_sigma * std_pre

    rolling = [
        float(np.mean(post[max(0, i - rolling_window + 1): i + 1]))
        for i in range(len(post))
    ]

    detection_idx = None
    for i in range(len(rolling) - sustain + 1):
        if all(rolling[i: i + sustain][j] > detect_threshold for j in range(sustain)):
            detection_idx = i
            break

    adaptation_idx = None
    if detection_idx is not None:
        for i in range(detection_idx, len(rolling) - sustain + 1):
            if all(rolling[i: i + sustain][j] < recover_threshold for j in range(sustain)):
                adaptation_idx = i
                break

    return {
        "insufficient_data": False,
        "num_pre_events": len(pre),
        "num_post_events": len(post),
        "mean_pre_error": round(mean_pre, 4),
        "std_pre_error": round(std_pre, 4),
        "detect_threshold": round(detect_threshold, 4),
        "recover_threshold": round(recover_threshold, 4),
        "detection_event_index": detection_idx,
        "adaptation_event_index": adaptation_idx,
        "events_to_detect": detection_idx,
        "events_from_detection_to_adapt": (
            (adaptation_idx - detection_idx) if (detection_idx is not None and adaptation_idx is not None) else None
        ),
        "events_to_fully_adapt": adaptation_idx,
        "peak_post_change_error": round(float(np.max(post)), 4),
        "final_rolling_error": round(rolling[-1], 4),
        "recovered_to_baseline_by_end": bool(rolling[-1] < recover_threshold),
    }
