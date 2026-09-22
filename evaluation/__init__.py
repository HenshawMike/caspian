"""Evaluation and baseline components for Project Caspian."""

from evaluation.metrics import (
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_r2,
    compute_baseline_gap,
    compute_cosine_similarity,
    compute_all_metrics,
)
from evaluation.baselines import (
    BasePredictor,
    PersistencePredictor,
    RandomPredictor,
    ReactivePredictor,
)
from evaluation.evaluator import ModelEvaluator

__all__ = [
    "compute_mse",
    "compute_rmse",
    "compute_mae",
    "compute_r2",
    "compute_baseline_gap",
    "compute_cosine_similarity",
    "compute_all_metrics",
    "BasePredictor",
    "PersistencePredictor",
    "RandomPredictor",
    "ReactivePredictor",
    "ModelEvaluator",
]
