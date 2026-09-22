"""Evaluation metrics and statistical measures for Project Caspian (Phase 1).

Implements MSE, RMSE, MAE, R^2, Baseline Gap, and internal representation metrics.
"""

from typing import Dict, Any, Union
import numpy as np


def compute_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error: (1/N) * sum((y_pred - y_true)^2)."""
    p = np.asarray(y_pred, dtype=np.float64)
    t = np.asarray(y_true, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(np.mean(np.square(p - t)))


def compute_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(compute_mse(y_pred, y_true)))


def compute_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error: (1/N) * sum(|y_pred - y_true|)."""
    p = np.asarray(y_pred, dtype=np.float64)
    t = np.asarray(y_true, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(np.mean(np.abs(p - t)))


def compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of Determination R^2 = 1 - SS_res / SS_tot."""
    p = np.asarray(y_pred, dtype=np.float64).flatten()
    t = np.asarray(y_true, dtype=np.float64).flatten()
    if t.size < 2:
        return 0.0
    ss_tot = np.sum(np.square(t - np.mean(t)))
    if ss_tot == 0.0:
        return 1.0 if np.allclose(p, t) else 0.0
    ss_res = np.sum(np.square(t - p))
    return float(1.0 - (ss_res / ss_tot))


def compute_baseline_gap(model_mse: float, baseline_mse: float) -> float:
    """Compute relative percentage error reduction over baseline: (MSE_base - MSE_model) / MSE_base * 100."""
    if baseline_mse == 0.0:
        return 0.0
    return float((baseline_mse - model_mse) / baseline_mse * 100.0)


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two representation vectors."""
    v1 = np.asarray(vec1, dtype=np.float64).flatten()
    v2 = np.asarray(vec2, dtype=np.float64).flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_all_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute a consolidated dictionary of standard regression metrics."""
    return {
        "mse": round(compute_mse(y_pred, y_true), 6),
        "rmse": round(compute_rmse(y_pred, y_true), 6),
        "mae": round(compute_mae(y_pred, y_true), 6),
        "r2_score": round(compute_r2(y_pred, y_true), 6),
    }
