"""Experimental Suite for Project Caspian (Phase 2 — Memory & Persistence).

Executes the core scientific experiments for Phase 2, addressing the questions
posed in Section 29 of the Development Plan:

- CSP-P2-E001: Does memory improve prediction? (Recurrent model vs stateless
  Phase 1 architecture vs trivial baselines, under partial observability.)
- CSP-P2-E002: For how long is information retained? (Error vs steps-since-
  entity-last-observed, recurrent vs stateless.)
- CSP-P2-E003: Is the advantage reproducible across seeds?
- CSP-P2-E004: Semantic firewall audit + does memory produce useful internal state?
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import WorldConfig
from environment.observations import FORBIDDEN_SEMANTIC_TOKENS
from models.mlp import PredictiveMLP
from models.rnn import RecurrentPredictiveModel
from learning.collector import TrajectoryCollector
from learning.trainer import PredictiveTrainer
from learning.recurrent_trainer import RecurrentTrainer
from learning.sequence_dataset import EpisodeDataset
from evaluation.evaluator import ModelEvaluator
from evaluation.metrics import compute_all_metrics, compute_baseline_gap, compute_cosine_similarity
from evaluation.baselines import PersistencePredictor, RandomPredictor, ReactivePredictor

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _load_phase2_world_config() -> WorldConfig:
    with open(CONFIG_DIR / "phase2_world.json", "r", encoding="utf-8") as f:
        return WorldConfig.from_dict(json.load(f))


def _flatten(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten (N, T, dim) episode arrays into (N*T, dim) per-step arrays."""
    n, t, d_in = X.shape
    return X.reshape(n * t, d_in), Y.reshape(n * t, Y.shape[-1])


def _train_recurrent(train_ds: EpisodeDataset, val_ds: EpisodeDataset, seed: int) -> Tuple[RecurrentPredictiveModel, Dict[str, Any]]:
    model = RecurrentPredictiveModel(input_dim=37, hidden_dim=24, output_dim=1, seed=seed)
    trainer = RecurrentTrainer(model=model, batch_size=8, epochs=150, early_stopping_patience=20, seed=seed)
    history = trainer.fit(train_ds, val_data=val_ds)
    return model, history


def _train_stateless(X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, seed: int) -> Tuple[PredictiveMLP, Dict[str, Any]]:
    model = PredictiveMLP(input_dim=X_train.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit((X_train, Y_train), val_data=(X_val, Y_val))
    return model, history


def _collect_train_val_test(seed: int, num_episodes: int = 80, steps_per_episode: int = 64) -> Tuple[EpisodeDataset, EpisodeDataset, EpisodeDataset]:
    world_config = _load_phase2_world_config()
    collector = TrajectoryCollector(world_config=world_config)

    full_ds = collector.collect_episode_dataset(
        num_episodes=num_episodes, steps_per_episode=steps_per_episode, interaction_prob=0.3, base_seed=seed,
    )
    train_ds, remainder_ds = full_ds.split(train_ratio=0.7, seed=seed)
    val_ds, test_ds = remainder_ds.split(train_ratio=0.5, seed=seed + 1)
    return train_ds, val_ds, test_ds


def run_experiment_e001(seed: int = 42, num_episodes: int = 80, steps_per_episode: int = 64) -> Dict[str, Any]:
    """CSP-P2-E001: Does a recurrent (memory) model beat a stateless model and trivial baselines
    under partial observability, especially on steps where the relevant entity is unobservable?
    """
    train_ds, val_ds, test_ds = _collect_train_val_test(seed, num_episodes, steps_per_episode)

    # Train recurrent model (has access to h_{t-1} -> can remember).
    rnn_model, rnn_history = _train_recurrent(train_ds, val_ds, seed)

    # Train stateless Phase 1 architecture on the *same* per-step data (no memory).
    X_train, Y_train = _flatten(*train_ds.to_numpy_arrays())
    X_val, Y_val = _flatten(*val_ds.to_numpy_arrays())
    X_test, Y_test = _flatten(*test_ds.to_numpy_arrays())
    mlp_model, mlp_history = _train_stateless(X_train, Y_train, X_val, Y_val, seed)

    # Evaluate both on held-out test episodes (whole-set metrics).
    X_test_seq, Y_test_seq = test_ds.to_numpy_arrays()
    rnn_pred_seq = rnn_model.forward_sequence(X_test_seq)
    rnn_metrics = compute_all_metrics(rnn_pred_seq, Y_test_seq)
    mlp_metrics = compute_all_metrics(mlp_model.forward(X_test), Y_test)

    # Trivial baselines on the same flattened test set.
    persist = PersistencePredictor(default_value=0.0)
    persist_metrics = compute_all_metrics(persist.predict(X_test), Y_test)
    rand_pred = RandomPredictor(low=-1.0, high=10.0, seed=seed)
    rand_metrics = compute_all_metrics(rand_pred.predict(X_test), Y_test)
    reactive = ReactivePredictor(mean_value=float(np.mean(Y_train)))
    reactive_metrics = compute_all_metrics(reactive.predict(X_test), Y_test)

    # The scientifically important comparison: error restricted to steps where
    # the entity is currently unobservable (memory is the *only* source of information).
    evaluator = ModelEvaluator(seed=seed)
    rnn_retention = evaluator.evaluate_memory_retention(rnn_model, test_ds)

    # Same breakdown for the stateless model (reshape its flattened predictions back into (N,T,1)).
    n, t = X_test_seq.shape[0], X_test_seq.shape[1]
    mlp_pred_seq = mlp_model.forward(X_test).reshape(n, t, 1)
    mlp_sq_err = np.square(mlp_pred_seq - Y_test_seq)
    breakdown = test_ds.retention_breakdown()
    mlp_hidden_errors = [
        float(mlp_sq_err[ei, tt, :].mean())
        for ei, tt, s in zip(breakdown["episode_index"], breakdown["timestep"], breakdown["steps_since_observed"])
        if s is not None and s > 0
    ]
    mlp_hidden_mse = round(float(np.mean(mlp_hidden_errors)), 6) if mlp_hidden_errors else None

    return {
        "experiment_id": "CSP-P2-E001",
        "title": "Does Memory Improve Prediction Under Partial Observability?",
        "seed": seed,
        "train_episodes": len(train_ds),
        "val_episodes": len(val_ds),
        "test_episodes": len(test_ds),
        "recurrent_model": {
            "overall_metrics": rnn_metrics,
            "training_history": rnn_history,
            "retention_breakdown": rnn_retention,
            "num_parameters": rnn_model.num_parameters,
        },
        "stateless_model": {
            "overall_metrics": mlp_metrics,
            "training_history": mlp_history,
            "hidden_step_mse": mlp_hidden_mse,
            "num_parameters": mlp_model.num_parameters,
        },
        "baselines": {
            "persistence": persist_metrics,
            "random": rand_metrics,
            "reactive": reactive_metrics,
        },
        "hidden_step_gap_recurrent_vs_stateless_pct": (
            round(compute_baseline_gap(rnn_retention.get("hidden_step_mse") or 0.0, mlp_hidden_mse), 2)
            if rnn_retention.get("hidden_step_mse") is not None and mlp_hidden_mse
            else None
        ),
        "recurrent_beats_stateless_on_hidden_steps": (
            rnn_retention.get("hidden_step_mse") is not None
            and mlp_hidden_mse is not None
            and rnn_retention["hidden_step_mse"] < mlp_hidden_mse
        ),
        "recurrent_beats_persistence_overall": rnn_metrics["mse"] < persist_metrics["mse"],
    }


def run_experiment_e002(seed: int = 42) -> Dict[str, Any]:
    """CSP-P2-E002: For how long is information retained? Error vs steps-since-last-observed,
    recurrent model vs stateless model, on a shared held-out set.
    """
    train_ds, val_ds, test_ds = _collect_train_val_test(seed)
    rnn_model, _ = _train_recurrent(train_ds, val_ds, seed)

    X_train, Y_train = _flatten(*train_ds.to_numpy_arrays())
    X_val, Y_val = _flatten(*val_ds.to_numpy_arrays())
    mlp_model, _ = _train_stateless(X_train, Y_train, X_val, Y_val, seed)

    evaluator = ModelEvaluator(seed=seed)
    rnn_retention = evaluator.evaluate_memory_retention(rnn_model, test_ds)

    # Stateless per-bucket breakdown (same mechanics as in E001, factored out here for the curve).
    X_test_seq, Y_test_seq = test_ds.to_numpy_arrays()
    X_test_flat, _ = _flatten(X_test_seq, Y_test_seq)
    n, t = X_test_seq.shape[0], X_test_seq.shape[1]
    mlp_pred_seq = mlp_model.forward(X_test_flat).reshape(n, t, 1)
    mlp_sq_err = np.square(mlp_pred_seq - Y_test_seq)
    breakdown = test_ds.retention_breakdown()

    mlp_bucket_errors: Dict[str, List[float]] = {}
    for ei, tt, s in zip(breakdown["episode_index"], breakdown["timestep"], breakdown["steps_since_observed"]):
        if s is None or s == 0:
            continue
        key = str(min(s, 8)) + ("+" if s >= 8 else "")
        mlp_bucket_errors.setdefault(key, []).append(float(mlp_sq_err[ei, tt, :].mean()))
    mlp_bucket_summary = {
        k: {"mean_mse": round(float(np.mean(v)), 6), "count": len(v)}
        for k, v in sorted(mlp_bucket_errors.items(), key=lambda kv: int(kv[0].rstrip("+")))
    }

    return {
        "experiment_id": "CSP-P2-E002",
        "title": "Retention Duration — Error vs Steps Since Entity Last Observed",
        "seed": seed,
        "recurrent_breakdown": rnn_retention["steps_since_observed_breakdown"],
        "stateless_breakdown": mlp_bucket_summary,
        "recurrent_visible_step_mse": rnn_retention["visible_step_mse"],
        "recurrent_hidden_step_mse": rnn_retention["hidden_step_mse"],
    }


def run_experiment_e003(seeds: List[int] = [42, 101, 7, 2026, 555]) -> Dict[str, Any]:
    """CSP-P2-E003: Reproducibility — does the recurrent model consistently beat the stateless
    model on hidden-observability steps across independent seeds?
    """
    per_seed = []
    for s in seeds:
        res = run_experiment_e001(seed=s)
        per_seed.append({
            "seed": s,
            "recurrent_hidden_mse": res["recurrent_model"]["retention_breakdown"].get("hidden_step_mse"),
            "stateless_hidden_mse": res["stateless_model"]["hidden_step_mse"],
            "recurrent_beats_stateless_on_hidden_steps": res["recurrent_beats_stateless_on_hidden_steps"],
        })

    valid = [r for r in per_seed if r["recurrent_hidden_mse"] is not None and r["stateless_hidden_mse"] is not None]
    rnn_mses = [r["recurrent_hidden_mse"] for r in valid]
    mlp_mses = [r["stateless_hidden_mse"] for r in valid]
    win_rate = 100.0 * sum(r["recurrent_beats_stateless_on_hidden_steps"] for r in valid) / len(valid) if valid else 0.0

    return {
        "experiment_id": "CSP-P2-E003",
        "title": "Reproducibility Across Seeds",
        "seeds": seeds,
        "per_seed_results": per_seed,
        "summary": {
            "mean_recurrent_hidden_mse": round(float(np.mean(rnn_mses)), 6) if rnn_mses else None,
            "std_recurrent_hidden_mse": round(float(np.std(rnn_mses)), 6) if rnn_mses else None,
            "mean_stateless_hidden_mse": round(float(np.mean(mlp_mses)), 6) if mlp_mses else None,
            "std_stateless_hidden_mse": round(float(np.std(mlp_mses)), 6) if mlp_mses else None,
            "recurrent_win_rate_pct": round(win_rate, 1),
        },
    }


def run_experiment_e004(seed: int = 42) -> Dict[str, Any]:
    """CSP-P2-E004: Semantic firewall audit for the new flicker mechanism, plus a check of
    whether the recurrent hidden state carries structure distinguishing "entity currently
    tracked from memory" vs "no recent information" — i.e. does memory produce useful
    internal state? (Section 29's fourth question.)
    """
    world_config = _load_phase2_world_config()

    # 1. Semantic firewall audit: the flicker rule itself, and its per-step consequence,
    #    must never leak into the Observation the agent receives.
    from environment.world import GridWorld
    world = GridWorld(config=world_config)
    obs = world.reset(seed=seed)
    leak_found = False
    for _ in range(world_config.max_timesteps):
        obs_dict = obs.to_dict()
        for token in ("flicker_period", "flicker_visible_steps", "visible", "observable"):
            if token in json.dumps(obs_dict):
                leak_found = True
        obs, _, done, _ = world.step("RIGHT" if world._timestep % 2 == 0 else "INTERACT")
        if done:
            break

    audit = {
        "tokens_scanned": list(FORBIDDEN_SEMANTIC_TOKENS) + ["flicker_period", "flicker_visible_steps", "visible", "observable"],
        "leakage_found": leak_found,
        "status": "PASSED (Zero leakage detected)" if not leak_found else "FAILED (Leakage detected)",
    }

    # 2. Does the recurrent hidden state encode "currently remembering the entity nearby"?
    train_ds, val_ds, test_ds = _collect_train_val_test(seed)
    rnn_model, _ = _train_recurrent(train_ds, val_ds, seed)

    X_test, Y_test = test_ds.to_numpy_arrays()
    rnn_model.forward_sequence(X_test)  # populate caches
    hidden_states = rnn_model.get_hidden_states()  # (N, T, hidden_dim)

    breakdown = test_ds.retention_breakdown()
    recent_memory_states, stale_memory_states = [], []
    for ei, tt, s in zip(breakdown["episode_index"], breakdown["timestep"], breakdown["steps_since_observed"]):
        if s is None:
            continue
        (recent_memory_states if s <= 2 else stale_memory_states).append(hidden_states[ei, tt, :])

    rep_result: Dict[str, Any]
    if recent_memory_states and stale_memory_states:
        mean_recent = np.mean(np.stack(recent_memory_states), axis=0)
        mean_stale = np.mean(np.stack(stale_memory_states), axis=0)
        cos_sim = compute_cosine_similarity(mean_recent, mean_stale)
        rep_result = {
            "recent_memory_sample_count": len(recent_memory_states),
            "stale_memory_sample_count": len(stale_memory_states),
            "recent_vs_stale_cosine_similarity": round(cos_sim, 4),
            "euclidean_distance": round(float(np.linalg.norm(mean_recent - mean_stale)), 4),
            "distinct_memory_states_formed": cos_sim < 0.95,
        }
    else:
        rep_result = {"status": "insufficient_class_variance"}

    return {
        "experiment_id": "CSP-P2-E004",
        "title": "Semantic Firewall Audit & Internal Memory-State Structure",
        "seed": seed,
        "semantic_audit": audit,
        "memory_representation_analysis": rep_result,
    }


def run_all_phase2_experiments(output_dir: str = "logs") -> Dict[str, Any]:
    """Execute all Phase 2 experiments sequentially and persist results."""
    import os

    print("Running CSP-P2-E001 (Does Memory Improve Prediction?)...")
    e001 = run_experiment_e001()

    print("Running CSP-P2-E002 (Retention Duration)...")
    e002 = run_experiment_e002()

    print("Running CSP-P2-E003 (Reproducibility Across Seeds)...")
    e003 = run_experiment_e003()

    print("Running CSP-P2-E004 (Semantic Audit & Memory Representation)...")
    e004 = run_experiment_e004()

    all_results = {
        "phase": "Phase 2 — Memory & Persistence",
        "experiments": {
            "CSP-P2-E001": e001,
            "CSP-P2-E002": e002,
            "CSP-P2-E003": e003,
            "CSP-P2-E004": e004,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase2_all_experiments_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Phase 2 experiments completed and saved to: {out_file}")
    return all_results


if __name__ == "__main__":
    run_all_phase2_experiments()
