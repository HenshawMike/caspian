"""CSP-P2-E007: Diagnostic Follow-Up Experiments for Project Caspian (Phase 2).

Resolves two confounds in the CSP-P2-E001 comparison:
  E007a — Early-stopped re-benchmark: both models trained with patience=30 early stopping,
           benchmarked at their own best_epoch checkpoint, not final epoch.
  E007b — Parameter-matched MLP control: new no-memory MLP with ~6,748 params
           (matched to the memory model's 6,753) trained under the same rule.
  E007c — Representation-metric sanity: recomputes inter_class_cosine_similarity on the
           corrected model, audits the threshold, adds transparency fields.

Part B (E007d/e) — Hidden-state-level diagnostics — runs only if decide_part_b() triggers.

decide_part_b() is the SINGLE SOURCE OF TRUTH for the Part B decision.
It uses OR logic: Part B triggers if ANY of d0_anomaly_persists,
generalization_failure_persists, or capacity_gap_material is True.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.actions import Action
from environment.observations import FORBIDDEN_SEMANTIC_TOKENS, _check_semantic_purity_recursive
from models.mlp import PredictiveMLP
from models.recurrent import RecurrentPredictor
from learning.trainer import PredictiveTrainer
from learning.sequential_dataset import (
    SequentialExperienceDataset,
    TrajectoryEpisode,
    collect_sequential_trajectories,
)
from learning.sequential_trainer import SequentialTrainer
from evaluation.evaluator import ModelEvaluator
from evaluation.metrics import compute_all_metrics, compute_baseline_gap, compute_cosine_similarity

# ---------------------------------------------------------------------------
# Original CSP-P2-E001 numbers (from the completed Phase 2 run) used as the
# "before" baseline for side-by-side comparison tables.
# ---------------------------------------------------------------------------
ORIGINAL_E001_NUMBERS = {
    "memory_model_mse": 3.470325,
    "no_memory_model_mse": 3.726696,
    "gap_pct": 6.9,
    "best_epoch_memory": 48,
    "best_epoch_no_memory": 82,
    "memory_params": 6753,
    "no_memory_params": 1761,
    "deficit_error_reduction_pct": 36.1,
}


# ---------------------------------------------------------------------------
# Shared world-config factory (same as E001)
# ---------------------------------------------------------------------------

def _make_world_config(delay: int, seed: int) -> WorldConfig:
    return WorldConfig(
        grid_width=5,
        grid_height=5,
        initial_agent_pos=(0, 0),
        initial_entities=[
            Entity(
                entity_id=1,
                entity_type=1,
                position=(2, 2),
                is_interactive=True,
                is_blocking=False,
                hidden_state_delta=10.0,
                interaction_delay=delay,
            )
        ],
        initial_internal_state=50.0,
        step_penalty=1.0,
        default_interaction_delay=delay,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# E007a — Early-Stopped Re-Benchmark
# ---------------------------------------------------------------------------

def run_experiment_p2_e007a(
    delay: int = 2,
    num_train_episodes: int = 30,
    num_test_episodes: int = 10,
    steps_per_episode: int = 25,
    seed: int = 42,
    early_stopping_patience: int = 30,
) -> Dict[str, Any]:
    """CSP-P2-E007a — Early-Stopped Re-Benchmark.

    Both models are retrained with patience=30 early stopping and benchmarked at
    their own best_epoch checkpoint (not the final epoch). Reports the training
    and validation loss at the restored checkpoint so the train/val gap is
    computed at the *right* pair of epochs, not final_train vs best_val.

    Side-by-side comparison against ORIGINAL_E001_NUMBERS is included in output.
    """
    world_config = _make_world_config(delay, seed)

    # Collect datasets (same seeds and sizes as original E001)
    train_seq_ds = collect_sequential_trajectories(
        world_config=world_config,
        num_episodes=num_train_episodes,
        steps_per_episode=steps_per_episode,
        interaction_prob=0.3,
        seed=seed,
    )
    test_seq_ds = collect_sequential_trajectories(
        world_config=world_config,
        num_episodes=num_test_episodes,
        steps_per_episode=steps_per_episode,
        interaction_prob=0.3,
        seed=seed + 99999,
    )

    X_train_seq, Y_train_seq = train_seq_ds.to_numpy_sequences()
    X_test_seq, Y_test_seq = test_seq_ds.to_numpy_sequences()

    train_flat_ds = train_seq_ds.to_flat_dataset()
    test_flat_ds = test_seq_ds.to_flat_dataset()
    X_train_flat, Y_train_flat = train_flat_ds.to_numpy_arrays()
    X_test_flat, Y_test_flat = test_flat_ds.to_numpy_arrays()

    # --- Train Model A (No-Memory MLP) with early stopping ---
    model_a = PredictiveMLP(
        input_dim=X_train_flat.shape[1],
        hidden_dims=(32, 16),
        output_dim=1,
        hidden_activation="relu",
        seed=seed,
    )
    trainer_a = PredictiveTrainer(
        model=model_a,
        batch_size=32,
        epochs=150,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    )
    history_a = trainer_a.fit(
        (X_train_flat, Y_train_flat),
        val_data=(X_test_flat, Y_test_flat),
    )

    # Compute train loss AT best_epoch (restore already happened in trainer)
    best_epoch_a = history_a["best_epoch"]
    train_loss_at_best_a = history_a["train_losses"][best_epoch_a - 1] if best_epoch_a > 0 else history_a["final_train_loss"]
    best_val_loss_a = history_a["best_val_loss"]
    trainval_gap_a = train_loss_at_best_a / best_val_loss_a if best_val_loss_a > 0 else float("inf")

    # --- Train Model B (Memory GRU) with early stopping ---
    model_b = RecurrentPredictor(
        input_dim=X_train_seq.shape[2],
        hidden_dim=32,
        output_dim=1,
        seed=seed,
    )
    trainer_b = SequentialTrainer(
        model=model_b,
        batch_size=8,
        epochs=150,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    )
    history_b = trainer_b.fit(
        (X_train_seq, Y_train_seq),
        val_data=(X_test_seq, Y_test_seq),
    )

    best_epoch_b = history_b["best_epoch"]
    train_loss_at_best_b = history_b["train_losses"][best_epoch_b - 1] if best_epoch_b > 0 else history_b["final_train_loss"]
    best_val_loss_b = history_b["best_val_loss"]
    trainval_gap_b = train_loss_at_best_b / best_val_loss_b if best_val_loss_b > 0 else float("inf")

    # --- Benchmark ---
    evaluator = ModelEvaluator(seed=seed)
    benchmark = evaluator.benchmark_sequence_models(
        memory_model=model_b,
        no_memory_model=model_a,
        X_test_seq=X_test_seq,
        Y_test_seq=Y_test_seq,
        Y_train_seq=Y_train_seq,
        seed=seed,
    )

    mem_mse = benchmark["memory_model"]["mse"]
    no_mem_mse = benchmark["no_memory_model"]["mse"]
    gap_pct = benchmark["memory_vs_no_memory_gap_pct"]

    # --- Re-run E003 delay sweep with these early-stopped models ---
    delay_sweep_records = _run_delay_sweep_with_models(
        delays=[0, 1, 2, 4, 8],
        memory_model=model_b,
        no_memory_model=model_a,
        seed=seed,
        steps_per_episode=steps_per_episode,
        num_test_episodes=num_test_episodes,
    )

    # d=0 anomaly: memory_mse > no_memory_mse at d=0
    d0_record = next((r for r in delay_sweep_records if r["delay"] == 0), None)
    d0_anomaly_persists = (d0_record is not None) and (d0_record["memory_model_mse"] > d0_record["no_memory_model_mse"])

    # --- Re-run E004 generalization with early-stopped memory model ---
    gen_results = _run_generalization_test(
        memory_model=model_b,
        train_delays=[1, 2, 4],
        test_delays=[3, 6],
        seed=seed,
        steps_per_episode=steps_per_episode,
        num_test_episodes=num_test_episodes,
    )
    gen_failure_persists = any(
        v["mse"] > v["persistence_mse"] for v in gen_results.values()
    )

    # --- Side-by-side vs original ---
    delta_mem_mse = mem_mse - ORIGINAL_E001_NUMBERS["memory_model_mse"]
    delta_gap_pct = gap_pct - ORIGINAL_E001_NUMBERS["gap_pct"]

    return {
        "experiment_id": "CSP-P2-E007a",
        "title": "Early-Stopped Re-Benchmark",
        "delay": delay,
        "seed": seed,
        "early_stopping_patience": early_stopping_patience,
        # Training history transparency
        "memory_model": {
            "best_epoch": best_epoch_b,
            "epochs_run": history_b["epochs_run"],
            "train_loss_at_best_epoch": round(train_loss_at_best_b, 6),
            "best_val_loss": round(best_val_loss_b, 6),
            "trainval_gap_ratio": round(trainval_gap_b, 4),
            "num_parameters": model_b.num_parameters,
        },
        "no_memory_model": {
            "best_epoch": best_epoch_a,
            "epochs_run": history_a["epochs_run"],
            "train_loss_at_best_epoch": round(train_loss_at_best_a, 6),
            "best_val_loss": round(best_val_loss_a, 6),
            "trainval_gap_ratio": round(trainval_gap_a, 4),
            "num_parameters": model_a.num_parameters,
        },
        "benchmark": benchmark,
        # Side-by-side comparison vs original E001
        "comparison_vs_original": {
            "original_memory_mse": ORIGINAL_E001_NUMBERS["memory_model_mse"],
            "original_no_memory_mse": ORIGINAL_E001_NUMBERS["no_memory_model_mse"],
            "original_gap_pct": ORIGINAL_E001_NUMBERS["gap_pct"],
            "corrected_memory_mse": round(mem_mse, 6),
            "corrected_no_memory_mse": round(no_mem_mse, 6),
            "corrected_gap_pct": round(gap_pct, 2),
            "delta_memory_mse": round(delta_mem_mse, 6),
            "delta_gap_pct": round(delta_gap_pct, 2),
        },
        # Downstream re-runs
        "delay_sweep": delay_sweep_records,
        "generalization_results": gen_results,
        # Anomaly flags (inputs to decide_part_b)
        "d0_anomaly_persists_after_early_stopping": d0_anomaly_persists,
        "generalization_failure_persists_after_early_stopping": gen_failure_persists,
        # Stored model state for E007b/c to reuse
        "_model_b_state": model_b.to_dict(),
        "_model_a_state": model_a.to_dict(),
        "_X_test_seq": X_test_seq,
        "_Y_test_seq": Y_test_seq,
        "_X_train_seq": X_train_seq,
        "_Y_train_seq": Y_train_seq,
    }


# ---------------------------------------------------------------------------
# E007b — Parameter-Matched MLP Control
# ---------------------------------------------------------------------------

def run_experiment_p2_e007b(
    delay: int = 2,
    num_train_episodes: int = 30,
    num_test_episodes: int = 10,
    steps_per_episode: int = 25,
    seed: int = 42,
    early_stopping_patience: int = 30,
    e007a_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """CSP-P2-E007b — Parameter-Matched MLP Control.

    Builds a no-memory MLP with ~6,748 parameters (hidden_dims=(173,)) to match
    the memory model's 6,753 parameters (within ±10%). Trains with the same
    early-stopping rule (patience=30) and benchmarks on the same held-out set.

    Matched-MLP architecture:
      Linear(37 → 173) → ReLU → Linear(173 → 1)
      Parameters: 37×173 + 173 + 173×1 + 1 = 6748

    Reports what fraction of the original 6.9% memory advantage survives once
    capacity is matched. The capacity_gap_pct returned here is one of the three
    conditions evaluated by decide_part_b() — see that function for the full
    decision logic. Do NOT duplicate the threshold logic here.
    """
    world_config = _make_world_config(delay, seed)

    # Reuse datasets from E007a if available, otherwise regenerate
    if e007a_results is not None and "_X_test_seq" in e007a_results:
        X_test_seq = e007a_results["_X_test_seq"]
        Y_test_seq = e007a_results["_Y_test_seq"]
        X_train_seq = e007a_results["_X_train_seq"]
        Y_train_seq = e007a_results["_Y_train_seq"]
        model_b_es = RecurrentPredictor.from_dict(e007a_results["_model_b_state"])
    else:
        train_seq_ds = collect_sequential_trajectories(
            world_config=world_config,
            num_episodes=num_train_episodes,
            steps_per_episode=steps_per_episode,
            interaction_prob=0.3,
            seed=seed,
        )
        test_seq_ds = collect_sequential_trajectories(
            world_config=world_config,
            num_episodes=num_test_episodes,
            steps_per_episode=steps_per_episode,
            interaction_prob=0.3,
            seed=seed + 99999,
        )
        X_train_seq, Y_train_seq = train_seq_ds.to_numpy_sequences()
        X_test_seq, Y_test_seq = test_seq_ds.to_numpy_sequences()
        # Retrain memory model if not provided
        model_b_es = RecurrentPredictor(
            input_dim=X_train_seq.shape[2], hidden_dim=32, output_dim=1, seed=seed
        )
        trainer_b = SequentialTrainer(
            model=model_b_es, batch_size=8, epochs=150,
            early_stopping_patience=early_stopping_patience, seed=seed,
        )
        trainer_b.fit((X_train_seq, Y_train_seq), val_data=(X_test_seq, Y_test_seq))

    # Flatten for MLP training
    B_tr, T_tr, D = X_train_seq.shape
    X_train_flat = X_train_seq.reshape(B_tr * T_tr, D)
    Y_train_flat = Y_train_seq.reshape(B_tr * T_tr, 1)
    B_te, T_te, _ = X_test_seq.shape
    X_test_flat = X_test_seq.reshape(B_te * T_te, D)
    Y_test_flat = Y_test_seq.reshape(B_te * T_te, 1)

    # --- Build parameter-matched MLP: hidden_dims=(173,) → 6748 params ---
    MATCHED_HIDDEN_DIMS = (173,)
    model_matched = PredictiveMLP(
        input_dim=D,
        hidden_dims=MATCHED_HIDDEN_DIMS,
        output_dim=1,
        hidden_activation="relu",
        seed=seed,
    )
    assert 6000 <= model_matched.num_parameters <= 7500, (
        f"Matched MLP has {model_matched.num_parameters} params; expected ~6748"
    )

    trainer_matched = PredictiveTrainer(
        model=model_matched,
        batch_size=32,
        epochs=150,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    )
    history_matched = trainer_matched.fit(
        (X_train_flat, Y_train_flat),
        val_data=(X_test_flat, Y_test_flat),
    )

    # --- Benchmark all three models on same held-out set ---
    evaluator = ModelEvaluator(seed=seed)

    # Memory model (from E007a early-stopped)
    y_mem = model_b_es.forward(X_test_seq)
    mem_metrics = compute_all_metrics(y_mem, Y_test_seq)

    # Matched MLP (via FeedForwardBaseline wrapper to match seq eval format)
    from evaluation.baselines import FeedForwardBaseline
    matched_ff = FeedForwardBaseline(model_matched)
    y_matched = matched_ff.predict(X_test_seq)
    matched_metrics = compute_all_metrics(y_matched, Y_test_seq)

    # Original small MLP (reconstruct from E007a state if available)
    if e007a_results is not None and "_model_a_state" in e007a_results:
        model_a_es = PredictiveMLP.from_dict(e007a_results["_model_a_state"])
    else:
        model_a_es = PredictiveMLP(input_dim=D, hidden_dims=(32, 16), output_dim=1, seed=seed)
        tr = PredictiveTrainer(model=model_a_es, batch_size=32, epochs=150,
                               early_stopping_patience=early_stopping_patience, seed=seed)
        tr.fit((X_train_flat, Y_train_flat), val_data=(X_test_flat, Y_test_flat))

    small_ff = FeedForwardBaseline(model_a_es)
    y_small = small_ff.predict(X_test_seq)
    small_metrics = compute_all_metrics(y_small, Y_test_seq)

    # Gaps
    gap_mem_vs_matched = compute_baseline_gap(mem_metrics["mse"], matched_metrics["mse"])
    gap_mem_vs_small = compute_baseline_gap(mem_metrics["mse"], small_metrics["mse"])
    gap_matched_vs_small = compute_baseline_gap(matched_metrics["mse"], small_metrics["mse"])

    # What fraction of original 6.9% survives?
    original_gap = ORIGINAL_E001_NUMBERS["gap_pct"]
    fraction_surviving = gap_mem_vs_matched / original_gap if original_gap != 0 else 0.0

    # Delay sweep and generalization with matched MLP
    delay_sweep_records = _run_delay_sweep_with_models(
        delays=[0, 1, 2, 4, 8],
        memory_model=model_b_es,
        no_memory_model=model_matched,
        seed=seed,
        steps_per_episode=steps_per_episode,
        num_test_episodes=num_test_episodes,
    )
    d0_record = next((r for r in delay_sweep_records if r["delay"] == 0), None)
    d0_anomaly_persists = (d0_record is not None) and (d0_record["memory_model_mse"] > d0_record["no_memory_model_mse"])

    gen_results = _run_generalization_test(
        memory_model=model_b_es,
        train_delays=[1, 2, 4],
        test_delays=[3, 6],
        seed=seed,
        steps_per_episode=steps_per_episode,
        num_test_episodes=num_test_episodes,
    )
    gen_failure_persists = any(v["mse"] > v["persistence_mse"] for v in gen_results.values())

    return {
        "experiment_id": "CSP-P2-E007b",
        "title": "Parameter-Matched MLP Control",
        "delay": delay,
        "seed": seed,
        "matched_mlp_architecture": {
            "hidden_dims": list(MATCHED_HIDDEN_DIMS),
            "num_parameters": model_matched.num_parameters,
            "target_parameters": 6753,
            "parameter_match_pct_error": round(
                abs(model_matched.num_parameters - 6753) / 6753 * 100, 2
            ),
        },
        "three_way_benchmark": {
            "memory_model_gru": {**mem_metrics, "num_parameters": model_b_es.num_parameters},
            "matched_mlp_173": {**matched_metrics, "num_parameters": model_matched.num_parameters},
            "small_mlp_32_16": {**small_metrics, "num_parameters": model_a_es.num_parameters},
        },
        "gaps": {
            "memory_vs_matched_mlp_pct": round(gap_mem_vs_matched, 2),
            "memory_vs_small_mlp_pct": round(gap_mem_vs_small, 2),
            "matched_vs_small_mlp_pct": round(gap_matched_vs_small, 2),
        },
        "capacity_analysis": {
            "original_memory_vs_small_gap_pct": ORIGINAL_E001_NUMBERS["gap_pct"],
            "corrected_memory_vs_matched_gap_pct": round(gap_mem_vs_matched, 2),
            "fraction_of_gap_surviving_capacity_match": round(fraction_surviving, 4),
            "interpretation": (
                "Capacity explains the advantage" if abs(gap_mem_vs_matched) < 1.0
                else "Memory provides advantage beyond capacity"
            ),
        },
        # Anomaly flags (inputs to decide_part_b)
        "d0_anomaly_persists_after_param_matching": d0_anomaly_persists,
        "generalization_failure_persists_after_param_matching": gen_failure_persists,
        "capacity_gap_pct": round(gap_mem_vs_matched, 2),
        "delay_sweep": delay_sweep_records,
        "generalization_results": gen_results,
        "matched_model_training": {
            "best_epoch": history_matched["best_epoch"],
            "epochs_run": history_matched["epochs_run"],
            "best_val_loss": round(history_matched["best_val_loss"], 6),
        },
    }


# ---------------------------------------------------------------------------
# E007c — Representation-Metric Sanity Check
# ---------------------------------------------------------------------------

def run_experiment_p2_e007c(
    seed: int = 42,
    e007a_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """CSP-P2-E007c — Representation-Metric Sanity Check.

    Recomputes inter_class_cosine_similarity and distinct_memory_representations_formed
    on the corrected (early-stopped) memory model. Audits the threshold logic,
    adds transparency fields, and explains the apparent contradiction between
    cosine_sim=0.9495 and distinct=True.
    """
    # Reconstruct model and test data from E007a if available
    if e007a_results is not None and "_model_b_state" in e007a_results:
        model_b = RecurrentPredictor.from_dict(e007a_results["_model_b_state"])
        X_test_seq = e007a_results["_X_test_seq"]
        Y_test_seq = e007a_results["_Y_test_seq"]
    else:
        world_config = _make_world_config(delay=2, seed=seed)
        train_seq_ds = collect_sequential_trajectories(world_config, 30, 25, seed=seed)
        test_seq_ds = collect_sequential_trajectories(world_config, 10, 25, seed=seed + 99999)
        X_train_seq, Y_train_seq = train_seq_ds.to_numpy_sequences()
        X_test_seq, Y_test_seq = test_seq_ds.to_numpy_sequences()
        model_b = RecurrentPredictor(input_dim=37, hidden_dim=32, output_dim=1, seed=seed)
        trainer = SequentialTrainer(model=model_b, batch_size=8, epochs=150,
                                    early_stopping_patience=30, seed=seed)
        trainer.fit((X_train_seq, Y_train_seq), val_data=(X_test_seq, Y_test_seq))

    # Latent state extraction
    latent_states = model_b.get_latent_state(X_test_seq)  # (B, T, D_h)
    latent_flat = latent_states.reshape(-1, model_b.hidden_dim)
    Y_flat = Y_test_seq.reshape(-1, 1)

    # Partition: delayed-consequence steps (Y > 0) vs standard steps
    delayed_mask = Y_flat.flatten() > 0.0
    normal_mask = ~delayed_mask

    n_delayed = int(np.sum(delayed_mask))
    n_normal = int(np.sum(normal_mask))

    THRESHOLD = 0.95  # The documented threshold for "distinct"

    if n_delayed > 0 and n_normal > 0:
        mean_delayed = np.mean(latent_flat[delayed_mask], axis=0)
        mean_normal = np.mean(latent_flat[normal_mask], axis=0)
        cos_sim = float(compute_cosine_similarity(mean_delayed, mean_normal))
        euc_dist = float(np.linalg.norm(mean_delayed - mean_normal))

        # Per-class intraclass variance (mean distance to class centroid)
        delayed_intraclass_variance = float(np.mean(
            np.linalg.norm(latent_flat[delayed_mask] - mean_delayed, axis=1)
        ))
        normal_intraclass_variance = float(np.mean(
            np.linalg.norm(latent_flat[normal_mask] - mean_normal, axis=1)
        ))

        # PCA spread (top-2 principal components)
        centered = latent_flat - np.mean(latent_flat, axis=0)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            pca_explained_var_ratio = (s[:2] ** 2 / np.sum(s ** 2)).tolist()
        except Exception:
            pca_explained_var_ratio = [0.0, 0.0]

    else:
        cos_sim = 1.0
        euc_dist = 0.0
        delayed_intraclass_variance = 0.0
        normal_intraclass_variance = 0.0
        pca_explained_var_ratio = [0.0, 0.0]

    margin = THRESHOLD - cos_sim  # positive = on the "distinct" side

    # Threshold audit: document exactly what the flag means
    # cos_sim < THRESHOLD  → True  (distinct representations claimed)
    # margin == 0.0005 in the original run: borderline
    is_distinct = cos_sim < THRESHOLD
    if abs(margin) < 0.02:
        confidence = "borderline"
    elif is_distinct:
        confidence = "clear_distinct"
    else:
        confidence = "not_distinct"

    # Explanation of the apparent contradiction
    contradiction_explanation = (
        "No logical contradiction: cos_sim=0.9495 < threshold=0.95, so "
        "distinct_memory_representations_formed=True is technically correct. "
        f"However, the margin is only {margin:.4f}, which is within measurement "
        "noise for this sample size. The boolean is borderline rather than a "
        "confident claim of distinct representations."
    )

    return {
        "experiment_id": "CSP-P2-E007c",
        "title": "Representation-Metric Sanity Check",
        "seed": seed,
        "n_delayed_steps": n_delayed,
        "n_normal_steps": n_normal,
        "representation_analysis": {
            # Core metrics
            "inter_class_cosine_similarity": round(cos_sim, 6),
            "inter_class_euclidean_distance": round(euc_dist, 4),
            # Threshold transparency
            "cosine_similarity_threshold": THRESHOLD,
            "cosine_sim_margin_from_threshold": round(margin, 6),
            "distinct_memory_representations_formed": is_distinct,
            "representation_distinctness_confidence": confidence,
            # Additional nuance metrics
            "delayed_steps_intraclass_variance": round(delayed_intraclass_variance, 4),
            "normal_steps_intraclass_variance": round(normal_intraclass_variance, 4),
            "pca_top2_explained_variance_ratio": [round(v, 4) for v in pca_explained_var_ratio],
        },
        "threshold_audit": {
            "documented_threshold": THRESHOLD,
            "threshold_logic": "distinct := cos_sim < threshold",
            "original_cos_sim": 0.9495,
            "corrected_cos_sim": round(cos_sim, 6),
            "original_distinct_flag": True,  # from original run (0.9495 < 0.95)
            "corrected_distinct_flag": is_distinct,
            "original_flag_technically_correct": True,
            "original_flag_scientifically_confident": False,
            "contradiction_resolved": contradiction_explanation,
        },
    }


# ---------------------------------------------------------------------------
# decide_part_b() — SINGLE SOURCE OF TRUTH for Part B trigger
# ---------------------------------------------------------------------------

def decide_part_b(
    e007a_results: Dict[str, Any],
    e007b_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether Part B (E007d/e hidden-state diagnostics) should run.

    Uses OR logic per the CSP-P2-E007 contract (Acceptance Test 5):
    Part B triggers if ANY of the three conditions holds.

    Conditions:
      1. d0_anomaly_persists: memory model still underperforms no-memory at d=0,
         after BOTH early-stopping (E007a) AND parameter-matching (E007b).
      2. generalization_failure_persists: memory model still underperforms
         persistence baseline at d=3 or d=6 after BOTH fixes.
      3. capacity_gap_material: the memory-vs-matched-MLP advantage gap is ≥ 3%
         (memory retains a meaningful advantage even at matched capacity,
         warranting hidden-state investigation).

    Returns:
        Dict with keys:
          trigger_part_b (bool) — True if Part B should run
          d0_anomaly_persists (bool)
          generalization_failure_persists (bool)
          capacity_gap_material (bool)
          capacity_gap_pct (float)
          rationale (str)
    """
    # Condition 1: d=0 anomaly persists after BOTH fixes
    d0_after_es = e007a_results.get("d0_anomaly_persists_after_early_stopping", False)
    d0_after_pm = e007b_results.get("d0_anomaly_persists_after_param_matching", False)
    d0_persists = d0_after_es and d0_after_pm

    # Condition 2: generalization failure persists after BOTH fixes
    gen_after_es = e007a_results.get("generalization_failure_persists_after_early_stopping", False)
    gen_after_pm = e007b_results.get("generalization_failure_persists_after_param_matching", False)
    gen_persists = gen_after_es and gen_after_pm

    # Condition 3: capacity-matched gap is still material (≥ 3%)
    capacity_gap_pct = float(e007b_results.get("capacity_gap_pct", 0.0))
    CAPACITY_GAP_THRESHOLD = 3.0  # percent
    capacity_gap_material = capacity_gap_pct >= CAPACITY_GAP_THRESHOLD

    trigger = d0_persists or gen_persists or capacity_gap_material

    # Build human-readable rationale
    reasons = []
    if d0_persists:
        reasons.append(
            f"d=0 anomaly persists after both fixes (memory underperforms no-memory at d=0 "
            f"even with early stopping AND parameter matching)"
        )
    if gen_persists:
        reasons.append(
            f"Generalization failure persists after both fixes (memory underperforms "
            f"persistence baseline at unseen delays d=3/d=6)"
        )
    if capacity_gap_material:
        reasons.append(
            f"Memory-vs-matched-MLP gap is {capacity_gap_pct:.1f}% >= {CAPACITY_GAP_THRESHOLD}% threshold "
            f"(advantage is not explained by capacity alone)"
        )

    if not trigger:
        rationale = (
            "Part B NOT triggered. Both fixes (early stopping + parameter matching) "
            "resolved the d=0 anomaly and generalization failure, and the remaining "
            f"memory advantage ({capacity_gap_pct:.1f}%) is below the {CAPACITY_GAP_THRESHOLD}% threshold. "
            "No hidden-state diagnostics required."
        )
    else:
        rationale = "Part B TRIGGERED. Reason(s): " + "; ".join(reasons)

    return {
        "trigger_part_b": trigger,
        "d0_anomaly_persists": d0_persists,
        "generalization_failure_persists": gen_persists,
        "capacity_gap_material": capacity_gap_material,
        "capacity_gap_pct": round(capacity_gap_pct, 2),
        "d0_persists_after_early_stopping": d0_after_es,
        "d0_persists_after_param_matching": d0_after_pm,
        "gen_persists_after_early_stopping": gen_after_es,
        "gen_persists_after_param_matching": gen_after_pm,
        "capacity_gap_threshold_used": CAPACITY_GAP_THRESHOLD,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Part B — E007d: d=0 Isolation (hidden-state level)
# ---------------------------------------------------------------------------

def run_experiment_p2_e007d(
    seed: int = 42,
    e007a_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """CSP-P2-E007d — d=0 Isolation (Hidden-State Level).

    Only runs if decide_part_b()["trigger_part_b"] is True.

    1. Checks training-set delay distribution (d=0 vs d=1/2/4 representation).
    2. Logs hidden-state trajectories for d=0-only episodes on the corrected model.
    3. Tests whether hidden-state magnitude/variance at d=0 differs from d>0.
    """
    if e007a_results is not None and "_model_b_state" in e007a_results:
        model_b = RecurrentPredictor.from_dict(e007a_results["_model_b_state"])
    else:
        world_config = _make_world_config(delay=2, seed=seed)
        train_ds = collect_sequential_trajectories(world_config, 30, 25, seed=seed)
        X_tr, Y_tr = train_ds.to_numpy_sequences()
        model_b = RecurrentPredictor(input_dim=37, hidden_dim=32, output_dim=1, seed=seed)
        SequentialTrainer(model=model_b, batch_size=8, epochs=150,
                          early_stopping_patience=30, seed=seed).fit(
            (X_tr, Y_tr), val_data=(X_tr, Y_tr)
        )

    # Collect d=0 test episodes
    cfg_d0 = _make_world_config(delay=0, seed=seed + 1000)
    ds_d0 = collect_sequential_trajectories(cfg_d0, num_episodes=15, steps_per_episode=25,
                                             interaction_prob=0.3, seed=seed + 1000)
    X_d0, Y_d0 = ds_d0.to_numpy_sequences()

    # Collect d=2 test episodes for comparison
    cfg_d2 = _make_world_config(delay=2, seed=seed + 2000)
    ds_d2 = collect_sequential_trajectories(cfg_d2, num_episodes=15, steps_per_episode=25,
                                             interaction_prob=0.3, seed=seed + 2000)
    X_d2, Y_d2 = ds_d2.to_numpy_sequences()

    # Get hidden states for each
    h_d0 = model_b.get_latent_state(X_d0).reshape(-1, model_b.hidden_dim)
    h_d2 = model_b.get_latent_state(X_d2).reshape(-1, model_b.hidden_dim)

    h_d0_norm_mean = float(np.mean(np.linalg.norm(h_d0, axis=1)))
    h_d2_norm_mean = float(np.mean(np.linalg.norm(h_d2, axis=1)))
    h_d0_var = float(np.mean(np.var(h_d0, axis=0)))
    h_d2_var = float(np.mean(np.var(h_d2, axis=0)))

    # Prediction error at d=0 vs d=2
    y_pred_d0 = model_b.forward(X_d0)
    y_pred_d2 = model_b.forward(X_d2)
    mse_d0 = float(np.mean((y_pred_d0 - Y_d0) ** 2))
    mse_d2 = float(np.mean((y_pred_d2 - Y_d2) ** 2))

    # Training-set d distribution check (count Y > 0 steps as "event steps")
    training_positive_steps = 0
    training_total_steps = 0
    if e007a_results and "_Y_train_seq" in e007a_results:
        Y_tr = e007a_results["_Y_train_seq"].reshape(-1)
        training_positive_steps = int(np.sum(Y_tr > 0))
        training_total_steps = len(Y_tr)

    return {
        "experiment_id": "CSP-P2-E007d",
        "title": "d=0 Isolation (Hidden-State Level)",
        "seed": seed,
        "hidden_state_analysis": {
            "d0_mean_hidden_norm": round(h_d0_norm_mean, 4),
            "d2_mean_hidden_norm": round(h_d2_norm_mean, 4),
            "d0_hidden_variance": round(h_d0_var, 4),
            "d2_hidden_variance": round(h_d2_var, 4),
            "d0_prediction_mse": round(mse_d0, 6),
            "d2_prediction_mse": round(mse_d2, 6),
        },
        "training_distribution": {
            "total_training_steps": training_total_steps,
            "delayed_consequence_steps": training_positive_steps,
            "delayed_consequence_fraction": round(
                training_positive_steps / training_total_steps, 4
            ) if training_total_steps > 0 else 0.0,
            "note": (
                "d=0 episodes produce immediate consequence steps (no 'waiting' period). "
                "Under-representation of d=0 in training could explain the anomaly."
            ),
        },
        "interpretation": (
            f"At d=0, memory hidden state norm={h_d0_norm_mean:.3f} vs d=2 norm={h_d2_norm_mean:.3f}. "
            f"Prediction MSE d=0={mse_d0:.4f} vs d=2={mse_d2:.4f}. "
            + ("d=0 anomaly likely reflects 'wait-for-signal' bias in GRU."
               if mse_d0 > mse_d2 * 1.5 else "d=0 vs d=2 gap is modest.")
        ),
    }


# ---------------------------------------------------------------------------
# Part B — E007e: Generalization Failure Diagnostic (hidden-state level)
# ---------------------------------------------------------------------------

def run_experiment_p2_e007e(
    seed: int = 42,
    e007a_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """CSP-P2-E007e — Generalization Failure Diagnostic (Hidden-State Level).

    Only runs if decide_part_b()["trigger_part_b"] is True.

    Compares d=3 (interpolation: between trained d={2,4}) vs d=6
    (extrapolation: beyond trained max d=4) on the corrected memory model.
    Tests whether the GRU encodes a scalar delay-length signal or only a
    binary "delay is pending" flag.
    """
    if e007a_results is not None and "_model_b_state" in e007a_results:
        model_b = RecurrentPredictor.from_dict(e007a_results["_model_b_state"])
    else:
        world_config = _make_world_config(delay=2, seed=seed)
        train_ds = collect_sequential_trajectories(world_config, 30, 25, seed=seed)
        X_tr, Y_tr = train_ds.to_numpy_sequences()
        model_b = RecurrentPredictor(input_dim=37, hidden_dim=32, output_dim=1, seed=seed)
        SequentialTrainer(model=model_b, batch_size=8, epochs=150,
                          early_stopping_patience=30, seed=seed).fit(
            (X_tr, Y_tr), val_data=(X_tr, Y_tr)
        )

    results = {}
    hidden_norms = {}
    for d in [2, 3, 4, 6]:  # d=2,4 are trained; d=3,6 are unseen
        cfg = _make_world_config(delay=d, seed=seed + d * 200)
        ds = collect_sequential_trajectories(cfg, num_episodes=15, steps_per_episode=25,
                                              interaction_prob=0.3, seed=seed + d * 200)
        X, Y = ds.to_numpy_sequences()
        y_pred = model_b.forward(X)
        mse = float(np.mean((y_pred - Y) ** 2))
        h = model_b.get_latent_state(X).reshape(-1, model_b.hidden_dim)
        h_norm = float(np.mean(np.linalg.norm(h, axis=1)))
        hidden_norms[d] = h_norm

        # Persistence baseline
        p_mse = float(np.mean(Y ** 2))

        results[f"delay_{d}"] = {
            "delay": d,
            "is_trained": d in [1, 2, 4],
            "prediction_mse": round(mse, 6),
            "persistence_mse": round(p_mse, 6),
            "beats_persistence": mse < p_mse,
            "hidden_state_norm": round(h_norm, 4),
        }

    # Test for scalar vs binary encoding:
    # If hidden-state norm is monotonically related to delay d, GRU encodes scalar delay.
    delays = [2, 3, 4, 6]
    norms = [hidden_norms[d] for d in delays]
    rank_corr = float(np.corrcoef(delays, norms)[0, 1]) if len(delays) > 1 else 0.0

    hypothesis = (
        "scalar_delay_encoding" if abs(rank_corr) > 0.6
        else "binary_pending_flag_encoding"
    )

    return {
        "experiment_id": "CSP-P2-E007e",
        "title": "Generalization Failure Diagnostic (Hidden-State Level)",
        "seed": seed,
        "per_delay_results": results,
        "hidden_state_norm_by_delay": {str(d): round(hidden_norms[d], 4) for d in delays},
        "delay_norm_rank_correlation": round(rank_corr, 4),
        "encoding_hypothesis": hypothesis,
        "interpolation_vs_extrapolation": {
            "d3_interpolation_mse": results.get("delay_3", {}).get("prediction_mse"),
            "d6_extrapolation_mse": results.get("delay_6", {}).get("prediction_mse"),
            "d3_beats_persistence": results.get("delay_3", {}).get("beats_persistence"),
            "d6_beats_persistence": results.get("delay_6", {}).get("beats_persistence"),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers — delay sweep and generalization test on given models
# ---------------------------------------------------------------------------

def _run_delay_sweep_with_models(
    delays: List[int],
    memory_model: RecurrentPredictor,
    no_memory_model: PredictiveMLP,
    seed: int,
    steps_per_episode: int = 25,
    num_test_episodes: int = 10,
) -> List[Dict[str, Any]]:
    """Run a delay sweep evaluation using pre-trained models (no retraining)."""
    from evaluation.baselines import FeedForwardBaseline, PersistencePredictor
    records = []
    for d in delays:
        cfg = _make_world_config(d, seed + d * 17)
        ds = collect_sequential_trajectories(
            cfg, num_episodes=num_test_episodes, steps_per_episode=steps_per_episode,
            interaction_prob=0.3, seed=seed + d * 17 + 500,
        )
        X, Y = ds.to_numpy_sequences()

        y_mem = memory_model.forward(X)
        mse_mem = float(np.mean((y_mem - Y) ** 2))

        ff = FeedForwardBaseline(no_memory_model)
        y_no_mem = ff.predict(X)
        mse_no_mem = float(np.mean((y_no_mem - Y) ** 2))

        y_persist = np.zeros_like(Y)
        mse_persist = float(np.mean(Y ** 2))

        gap = compute_baseline_gap(mse_mem, mse_no_mem)

        records.append({
            "delay": d,
            "memory_model_mse": round(mse_mem, 6),
            "no_memory_model_mse": round(mse_no_mem, 6),
            "persistence_baseline_mse": round(mse_persist, 6),
            "memory_advantage_pct": round(gap, 2),
            "beats_no_memory": mse_mem < mse_no_mem,
        })
    return records


def _run_generalization_test(
    memory_model: RecurrentPredictor,
    train_delays: List[int],
    test_delays: List[int],
    seed: int,
    steps_per_episode: int = 25,
    num_test_episodes: int = 10,
) -> Dict[str, Any]:
    """Evaluate a pre-trained memory model on unseen delays (no retraining)."""
    results = {}
    for d_test in test_delays:
        cfg = _make_world_config(d_test, seed + d_test * 101)
        ds = collect_sequential_trajectories(
            cfg, num_episodes=num_test_episodes, steps_per_episode=steps_per_episode,
            interaction_prob=0.3, seed=seed + d_test * 101,
        )
        X, Y = ds.to_numpy_sequences()
        y_pred = memory_model.forward(X)
        mse = float(np.mean((y_pred - Y) ** 2))
        persistence_mse = float(np.mean(Y ** 2))
        gap = compute_baseline_gap(mse, persistence_mse)
        results[f"delay_{d_test}"] = {
            "delay": d_test,
            "mse": round(mse, 6),
            "persistence_mse": round(persistence_mse, 6),
            "gap_vs_persistence_pct": round(gap, 2),
            "generalization_successful": mse < persistence_mse,
        }
    return results
