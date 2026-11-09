"""Experimental Suite for Project Caspian (Phase 3 — Generalization).

Addresses Section 30 of the Development Plan: does Caspian learn a reusable
relationship, or does it memorize the specific training configuration?

- CSP-P3-E001: Position generalization — train on one subset of grid cells,
  evaluate on a disjoint held-out subset never seen during training.
- CSP-P3-E002: Layout generalization — train on a 5x5 grid, evaluate on an
  unseen 7x7 grid.
- CSP-P3-E003: Entity-appearance generalization — train with exactly one
  entity of one type, evaluate on (a) two entities present simultaneously and
  (b) a single entity of a never-seen entity_type code.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import WorldConfig
from models.mlp import PredictiveMLP
from models.deepsets import DeepSetsPredictiveModel
from learning.collector import TrajectoryCollector
from learning.trainer import PredictiveTrainer
from learning.deepsets_trainer import DeepSetsTrainer
from evaluation.metrics import compute_all_metrics
from evaluation.baselines import PersistencePredictor, RandomPredictor, ReactivePredictor

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _load_world(name: str) -> WorldConfig:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return WorldConfig.from_dict(json.load(f))


def _train_mlp(world_config: WorldConfig, seed: int, num_steps: int = 4000):
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed)
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)

    X_train, Y_train = train_ds.to_numpy_arrays()
    X_val, Y_val = val_ds.to_numpy_arrays()

    model = PredictiveMLP(input_dim=X_train.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit((X_train, Y_train), val_data=(X_val, Y_val))
    return model, history, (X_val, Y_val)


def _eval_on_world(model: PredictiveMLP, world_config: WorldConfig, seed: int, num_steps: int = 1000) -> Dict[str, Any]:
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed + 9000)
    X, Y = dataset.to_numpy_arrays()
    return compute_all_metrics(model.forward(X), Y), (X, Y)


def _baseline_metrics(Y_train: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
    persist = PersistencePredictor(default_value=0.0)
    rand = RandomPredictor(low=-1.0, high=10.0, seed=0)
    reactive = ReactivePredictor(mean_value=float(np.mean(Y_train)))
    return {
        "persistence": compute_all_metrics(persist.predict(X), Y),
        "random": compute_all_metrics(rand.predict(X), Y),
        "reactive": compute_all_metrics(reactive.predict(X), Y),
    }


def run_experiment_e001(seed: int = 42) -> Dict[str, Any]:
    """CSP-P3-E001: Position generalization — held-out grid cells never used in training."""
    train_world = _load_world("phase3_world_train_positions.json")
    test_world = _load_world("phase3_world_test_positions.json")

    model, history, (X_val, Y_val) = _train_mlp(train_world, seed)
    id_metrics = compute_all_metrics(model.forward(X_val), Y_val)  # in-distribution (held-out steps, same cell set)

    ood_metrics, (X_ood, Y_ood) = _eval_on_world(model, test_world, seed)
    baselines = _baseline_metrics(Y_val, X_ood, Y_ood)

    gen_gap_pct = round(100.0 * (ood_metrics["mse"] - id_metrics["mse"]) / (id_metrics["mse"] + 1e-10), 2)

    return {
        "experiment_id": "CSP-P3-E001",
        "title": "Position Generalization",
        "seed": seed,
        "train_cells": len(train_world.allowed_entity_positions),
        "test_cells": len(test_world.allowed_entity_positions),
        "in_distribution_metrics": id_metrics,
        "held_out_position_metrics": ood_metrics,
        "baselines_on_held_out_positions": baselines,
        "generalization_gap_pct": gen_gap_pct,
        "generalizes_beyond_trivial_baseline": ood_metrics["mse"] < baselines["persistence"]["mse"],
        "training_history": {"epochs_run": history["epochs_run"], "final_train_loss": history["final_train_loss"]},
    }


def run_experiment_e002(seed: int = 42) -> Dict[str, Any]:
    """CSP-P3-E002: Layout generalization — train on a 5x5 grid, test on an unseen 7x7 grid."""
    train_world = _load_world("phase3_world_train_layout.json")
    test_world = _load_world("phase3_world_test_layout.json")

    model, history, (X_val, Y_val) = _train_mlp(train_world, seed)
    id_metrics = compute_all_metrics(model.forward(X_val), Y_val)

    ood_metrics, (X_ood, Y_ood) = _eval_on_world(model, test_world, seed)
    baselines = _baseline_metrics(Y_val, X_ood, Y_ood)

    gen_gap_pct = round(100.0 * (ood_metrics["mse"] - id_metrics["mse"]) / (id_metrics["mse"] + 1e-10), 2)

    return {
        "experiment_id": "CSP-P3-E002",
        "title": "Layout Generalization (5x5 -> 7x7 grid)",
        "seed": seed,
        "train_grid": f"{train_world.grid_width}x{train_world.grid_height}",
        "test_grid": f"{test_world.grid_width}x{test_world.grid_height}",
        "in_distribution_metrics": id_metrics,
        "unseen_layout_metrics": ood_metrics,
        "baselines_on_unseen_layout": baselines,
        "generalization_gap_pct": gen_gap_pct,
        "generalizes_beyond_trivial_baseline": ood_metrics["mse"] < baselines["persistence"]["mse"],
        "training_history": {"epochs_run": history["epochs_run"], "final_train_loss": history["final_train_loss"]},
    }


def _train_deepsets(world_config: WorldConfig, seed: int, num_steps: int = 4000):
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed)
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)

    train_data = train_ds.to_deepsets_arrays()
    val_data = val_ds.to_deepsets_arrays()

    model = DeepSetsPredictiveModel(
        entity_feature_dim=4, entity_hidden_dim=16, global_feature_dim=train_data[2].shape[1],
        head_hidden_dim=24, output_dim=1, max_entities=5, seed=seed,
    )
    trainer = DeepSetsTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit(train_data, val_data=val_data)
    return model, history, val_data


def _eval_deepsets_on_world(model: DeepSetsPredictiveModel, world_config: WorldConfig, seed: int, num_steps: int = 1000):
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed + 9000)
    EF, PM, GF, Y = dataset.to_deepsets_arrays()
    return compute_all_metrics(model.forward(EF, PM, GF), Y), (EF, PM, GF, Y)


def run_experiment_e003(seed: int = 42) -> Dict[str, Any]:
    """CSP-P3-E003: Entity-appearance generalization — unseen entity count and unseen entity_type.

    Compares two architectures head-to-head: the fixed-slot stateless MLP (which
    assigns each entity to a specific input position) against a permutation-invariant
    DeepSets model (which pools over entities as an unordered set, robust to count).
    """
    train_world = _load_world("phase3_world_train_entity.json")
    multi_world = _load_world("phase3_world_test_multi_entity.json")
    newtype_world = _load_world("phase3_world_test_new_entity_type.json")

    # --- Fixed-slot MLP (baseline architecture) ---
    mlp_model, mlp_history, (X_val, Y_val) = _train_mlp(train_world, seed)
    mlp_id_metrics = compute_all_metrics(mlp_model.forward(X_val), Y_val)
    mlp_multi_metrics, (X_multi, Y_multi) = _eval_on_world(mlp_model, multi_world, seed)
    mlp_newtype_metrics, (X_new, Y_new) = _eval_on_world(mlp_model, newtype_world, seed)
    baselines_multi = _baseline_metrics(Y_val, X_multi, Y_multi)
    baselines_newtype = _baseline_metrics(Y_val, X_new, Y_new)

    # --- Permutation-invariant DeepSets model ---
    # 10,000 collection steps (vs. 4,000 for the MLP): the DeepSets model's entity
    # encoder sees only 4 raw features per entity (no dilution from 33 other input
    # dims like the flat MLP has), so it needs more data to properly regularize the
    # weight on entity_type -- which is constant across all training episodes and
    # therefore gets near-zero training signal either way. Max-pooling (not mean-
    # pooling) is what actually solved the multi-entity generalization gap: see
    # DeepSetsPredictiveModel's docstring/comments for why.
    ds_model, ds_history, ds_val_data = _train_deepsets(train_world, seed, num_steps=10000)
    ds_id_metrics = compute_all_metrics(ds_model.forward(*ds_val_data[:3]), ds_val_data[3])
    ds_multi_metrics, _ = _eval_deepsets_on_world(ds_model, multi_world, seed)
    ds_newtype_metrics, _ = _eval_deepsets_on_world(ds_model, newtype_world, seed)

    def gap_pct(ood_mse, id_mse):
        return round(100.0 * (ood_mse - id_mse) / (id_mse + 1e-10), 2)

    return {
        "experiment_id": "CSP-P3-E003",
        "title": "Entity-Appearance Generalization (novel count / novel type code)",
        "seed": seed,
        "fixed_slot_mlp": {
            "in_distribution_metrics": mlp_id_metrics,
            "two_entities_simultaneous_metrics": mlp_multi_metrics,
            "novel_entity_type_metrics": mlp_newtype_metrics,
            "generalization_gap_pct_multi_entity": gap_pct(mlp_multi_metrics["mse"], mlp_id_metrics["mse"]),
            "generalization_gap_pct_novel_type": gap_pct(mlp_newtype_metrics["mse"], mlp_id_metrics["mse"]),
        },
        "permutation_invariant_deepsets": {
            "in_distribution_metrics": ds_id_metrics,
            "two_entities_simultaneous_metrics": ds_multi_metrics,
            "novel_entity_type_metrics": ds_newtype_metrics,
            "generalization_gap_pct_multi_entity": gap_pct(ds_multi_metrics["mse"], ds_id_metrics["mse"]),
            "generalization_gap_pct_novel_type": gap_pct(ds_newtype_metrics["mse"], ds_id_metrics["mse"]),
        },
        "baselines_two_entities": baselines_multi,
        "baselines_novel_entity_type": baselines_newtype,
        "deepsets_beats_mlp_on_multi_entity": ds_multi_metrics["mse"] < mlp_multi_metrics["mse"],
        "deepsets_beats_mlp_on_novel_type": ds_newtype_metrics["mse"] < mlp_newtype_metrics["mse"],
        "multi_entity_mse_reduction_pct": round(
            100.0 * (mlp_multi_metrics["mse"] - ds_multi_metrics["mse"]) / (mlp_multi_metrics["mse"] + 1e-10), 2
        ),
        "handles_multi_entity_beyond_baseline": ds_multi_metrics["mse"] < baselines_multi["persistence"]["mse"],
        "handles_novel_type_beyond_baseline": ds_newtype_metrics["mse"] < baselines_newtype["persistence"]["mse"],
        "mlp_training_history": {"epochs_run": mlp_history["epochs_run"], "final_train_loss": mlp_history["final_train_loss"]},
        "deepsets_training_history": {"epochs_run": ds_history["epochs_run"], "final_train_loss": ds_history["final_train_loss"]},
    }


def run_all_phase3_experiments(output_dir: str = "logs") -> Dict[str, Any]:
    """Execute all Phase 3 experiments sequentially and persist results."""
    import os

    print("Running CSP-P3-E001 (Position Generalization)...")
    e001 = run_experiment_e001()

    print("Running CSP-P3-E002 (Layout Generalization)...")
    e002 = run_experiment_e002()

    print("Running CSP-P3-E003 (Entity-Appearance Generalization)...")
    e003 = run_experiment_e003()

    all_results = {
        "phase": "Phase 3 — Generalization",
        "experiments": {
            "CSP-P3-E001": e001,
            "CSP-P3-E002": e002,
            "CSP-P3-E003": e003,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase3_all_experiments_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Phase 3 experiments completed and saved to: {out_file}")
    return all_results


if __name__ == "__main__":
    run_all_phase3_experiments()
