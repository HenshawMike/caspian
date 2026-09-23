"""Comprehensive evaluation driver for Project Caspian (Phase 1).

Benchmarks learned neural models against trivial baselines on test datasets,
held-out spatial configurations, and internal representation quality.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from models.mlp import PredictiveMLP
from models.recurrent import RecurrentPredictor
from evaluation.metrics import compute_all_metrics, compute_baseline_gap, compute_cosine_similarity
from evaluation.baselines import PersistencePredictor, RandomPredictor, ReactivePredictor, FeedForwardBaseline
from learning.dataset import ExperienceDataset
from learning.collector import TrajectoryCollector
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity


class ModelEvaluator:
    """Evaluates predictive accuracy and representation quality of Caspian models."""

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed

    def evaluate_model(
        self,
        model: PredictiveMLP,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate a model on test features and ground-truth targets."""
        y_pred = model.forward(X_test)
        return compute_all_metrics(y_pred, Y_test)

    def benchmark_against_baselines(
        self,
        model: PredictiveMLP,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        Y_train: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full benchmark comparing Caspian model against all baselines.

        Args:
            model: Trained PredictiveMLP model.
            X_test: Test features of shape (N, input_dim).
            Y_test: Test targets of shape (N, 1).
            Y_train: Optional training targets to fit the Reactive baseline.
            seed: Seed for random baseline.

        Returns:
            Dict[str, Any]: Detailed metrics for all models and relative baseline gaps.
        """
        eval_seed = seed if seed is not None else self.seed

        # 1. Evaluate Learned Model
        y_model_pred = model.forward(X_test)
        model_metrics = compute_all_metrics(y_model_pred, Y_test)

        # 2. Evaluate Persistence Baseline
        persistence = PersistencePredictor(default_value=0.0)
        y_persist_pred = persistence.predict(X_test)
        persist_metrics = compute_all_metrics(y_persist_pred, Y_test)

        # 3. Evaluate Random Baseline
        random_pred = RandomPredictor(low=-1.0, high=10.0, seed=eval_seed)
        y_rand_pred = random_pred.predict(X_test)
        rand_metrics = compute_all_metrics(y_rand_pred, Y_test)

        # 4. Evaluate Reactive (Mean) Baseline
        mean_val = float(np.mean(Y_train)) if Y_train is not None else float(np.mean(Y_test))
        reactive = ReactivePredictor(mean_value=mean_val)
        y_react_pred = reactive.predict(X_test)
        react_metrics = compute_all_metrics(y_react_pred, Y_test)

        # 5. Compute Baseline Gaps (% error reduction)
        gap_persistence = compute_baseline_gap(model_metrics["mse"], persist_metrics["mse"])
        gap_random = compute_baseline_gap(model_metrics["mse"], rand_metrics["mse"])
        gap_reactive = compute_baseline_gap(model_metrics["mse"], react_metrics["mse"])

        return {
            "caspian_model": model_metrics,
            "persistence_baseline": persist_metrics,
            "random_baseline": rand_metrics,
            "reactive_baseline": react_metrics,
            "baseline_gap_persistence_pct": round(gap_persistence, 2),
            "baseline_gap_random_pct": round(gap_random, 2),
            "baseline_gap_reactive_pct": round(gap_reactive, 2),
            "beats_persistence": model_metrics["mse"] < persist_metrics["mse"],
            "beats_random": model_metrics["mse"] < rand_metrics["mse"],
            "beats_reactive": model_metrics["mse"] < react_metrics["mse"],
        }

    def evaluate_spatial_generalization(
        self,
        model: PredictiveMLP,
        test_positions: List[Tuple[int, int]],
        grid_size: Tuple[int, int] = (5, 5),
        steps_per_pos: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Test model on unseen entity spatial positions.

        Args:
            model: Trained model.
            test_positions: List of (x, y) coordinates for the entity.
            grid_size: (width, height) of the grid.
            steps_per_pos: Number of exploratory steps per position.
            seed: Seed for evaluation.

        Returns:
            Dict[str, Any]: Generalization results per entity position.
        """
        results_per_pos = {}
        all_mses = []

        for pos in test_positions:
            config = WorldConfig(
                grid_width=grid_size[0],
                grid_height=grid_size[1],
                initial_agent_pos=(0, 0),
                initial_entities=[
                    Entity(
                        entity_id=1,
                        entity_type=1,
                        position=pos,
                        is_interactive=True,
                        is_blocking=False,
                        hidden_state_delta=10.0,
                    )
                ],
                seed=seed,
            )
            collector = TrajectoryCollector(world_config=config)
            test_ds = collector.collect_exploratory_dataset(
                num_steps=steps_per_pos,
                interaction_prob=0.25,
                seed=seed + pos[0] * 10 + pos[1],
            )
            X_test, Y_test = test_ds.to_numpy_arrays()
            metrics = self.evaluate_model(model, X_test, Y_test)
            results_per_pos[f"pos_{pos[0]}_{pos[1]}"] = metrics
            all_mses.append(metrics["mse"])

        return {
            "per_position_results": results_per_pos,
            "mean_generalization_mse": round(float(np.mean(all_mses)), 6),
            "max_generalization_mse": round(float(np.max(all_mses)), 6),
            "min_generalization_mse": round(float(np.min(all_mses)), 6),
        }

    def evaluate_representation_structure(
        self,
        model: PredictiveMLP,
        dataset: ExperienceDataset,
    ) -> Dict[str, Any]:
        """Analyze the geometric structure of latent states S_t."""
        X, Y = dataset.to_numpy_arrays()
        if len(X) == 0:
            return {"status": "empty_dataset"}

        latent_states = model.get_latent_state(X)  # shape (N, d_latent)

        # Distinguish interaction events (delta > 0) from movement steps (delta < 0)
        interaction_mask = (Y.flatten() > 0.0)
        movement_mask = ~interaction_mask

        if np.sum(interaction_mask) == 0 or np.sum(movement_mask) == 0:
            return {"status": "insufficient_class_variance"}

        interaction_states = latent_states[interaction_mask]
        movement_states = latent_states[movement_mask]

        mean_interaction = np.mean(interaction_states, axis=0)
        mean_movement = np.mean(movement_states, axis=0)

        cosine_sim_between_classes = compute_cosine_similarity(mean_interaction, mean_movement)
        euclidean_dist_between_classes = float(np.linalg.norm(mean_interaction - mean_movement))

        # Intraclass cohesion
        inter_intraclass_sim = float(np.mean([
            compute_cosine_similarity(s, mean_interaction) for s in interaction_states[:50]
        ]))
        move_intraclass_sim = float(np.mean([
            compute_cosine_similarity(s, mean_movement) for s in movement_states[:50]
        ]))

        return {
            "latent_dimension": latent_states.shape[1],
            "inter_class_cosine_similarity": round(cosine_sim_between_classes, 4),
            "inter_class_euclidean_distance": round(euclidean_dist_between_classes, 4),
            "interaction_intraclass_cohesion": round(inter_intraclass_sim, 4),
            "movement_intraclass_cohesion": round(move_intraclass_sim, 4),
            "distinct_representations_formed": cosine_sim_between_classes < 0.95,
        }

    def benchmark_sequence_models(
        self,
        memory_model: RecurrentPredictor,
        no_memory_model: PredictiveMLP,
        X_test_seq: np.ndarray,
        Y_test_seq: np.ndarray,
        Y_train_seq: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Benchmark Memory model (Model B) against No-Memory baseline (Model A) and trivial baselines.

        Args:
            memory_model: Trained RecurrentPredictor (Model B).
            no_memory_model: Trained PredictiveMLP (Model A).
            X_test_seq: 3D test sequence features of shape (num_episodes, seq_len, in_dim).
            Y_test_seq: 3D test sequence targets of shape (num_episodes, seq_len, 1).
            Y_train_seq: Optional training sequence targets for Reactive baseline.
            seed: Random seed for stochastic baselines.

        Returns:
            Dict[str, Any]: Comprehensive comparison metrics across models and baselines.
        """
        eval_seed = seed if seed is not None else self.seed

        # 1. Model B — Memory Model (RecurrentPredictor)
        y_memory_pred = memory_model.forward(X_test_seq)
        memory_metrics = compute_all_metrics(y_memory_pred, Y_test_seq)

        # 2. Model A — No-Memory Baseline (PredictiveMLP)
        ff_baseline = FeedForwardBaseline(no_memory_model)
        y_no_memory_pred = ff_baseline.predict(X_test_seq)
        no_memory_metrics = compute_all_metrics(y_no_memory_pred, Y_test_seq)

        # 3. Persistence Baseline
        persistence = PersistencePredictor(default_value=0.0)
        y_persist_pred = persistence.predict(X_test_seq)
        persist_metrics = compute_all_metrics(y_persist_pred, Y_test_seq)

        # 4. Random Baseline
        random_pred = RandomPredictor(low=-1.0, high=10.0, seed=eval_seed)
        y_rand_pred = random_pred.predict(X_test_seq)
        rand_metrics = compute_all_metrics(y_rand_pred, Y_test_seq)

        # 5. Reactive (Empirical Mean) Baseline
        mean_val = float(np.mean(Y_train_seq)) if Y_train_seq is not None else float(np.mean(Y_test_seq))
        reactive = ReactivePredictor(mean_value=mean_val)
        y_react_pred = reactive.predict(X_test_seq)
        react_metrics = compute_all_metrics(y_react_pred, Y_test_seq)

        # 6. Comparative Gaps
        gap_vs_no_memory = compute_baseline_gap(memory_metrics["mse"], no_memory_metrics["mse"])
        gap_vs_persistence = compute_baseline_gap(memory_metrics["mse"], persist_metrics["mse"])
        gap_vs_random = compute_baseline_gap(memory_metrics["mse"], rand_metrics["mse"])
        gap_vs_reactive = compute_baseline_gap(memory_metrics["mse"], react_metrics["mse"])

        return {
            "memory_model": memory_metrics,
            "no_memory_model": no_memory_metrics,
            "persistence_baseline": persist_metrics,
            "random_baseline": rand_metrics,
            "reactive_baseline": react_metrics,
            "memory_vs_no_memory_gap_pct": round(gap_vs_no_memory, 2),
            "memory_vs_persistence_gap_pct": round(gap_vs_persistence, 2),
            "memory_vs_random_gap_pct": round(gap_vs_random, 2),
            "memory_vs_reactive_gap_pct": round(gap_vs_reactive, 2),
            "beats_no_memory": memory_metrics["mse"] < no_memory_metrics["mse"],
            "beats_persistence": memory_metrics["mse"] < persist_metrics["mse"],
            "beats_random": memory_metrics["mse"] < rand_metrics["mse"],
            "beats_reactive": memory_metrics["mse"] < react_metrics["mse"],
        }
