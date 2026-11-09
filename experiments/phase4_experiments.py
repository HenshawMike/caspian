"""Experimental Suite for Project Caspian (Phase 4 — World Model).

Addresses Section 31 of the Development Plan: move from single-step ΔE
prediction to a model of P(E_{t+1}|E_t,A_t), usable for multi-step simulation
(Ê_{t+1} -> Ê_{t+2} -> ...) and planning.

- CSP-P4-E001: Single-step full-state prediction (position delta + internal-state
  delta), including an explicit check of the "entities are static" assumption
  the rollout machinery relies on -- verified, not assumed.
- CSP-P4-E002: Multi-step open-loop rollout fidelity -- how fast do simulated
  trajectories diverge from ground truth as horizon grows?
- CSP-P4-E003: Does planning with the world model (MPC-style lookahead) actually
  outperform single-step reactive action selection?
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import WorldConfig, GridWorld
from environment.actions import Action
from environment.entities import Entity
from models.mlp import PredictiveMLP
from learning.collector import TrajectoryCollector
from learning.trainer import PredictiveTrainer
from evaluation.metrics import compute_all_metrics
from evaluation.world_model_rollout import simulate_rollout_open_loop, plan_action

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _load_world(name: str = "phase4_world.json") -> WorldConfig:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return WorldConfig.from_dict(json.load(f))


class NaivePhysicsBaseline:
    """Predicts the next state as if the chosen action always succeeds with no hidden
    rules at all: position moves by the action's nominal delta, state always drops by
    exactly 1 (nominal step_penalty). Cannot represent collisions or interactions.
    This is the scientifically meaningful baseline for a world model: it isolates
    exactly what the model had to *learn* (collision and interaction rules) from what
    it could get for free (which raw action was taken).
    """
    _DELTAS = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (0, 0), 5: (0, 0)}  # UP,DOWN,LEFT,RIGHT,INTERACT,NOOP

    def predict(self, X: np.ndarray) -> np.ndarray:
        action_codes = np.argmax(X[:, -6:], axis=1)
        out = np.zeros((X.shape[0], 3))
        for i, code in enumerate(action_codes):
            dx, dy = self._DELTAS[int(code)]
            out[i] = [dx, dy, -1.0]
        return out


def _verify_entities_are_static(world_config: WorldConfig, num_episodes: int = 5, steps: int = 30) -> bool:
    """Explicit check backing the rollout assumption that entity positions never change."""
    for seed in range(num_episodes):
        world = GridWorld(config=world_config)
        world.reset(seed=seed)
        initial_positions = [e.position for e in world._entities]
        for _ in range(steps):
            world.step(Action.NOOP if seed % 2 == 0 else Action.INTERACT)
            current_positions = [e.position for e in world._entities]
            if current_positions != initial_positions:
                return False
    return True


def _train_world_model(world_config: WorldConfig, seed: int, num_steps: int = 6000):
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed)
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)

    X_train, Y_train = train_ds.to_world_model_arrays()
    X_val, Y_val = val_ds.to_world_model_arrays()

    model = PredictiveMLP(input_dim=X_train.shape[1], hidden_dims=(32, 16), output_dim=3, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit((X_train, Y_train), val_data=(X_val, Y_val))
    return model, history, (X_val, Y_val)


def run_experiment_e001(seed: int = 42) -> Dict[str, Any]:
    """CSP-P4-E001: Single-step full-state (position + internal-state) prediction."""
    world_config = _load_world()

    entities_static = _verify_entities_are_static(world_config)

    model, history, (X_val, Y_val) = _train_world_model(world_config, seed)
    model_metrics = compute_all_metrics(model.forward(X_val), Y_val)

    naive = NaivePhysicsBaseline()
    naive_pred = naive.predict(X_val)
    naive_metrics = compute_all_metrics(naive_pred, Y_val)

    # Breakdown by component: position (dx,dy) vs internal-state delta, since they are
    # different kinds of quantities (discrete-ish vs continuous) and a single pooled MSE
    # could hide one being learned much better than the other.
    pos_mse_model = float(np.mean(np.square(model.forward(X_val)[:, :2] - Y_val[:, :2])))
    pos_mse_naive = float(np.mean(np.square(naive_pred[:, :2] - Y_val[:, :2])))
    state_mse_model = float(np.mean(np.square(model.forward(X_val)[:, 2] - Y_val[:, 2])))
    state_mse_naive = float(np.mean(np.square(naive_pred[:, 2] - Y_val[:, 2])))

    return {
        "experiment_id": "CSP-P4-E001",
        "title": "Single-Step Full-State World Model",
        "seed": seed,
        "entities_verified_static": entities_static,
        "overall_metrics": {"world_model": model_metrics, "naive_physics_baseline": naive_metrics},
        "position_delta_mse": {"world_model": pos_mse_model, "naive_physics_baseline": pos_mse_naive},
        "internal_state_delta_mse": {"world_model": state_mse_model, "naive_physics_baseline": state_mse_naive},
        "beats_naive_physics_overall": model_metrics["mse"] < naive_metrics["mse"],
        "beats_naive_physics_on_position": pos_mse_model < pos_mse_naive,
        "beats_naive_physics_on_state": state_mse_model < state_mse_naive,
        "training_history": {"epochs_run": history["epochs_run"], "final_train_loss": history["final_train_loss"]},
    }


def run_experiment_e002(seed: int = 42, num_trajectories: int = 40, horizon: int = 15) -> Dict[str, Any]:
    """CSP-P4-E002: Multi-step open-loop rollout fidelity -- error vs. horizon step."""
    world_config = _load_world()
    model, _, _ = _train_world_model(world_config, seed)

    rng = np.random.RandomState(seed)
    action_space = list(Action)

    pos_errors_by_h: Dict[int, List[float]] = {h: [] for h in range(1, horizon + 1)}
    state_errors_by_h: Dict[int, List[float]] = {h: [] for h in range(1, horizon + 1)}

    for traj_idx in range(num_trajectories):
        world = GridWorld(config=world_config)
        obs = world.reset(seed=seed * 1000 + traj_idx)
        entities = list(world._entities)
        actions = [action_space[rng.randint(0, len(action_space))] for _ in range(horizon)]

        sim_traj = simulate_rollout_open_loop(
            model, obs, actions, entities,
            grid_width=world_config.grid_width, grid_height=world_config.grid_height,
            min_state=world_config.min_internal_state, max_state=world_config.max_internal_state,
        )

        true_traj = [obs]
        for a in actions:
            o, _, _, _ = world.step(a)
            true_traj.append(o)

        for h in range(1, horizon + 1):
            sim_o, true_o = sim_traj[h], true_traj[h]
            pos_err = abs(sim_o.agent_position[0] - true_o.agent_position[0]) + abs(sim_o.agent_position[1] - true_o.agent_position[1])
            state_err = abs(sim_o.internal_state - true_o.internal_state)
            pos_errors_by_h[h].append(pos_err)
            state_errors_by_h[h].append(state_err)

    horizon_curve = {
        h: {
            "mean_position_error": round(float(np.mean(pos_errors_by_h[h])), 4),
            "mean_state_error": round(float(np.mean(state_errors_by_h[h])), 4),
            "exact_position_match_rate": round(float(np.mean([e == 0 for e in pos_errors_by_h[h]])), 4),
        }
        for h in range(1, horizon + 1)
    }

    return {
        "experiment_id": "CSP-P4-E002",
        "title": "Multi-Step Open-Loop Rollout Fidelity",
        "seed": seed,
        "num_trajectories": num_trajectories,
        "horizon": horizon,
        "horizon_curve": horizon_curve,
        "position_error_at_h1": horizon_curve[1]["mean_position_error"],
        "position_error_at_final_h": horizon_curve[horizon]["mean_position_error"],
        "state_error_at_h1": horizon_curve[1]["mean_state_error"],
        "state_error_at_final_h": horizon_curve[horizon]["mean_state_error"],
    }


def run_experiment_e003(seed: int = 42, num_episodes: int = 20, episode_steps: int = 40, planning_horizon: int = 2) -> Dict[str, Any]:
    """CSP-P4-E003: Does world-model-based planning (MPC lookahead) outperform
    single-step reactive action selection using the same trained model?
    """
    world_config = _load_world()
    model, _, _ = _train_world_model(world_config, seed)

    def run_policy(policy_fn, tag_seed_offset: int) -> List[float]:
        totals = []
        for ep in range(num_episodes):
            world = GridWorld(config=world_config)
            obs = world.reset(seed=seed * 500 + tag_seed_offset + ep)
            entities = list(world._entities)
            total_internal_state = obs.internal_state
            for _ in range(episode_steps):
                action = policy_fn(obs, entities, world_config)
                obs, delta, done, info = world.step(action)
                total_internal_state = obs.internal_state
                if done:
                    break
            totals.append(total_internal_state)
        return totals

    def reactive_policy(obs, entities, wc):
        # Single-step greedy: evaluate every action with the SAME world model, no lookahead.
        best_action, best_val = None, -np.inf
        for act in Action:
            from learning.dataset import encode_state_action
            feat = encode_state_action(obs, act)
            pred = model.forward(feat[None, :])[0]
            val = pred[2]  # predicted immediate internal-state delta
            if val > best_val:
                best_val, best_action = val, act
        return best_action

    def planning_policy(obs, entities, wc):
        return plan_action(
            model, obs, entities, grid_width=wc.grid_width, grid_height=wc.grid_height,
            min_state=wc.min_internal_state, max_state=wc.max_internal_state, horizon=planning_horizon,
        )

    def random_policy(obs, entities, wc):
        return list(Action)[np.random.randint(0, len(Action))]

    random_totals = run_policy(random_policy, 0)
    reactive_totals = run_policy(reactive_policy, 10000)
    planning_totals = run_policy(planning_policy, 20000)

    return {
        "experiment_id": "CSP-P4-E003",
        "title": "Does World-Model Planning Outperform Reactive Action Selection?",
        "seed": seed,
        "num_episodes": num_episodes,
        "episode_steps": episode_steps,
        "planning_horizon": planning_horizon,
        "mean_final_internal_state": {
            "random_policy": round(float(np.mean(random_totals)), 3),
            "reactive_single_step": round(float(np.mean(reactive_totals)), 3),
            "mpc_planning": round(float(np.mean(planning_totals)), 3),
        },
        "std_final_internal_state": {
            "random_policy": round(float(np.std(random_totals)), 3),
            "reactive_single_step": round(float(np.std(reactive_totals)), 3),
            "mpc_planning": round(float(np.std(planning_totals)), 3),
        },
        "planning_beats_reactive": float(np.mean(planning_totals)) > float(np.mean(reactive_totals)),
        "planning_beats_random": float(np.mean(planning_totals)) > float(np.mean(random_totals)),
    }


def run_all_phase4_experiments(output_dir: str = "logs") -> Dict[str, Any]:
    """Execute all Phase 4 experiments sequentially and persist results."""
    import os

    print("Running CSP-P4-E001 (Single-Step World Model)...")
    e001 = run_experiment_e001()

    print("Running CSP-P4-E002 (Rollout Fidelity)...")
    e002 = run_experiment_e002()

    print("Running CSP-P4-E003 (Planning vs. Reactive)...")
    e003 = run_experiment_e003()

    all_results = {
        "phase": "Phase 4 — World Model",
        "experiments": {
            "CSP-P4-E001": e001,
            "CSP-P4-E002": e002,
            "CSP-P4-E003": e003,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase4_all_experiments_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Phase 4 experiments completed and saved to: {out_file}")
    return all_results


if __name__ == "__main__":
    run_all_phase4_experiments()
