"""Baseline Comparison Experiment for Project Caspian (Phase 0).

Evaluates and compares RandomAgent and OracleAgent baselines across multiple
random seeds, recording performance metrics, state changes, and interaction counts.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import GridWorld, WorldConfig
from agent.random_agent import RandomAgent
from agent.oracle_agent import OracleAgent


def evaluate_agent(agent_type: str, seeds: List[int], steps_per_run: int = 50) -> Dict[str, Any]:
    """Evaluate an agent across multiple random seeds.

    Args:
        agent_type: "random" or "oracle".
        seeds: List of random seeds to evaluate.
        steps_per_run: Number of steps to execute per seed.

    Returns:
        Dict[str, Any]: Summary metrics across seeds.
    """
    config_path = Path(__file__).parent.parent / "configs" / "default_world.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    runs = []

    for seed in seeds:
        config_data["seed"] = seed
        config_data["max_timesteps"] = steps_per_run + 10
        config = WorldConfig.from_dict(config_data)
        world = GridWorld(config=config)

        if agent_type == "random":
            agent = RandomAgent(seed=seed)
        elif agent_type == "oracle":
            agent = OracleAgent(interaction_radius=1, seed=seed)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        obs = world.get_observation()
        total_delta = 0.0
        interactions = 0
        collisions = 0
        energy_history = [obs.internal_state]

        for _ in range(steps_per_run):
            action = agent.act(obs)
            obs, delta, done, info = world.step(action)
            agent.observe(obs, delta, done, info)

            total_delta += delta
            if info.get("interaction", False):
                interactions += 1
            if info.get("collision", False):
                collisions += 1
            energy_history.append(obs.internal_state)

            if done:
                break

        runs.append({
            "seed": seed,
            "steps_executed": len(energy_history) - 1,
            "final_energy": obs.internal_state,
            "average_energy": round(sum(energy_history) / len(energy_history), 2),
            "cumulative_delta": round(total_delta, 2),
            "interactions": interactions,
            "collisions": collisions,
        })

    avg_final = sum(r["final_energy"] for r in runs) / len(runs)
    avg_interactions = sum(r["interactions"] for r in runs) / len(runs)
    avg_collisions = sum(r["collisions"] for r in runs) / len(runs)
    avg_cumulative_delta = sum(r["cumulative_delta"] for r in runs) / len(runs)

    return {
        "agent_type": agent_type,
        "runs": runs,
        "summary": {
            "avg_final_energy": round(avg_final, 2),
            "avg_interactions": round(avg_interactions, 2),
            "avg_collisions": round(avg_collisions, 2),
            "avg_cumulative_delta": round(avg_cumulative_delta, 2),
        },
    }


def run_experiment(output_dir: str = "logs") -> Dict[str, Any]:
    """Execute complete baseline comparison experiment."""
    seeds = [42, 101, 2026, 777, 999]
    steps_per_run = 50

    print("Running RandomAgent baseline...")
    random_results = evaluate_agent("random", seeds=seeds, steps_per_run=steps_per_run)

    print("Running OracleAgent baseline...")
    oracle_results = evaluate_agent("oracle", seeds=seeds, steps_per_run=steps_per_run)

    experiment_data = {
        "seeds_evaluated": seeds,
        "steps_per_run": steps_per_run,
        "random_baseline": random_results,
        "oracle_baseline": oracle_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "baseline_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)

    return experiment_data


if __name__ == "__main__":
    results = run_experiment()
    rand_sum = results["random_baseline"]["summary"]
    ora_sum = results["oracle_baseline"]["summary"]

    print("\n" + "=" * 60)
    print("PROJECT CASPIAN — PHASE 0 BASELINE COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Random Agent':<15} | {'Oracle Agent':<15}")
    print("-" * 60)
    print(f"{'Avg Final Energy':<25} | {rand_sum['avg_final_energy']:<15} | {ora_sum['avg_final_energy']:<15}")
    print(f"{'Avg Cumulative Delta':<25} | {rand_sum['avg_cumulative_delta']:<15} | {ora_sum['avg_cumulative_delta']:<15}")
    print(f"{'Avg Interactions':<25} | {rand_sum['avg_interactions']:<15} | {ora_sum['avg_interactions']:<15}")
    print(f"{'Avg Collisions':<25} | {rand_sum['avg_collisions']:<15} | {ora_sum['avg_collisions']:<15}")
    print("=" * 60)
    print(f"Full benchmark saved to: logs/baseline_comparison.json\n")
