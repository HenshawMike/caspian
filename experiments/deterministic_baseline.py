"""Deterministic Environment Baseline Experiment for Project Caspian (Phase 0).

Executes a predefined action sequence under fixed seed across multiple resets,
verifying bitwise reproducibility of observations and state transitions, and
saving trajectory recordings.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import GridWorld, WorldConfig
from environment.actions import Action


def run_experiment(
    seed: int = 42,
    num_resets: int = 5,
    output_dir: str = "logs",
) -> Dict[str, Any]:
    """Run the deterministic baseline experiment.

    Args:
        seed: Random seed for initialization.
        num_resets: Number of sequential resets to execute and compare.
        output_dir: Directory path to save trajectory recordings.

    Returns:
        Dict[str, Any]: Summary results of the experiment.
    """
    config_path = Path(__file__).parent.parent / "configs" / "default_world.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    config_data["seed"] = seed
    config = WorldConfig.from_dict(config_data)

    action_sequence = [
        Action.RIGHT,
        Action.UP,
        Action.RIGHT,
        Action.UP,
        Action.INTERACT,
        Action.UP,        # Move to (2, 3)
        Action.INTERACT,  # Interact within radius 1 of entity (2, 2)
        Action.UP,        # Move to (2, 4)
        Action.UP,        # Attempt boundary collision at top
        Action.LEFT,      # Move to (1, 4)
        Action.LEFT,      # Move to (0, 4)
        Action.LEFT,      # Attempt boundary collision at left
        Action.DOWN,      # Move to (0, 3)
        Action.NOOP,      # Idle
        Action.DOWN,      # Move to (0, 2)
    ]

    world = GridWorld(config=config)
    run_hashes: List[str] = []
    trajectories: List[List[Dict[str, Any]]] = []

    for run_idx in range(num_resets):
        world.reset(seed=seed)
        trajectory: List[Dict[str, Any]] = []

        # Initial observation at t=0
        init_obs = world.get_observation()
        trajectory.append({
            "step": 0,
            "action_executed": None,
            "observation": init_obs.to_dict(),
            "state_delta": 0.0,
            "terminated": False,
            "info": {},
        })

        for step_idx, act in enumerate(action_sequence, start=1):
            obs, delta, done, info = world.step(act)
            trajectory.append({
                "step": step_idx,
                "action_executed": act.name,
                "observation": obs.to_dict(),
                "state_delta": delta,
                "terminated": done,
                "info": info,
            })

        # Compute deterministic trajectory hash
        traj_json = json.dumps(trajectory, sort_keys=True)
        traj_hash = hashlib.sha256(traj_json.encode("utf-8")).hexdigest()
        run_hashes.append(traj_hash)
        trajectories.append(trajectory)

    # Verification: all runs must have identical hashes
    is_deterministic = len(set(run_hashes)) == 1
    primary_hash = run_hashes[0]

    # Save trajectory to output directory
    os.makedirs(output_dir, exist_ok=True)
    traj_file = os.path.join(output_dir, f"trajectory_seed_{seed}.json")
    with open(traj_file, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "num_resets": num_resets,
            "deterministic_match": is_deterministic,
            "sha256_hash": primary_hash,
            "action_count": len(action_sequence),
            "trajectory": trajectories[0],
        }, f, indent=2)

    return {
        "seed": seed,
        "num_resets": num_resets,
        "is_deterministic": is_deterministic,
        "sha256_hash": primary_hash,
        "action_steps": len(action_sequence),
        "trajectory_file": traj_file,
    }


if __name__ == "__main__":
    result = run_experiment(seed=42, num_resets=5)
    print("==================================================")
    print("PROJECT CASPIAN — PHASE 0 BASELINE EXPERIMENT")
    print("==================================================")
    print(f"Random Seed:          {result['seed']}")
    print(f"Resets Evaluated:     {result['num_resets']}")
    print(f"Action Steps / Run:   {result['action_steps']}")
    print(f"Trajectory SHA-256:   {result['sha256_hash']}")
    print(f"Deterministic Match:  {'PASSED' if result['is_deterministic'] else 'FAILED'}")
    print(f"Saved Trajectory:     {result['trajectory_file']}")
    print("==================================================")

    if not result["is_deterministic"]:
        sys.exit(1)
