"""Experimental Suite for Project Caspian (Phase 9 -- Scale Study).

Section 37: "Scaling is an experiment, not a default strategy." The question is which
capabilities emerge, and when, as environmental complexity and model capacity grow.

CSP-P9-E001 sweeps a full factorial grid, 5 seeds per cell:
  model size        hidden widths (4,), (8,), (16,8), (32,16), (64,32), (128,64)  [parameter counts logged]
  training data     500 / 2000 / 8000 environment steps (passive exploration)
  world complexity  C1: 1 entity, 2 types | C2: 3 entities, 4 types | C3: 5 entities, 8 types
Two capabilities are scored on independent test sets, interaction events only:
  in-distribution prediction (what do entities do, at positions seen in training)
  unseen-position generalization (entities placed only in a held-out 2x2 corner block that
  never held an entity during training -- the Phase 3 position-generalization gap)

Type codes are opaque scalars (index-like), so more types genuinely require more capacity to
tabulate; this is deliberate: it is the capacity-vs-complexity probe.

PRE-REGISTERED hypotheses (written before any run):
  H1 (more data/size helps)  mean in-distribution R2 is non-decreasing (tolerance 0.05) along
      the data axis at the largest model, and along the size axis at the largest data, in all
      three complexity levels.
  H2 (data beats size for generalization)  from smallest to largest, the shrinkage of the
      in-dist minus unseen-position gap is larger along the data axis than along the size axis,
      in >= 2 of 3 complexity levels.
  H3 (capacity requirement grows with complexity)  at the largest data, the smallest model
      reaching mean in-dist R2 >= 0.9 has non-decreasing parameter count from C1 to C3, with at
      least one strict increase (an unreachable level counts as infinite).
No hypothesis is altered after seeing results; failures are reported.

NOT covered (stated up front, per 'earn complexity'): memory capacity (recurrent models),
action-space complexity, observation richness. They are listed as open scale variables.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import WorldConfig
from environment.entities import Entity
from models.mlp import PredictiveMLP
from learning.collector import TrajectoryCollector
from learning.dataset import encode_state_action
from learning.trainer import PredictiveTrainer
from evaluation.metrics import compute_all_metrics

MODEL_SIZES = {"tiny": (4,), "small": (8,), "medium-s": (16, 8), "medium": (32, 16), "large": (64, 32), "xlarge": (128, 64)}
DATA_STEPS = [500, 2000, 8000]
TYPE_TABLE = {1: 8.0, 2: -8.0, 3: 4.0, 4: -4.0, 5: 6.0, 6: -6.0, 7: 2.0, 8: -2.0}
COMPLEXITY = {"C1": (1, 2), "C2": (3, 4), "C3": (5, 8)}  # (entities per episode, entity types)
HELD_OUT = [(4, 4), (4, 5), (5, 4), (5, 5)]


def make_world(level: str, held_out_only: bool = False) -> WorldConfig:
    n_ent, n_types = COMPLEXITY[level]
    types = list(TYPE_TABLE.items())[:n_types]
    all_cells = [(x, y) for x in range(6) for y in range(6)]
    allowed = HELD_OUT if held_out_only else [c for c in all_cells if c not in HELD_OUT]
    n = min(n_ent, len(allowed))
    # Held-out test worlds start the agent beside the held-out block: a passive walker starting at the
    # origin almost never reaches the far corner (first smoke run: 3 interaction events in 1500 steps,
    # which made unseen-position R2 degenerate at 0.0).
    return WorldConfig(
        grid_width=6, grid_height=6, initial_agent_pos=(3, 3) if held_out_only else (0, 0),
        initial_entities=[Entity(entity_id=i + 1, entity_type=types[0][0], position=allowed[i], hidden_state_delta=types[0][1]) for i in range(n)],
        initial_internal_state=0.0, step_penalty=1.0, min_internal_state=-1000.0, max_internal_state=1000.0,
        max_timesteps=25, interaction_radius=1, terminate_on_depletion=False, seed=42,
        randomize_entity_positions=True, allowed_entity_positions=allowed,
        entity_type_pool=[{"entity_type": t, "hidden_state_delta": d, "is_interactive": True, "is_blocking": False} for t, d in types],
    )


def _collect(cfg: WorldConfig, seed: int, steps: int):
    return TrajectoryCollector(world_config=cfg).collect_exploratory_dataset(num_steps=steps, interaction_prob=0.4, seed=seed)


def _interaction_xy(ds):
    inter = [t for t in ds.transitions if t.info.get("interaction", False)]
    X = np.stack([encode_state_action(t.obs, t.action) for t in inter], axis=0)
    Y = np.stack([t.to_target_vector() for t in inter], axis=0)
    return X, Y


def _train(X, Y, hidden, seed: int, epochs: int = 60) -> PredictiveMLP:
    model = PredictiveMLP(input_dim=X.shape[1], hidden_dims=hidden, output_dim=1, hidden_activation="relu", seed=seed)
    PredictiveTrainer(model=model, batch_size=32, epochs=epochs, early_stopping_patience=None, seed=seed).fit((X, Y))
    return model


def _r2(model, X, Y) -> float:
    return float(compute_all_metrics(model.forward(X), Y)["r2_score"])


def run_experiment_e001(seeds: List[int] = [42, 101, 7, 2026, 555], sizes: Dict[str, tuple] = None,
                        data_steps: List[int] = None, levels: List[str] = None, epochs: int = 60,
                        test_steps: int = 4000, verbose: bool = True) -> Dict[str, Any]:
    sizes = sizes or MODEL_SIZES; data_steps = data_steps or DATA_STEPS; levels = levels or list(COMPLEXITY)
    cells: Dict[str, Any] = {}
    param_counts: Dict[str, int] = {}
    for lv in levels:
        train_cfg, ho_cfg = make_world(lv), make_world(lv, held_out_only=True)
        per_seed = {}
        for s in seeds:
            Xi, Yi = _interaction_xy(_collect(train_cfg, seed=s + 5000, steps=test_steps))
            Xu, Yu = _interaction_xy(_collect(ho_cfg, seed=s + 6000, steps=test_steps))
            for D in data_steps:
                Xd, Yd = _collect(train_cfg, seed=s, steps=D).to_numpy_arrays()
                for name, hidden in sizes.items():
                    m = _train(Xd, Yd, hidden, seed=s, epochs=epochs)
                    param_counts[name] = int(m.num_parameters)
                    per_seed.setdefault((D, name), []).append({"seed": s, "in_dist_r2": round(_r2(m, Xi, Yi), 4), "unseen_pos_r2": round(_r2(m, Xu, Yu), 4)})
            if verbose:
                print("E001", lv, "seed", s, "done", flush=True)
        for (D, name), rows in per_seed.items():
            a = [r["in_dist_r2"] for r in rows]; u = [r["unseen_pos_r2"] for r in rows]
            cells[f"{lv}|{D}|{name}"] = {"level": lv, "data_steps": D, "size": name, "params": param_counts[name],
                                         "in_dist_r2_mean": round(float(np.mean(a)), 4), "in_dist_r2_std": round(float(np.std(a)), 4),
                                         "unseen_pos_r2_mean": round(float(np.mean(u)), 4), "unseen_pos_r2_std": round(float(np.std(u)), 4),
                                         "gap_mean": round(float(np.mean(a) - np.mean(u)), 4), "per_seed": rows}
    return {"experiment_id": "CSP-P9-E001", "title": "Model Size x Training Data x World Complexity", "seeds": seeds,
            "sizes": {k: list(v) for k, v in sizes.items()}, "param_counts": param_counts, "data_steps": data_steps, "levels": levels,
            "epochs": epochs, "cells": cells, "preregistered_checks": evaluate_hypotheses(cells, sizes, data_steps, levels)}


def evaluate_hypotheses(cells, sizes, data_steps, levels) -> Dict[str, Any]:
    names = list(sizes); big, small = names[-1], names[0]; Dmax, Dmin = max(data_steps), min(data_steps)
    g = lambda lv, D, n, k: cells[f"{lv}|{D}|{n}"][k]
    h1, h2_votes, mins = {}, 0, {}
    for lv in levels:
        along_data = [g(lv, D, big, "in_dist_r2_mean") for D in sorted(data_steps)]
        along_size = [g(lv, Dmax, n, "in_dist_r2_mean") for n in names]
        nondec = lambda v: all(v[i + 1] >= v[i] - 0.05 for i in range(len(v) - 1))
        h1[lv] = {"data_axis": along_data, "size_axis": along_size, "data_nondecreasing": nondec(along_data), "size_nondecreasing": nondec(along_size)}
        gap_data = g(lv, Dmin, big, "gap_mean") - g(lv, Dmax, big, "gap_mean")
        gap_size = g(lv, Dmax, small, "gap_mean") - g(lv, Dmax, big, "gap_mean")
        h1[lv].update({"gap_shrink_by_data": round(gap_data, 4), "gap_shrink_by_size": round(gap_size, 4)})
        h2_votes += int(gap_data > gap_size)
        reach = [cells[f"{lv}|{Dmax}|{n}"]["params"] for n in names if g(lv, Dmax, n, "in_dist_r2_mean") >= 0.9]
        mins[lv] = min(reach) if reach else float("inf")
    seq = [mins[lv] for lv in levels]
    h3 = all(seq[i + 1] >= seq[i] for i in range(len(seq) - 1)) and seq[-1] > seq[0]
    return {"H1_met": bool(all(v["data_nondecreasing"] and v["size_nondecreasing"] for v in h1.values())), "H1_detail": h1,
            "H2_votes": h2_votes, "H2_met": bool(h2_votes >= 2),
            "H3_min_params_for_R2_0.9": {k: (v if v != float("inf") else None) for k, v in mins.items()}, "H3_met": bool(h3)}


def run_experiment_e002(seeds: List[int] = [42, 101, 7], epochs: int = 60) -> Dict[str, Any]:
    """CSP-P9-E002: is E001's failure at C2/C3 limited by DATA or by CAPACITY?
    E001 never trained C2/C3 to convergence at <= 8000 steps. Here C2 gets 32000 steps and C3 gets
    32000 and 96000, with models tiny / small / medium / xlarge, 3 seeds. Hypotheses recorded BEFORE
    running, after E001 and informed by it (a follow-up, not a rescue of E001's failed H1):
      H6  C2 reaches mean in-dist R2 >= 0.9 at 32000 steps for medium and xlarge (data-limited).
      H7  C3 reaches mean in-dist R2 >= 0.9 at 96000 steps for medium and xlarge.
      H8  tiny (157 params) stays below 0.9 at C2 and C3 even with ample data (a capacity floor
          that rises with complexity)."""
    sizes = {k: MODEL_SIZES[k] for k in ["tiny", "small", "medium", "xlarge"]}
    plan = [("C2", 32000), ("C3", 32000), ("C3", 96000)]
    out = {}
    for lv, D in plan:
        r = run_experiment_e001(seeds=seeds, sizes=sizes, data_steps=[D], levels=[lv], epochs=epochs, verbose=False)
        out.update(r["cells"])
        print("E002", lv, D, {n: out[f"{lv}|{D}|{n}"]["in_dist_r2_mean"] for n in sizes}, flush=True)
    g = lambda lv, D, n: out[f"{lv}|{D}|{n}"]["in_dist_r2_mean"]
    checks = {"H6_met": bool(g("C2", 32000, "medium") >= 0.9 and g("C2", 32000, "xlarge") >= 0.9),
              "H7_met": bool(g("C3", 96000, "medium") >= 0.9 and g("C3", 96000, "xlarge") >= 0.9),
              "H8_met": bool(g("C2", 32000, "tiny") < 0.9 and g("C3", 96000, "tiny") < 0.9)}
    return {"experiment_id": "CSP-P9-E002", "title": "Data-Limited or Capacity-Limited? Complex Worlds With Ample Data", "seeds": seeds,
            "sizes": {k: list(v) for k, v in sizes.items()}, "plan": plan, "cells": out, "preregistered_checks": checks}


def run_all_phase9_experiments(output_dir: str = "logs") -> Dict[str, Any]:
    import os
    print("Running CSP-P9-E001 ...")
    e001 = run_experiment_e001()
    all_results = {"phase": "Phase 9 -- Scale Study", "experiments": {"CSP-P9-E001": e001}}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "phase9_all_experiments_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    return all_results


if __name__ == "__main__":
    run_all_phase9_experiments()
