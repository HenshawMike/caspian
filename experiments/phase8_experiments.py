"""Experimental Suite for Project Caspian (Phase 8 -- Novel Worlds).

Section 36 of the Development Plan: introduce worlds Caspian has never seen, changing
geometry, entity appearance, dynamics, environmental rules, resource distributions and
action consequences, to test whether learned representations transfer.

Design (one source world, six novel worlds, one null world; 5 seeds; all worlds use
UNCLAMPED, linear internal_state so no ceiling confound; all test sets are independent):

  source      5x5, one entity per episode, types {1:+10, 2:-10}, step penalty 1, radius 1
  identical   the source world again (null: zero-shot should match in-distribution)
  geometry    8x8 grid
  appearance  type codes {7:+10, 8:-10} (same functional rule, never-seen codes)
  dynamics    step penalty 3 and interaction radius 2
  rules       effect magnitudes {1:+4, 2:-16} (same signs, new sizes)
  resources   3 entities per episode instead of 1
  actions     every INTERACT costs an extra 2 (action consequence changes)

Two measurements per novel world:
  ZERO-SHOT   the source-trained model scored with no adaptation (R2 on all transitions
              of an independent test set from the novel world).
  TRANSFER    adaptation efficiency: fine-tune the source model vs train from scratch vs a
              CONTROL that is fine-tuned from a model pretrained on SHUFFLED targets (same
              warm start, no knowledge), at several novel-world budgets.

PRE-REGISTERED criteria (written before any run):
  H1 sanity   source in-distribution R2 >= 0.9 and 'identical' zero-shot within 0.1 of it.
  H2 novelty  zero-shot R2 is at least 0.3 below in-distribution on >= 3 of the 6 novel
              worlds (otherwise the worlds are not meaningfully novel).
  H3 transfer on a given world, transfer is SUPPORTED if, at >= 2 of the 4 budgets, the
              fine-tuned model beats scratch by >= 0.1 mean R2 AND beats the shuffled-
              pretrain control by >= 0.05. Overall Phase 8 is judged by the number of
              worlds supported, reported per world; no world is dropped after seeing results.
"""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import WorldConfig
from environment.entities import Entity
from models.mlp import PredictiveMLP
from learning.dataset import ExperienceDataset
from learning.collector import TrajectoryCollector
from learning.trainer import PredictiveTrainer
from models.optimizers import Adam
from evaluation.metrics import compute_all_metrics

WORLD_NAMES = ["identical", "geometry", "appearance", "dynamics", "rules", "resources", "actions"]


def _pool(types: Dict[int, float]) -> List[Dict[str, Any]]:
    return [{"entity_type": t, "hidden_state_delta": d, "is_interactive": True, "is_blocking": False} for t, d in types.items()]


def make_world(name: str) -> WorldConfig:
    """Build the source world or one named novel world (all unclamped / linear)."""
    cfg = WorldConfig(
        grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
        initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2), hidden_state_delta=10.0)],
        initial_internal_state=0.0, step_penalty=1.0, min_internal_state=-1000.0, max_internal_state=1000.0,
        max_timesteps=25, interaction_radius=1, terminate_on_depletion=False, seed=42,
        randomize_entity_positions=True, entity_type_pool=_pool({1: 10.0, 2: -10.0}),
    )
    if name in ("source", "identical"):
        return cfg
    if name == "geometry":
        cfg.grid_width, cfg.grid_height = 8, 8
        cfg.max_timesteps = 40
    elif name == "appearance":
        cfg.entity_type_pool = _pool({7: 10.0, 8: -10.0})
        cfg.initial_entities = [Entity(entity_id=1, entity_type=7, position=(2, 2), hidden_state_delta=10.0)]
    elif name == "dynamics":
        cfg.step_penalty = 3.0
        cfg.interaction_radius = 2
    elif name == "rules":
        cfg.entity_type_pool = _pool({1: 4.0, 2: -16.0})
        cfg.initial_entities = [Entity(entity_id=1, entity_type=1, position=(2, 2), hidden_state_delta=4.0)]
    elif name == "resources":
        cfg.initial_entities = [Entity(entity_id=i + 1, entity_type=1, position=p, hidden_state_delta=10.0) for i, p in enumerate([(2, 2), (3, 1), (1, 3)])]
    elif name == "actions":
        cfg.interact_cost = 2.0
    else:
        raise ValueError(name)
    return cfg


def _collect(cfg: WorldConfig, seed: int, steps: int) -> ExperienceDataset:
    return TrajectoryCollector(world_config=cfg).collect_exploratory_dataset(num_steps=steps, interaction_prob=0.4, seed=seed)


def _train(X, Y, seed: int, epochs: int = 100) -> PredictiveMLP:
    model = PredictiveMLP(input_dim=X.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    PredictiveTrainer(model=model, batch_size=32, epochs=epochs, early_stopping_patience=None, seed=seed).fit((X, Y))
    return model


def _finetune(src: PredictiveMLP, X, Y, seed: int, epochs: int = 60, lr: float = 0.003) -> PredictiveMLP:
    """Fine-tune a COPY of `src`, keeping the source input normalizer (refitting it on a tiny
    sample would change the input scaling the pretrained weights depend on)."""
    m = PredictiveMLP.from_dict(src.to_dict())
    m.fit_normalizer = lambda X_: None  # trainer calls this; keep the source normalizer
    trainer = PredictiveTrainer(model=m, optimizer=Adam(params=m.get_params(), lr=lr, weight_decay=1e-4),
                                batch_size=32, epochs=epochs, early_stopping_patience=None, seed=seed)
    trainer.fit((X, Y))
    return m


def _r2(model: PredictiveMLP, X, Y) -> float:
    return float(compute_all_metrics(model.forward(X), Y)["r2_score"])


def run_experiment_e001(seeds: List[int] = [42, 101, 7, 2026, 555], source_steps: int = 3000,
                        budgets: List[int] = [50, 150, 500, 1500], test_steps: int = 3000) -> Dict[str, Any]:
    """CSP-P8-E001: zero-shot behavior and few-shot transfer across six novel worlds."""
    src_cfg = make_world("source")
    rows = []
    for s in seeds:
        src_ds = _collect(src_cfg, seed=s, steps=source_steps)
        Xs, Ys = src_ds.to_numpy_arrays()
        src_model = _train(Xs, Ys, seed=s)
        rng = np.random.default_rng(s)
        ctrl_model = _train(Xs, rng.permutation(Ys), seed=s)  # warm start carrying no knowledge
        Xt, Yt = _collect(src_cfg, seed=s + 5000, steps=test_steps).to_numpy_arrays()
        row = {"seed": s, "source_in_dist_r2": round(_r2(src_model, Xt, Yt), 4), "worlds": {}}
        for w in WORLD_NAMES:
            cfg = make_world(w)
            Xw, Yw = _collect(cfg, seed=s + 6000, steps=test_steps).to_numpy_arrays()
            entry = {"zero_shot_r2": round(_r2(src_model, Xw, Yw), 4), "budgets": {}}
            for B in budgets:
                Xb, Yb = _collect(cfg, seed=s + 100 + B, steps=B).to_numpy_arrays()
                entry["budgets"][str(B)] = {
                    "scratch": round(_r2(_train(Xb, Yb, seed=s), Xw, Yw), 4),
                    "finetune": round(_r2(_finetune(src_model, Xb, Yb, seed=s), Xw, Yw), 4),
                    "control_shuffled_pretrain": round(_r2(_finetune(ctrl_model, Xb, Yb, seed=s), Xw, Yw), 4),
                }
            row["worlds"][w] = entry
        rows.append(row)
        print("E001 seed", s, "in-dist", row["source_in_dist_r2"], {w: row["worlds"][w]["zero_shot_r2"] for w in WORLD_NAMES}, flush=True)

    in_dist = float(np.mean([r["source_in_dist_r2"] for r in rows]))
    summary = {}
    for w in WORLD_NAMES:
        zs = float(np.mean([r["worlds"][w]["zero_shot_r2"] for r in rows]))
        per_b = {}
        supported_budgets = 0
        for B in budgets:
            g = lambda k: float(np.mean([r["worlds"][w]["budgets"][str(B)][k] for r in rows]))
            sc, ft, ct = g("scratch"), g("finetune"), g("control_shuffled_pretrain")
            ok = (ft - sc >= 0.1) and (ft - ct >= 0.05)
            supported_budgets += int(ok)
            per_b[str(B)] = {"scratch": round(sc, 4), "finetune": round(ft, 4), "control": round(ct, 4),
                             "gain_vs_scratch": round(ft - sc, 4), "gain_vs_control": round(ft - ct, 4), "criterion_met": bool(ok)}
        summary[w] = {"zero_shot_r2_mean": round(zs, 4), "zero_shot_drop": round(in_dist - zs, 4),
                      "budgets": per_b, "transfer_supported": bool(supported_budgets >= 2), "budgets_meeting_criterion": supported_budgets}
    novel = [w for w in WORLD_NAMES if w != "identical"]
    checks = {
        "H1_source_in_dist_ge_0.9": bool(in_dist >= 0.9),
        "H1_identical_within_0.1": bool(abs(summary["identical"]["zero_shot_drop"]) <= 0.1),
        "H2_novelty_worlds_with_drop_ge_0.3": sum(summary[w]["zero_shot_drop"] >= 0.3 for w in novel),
        "H2_met": bool(sum(summary[w]["zero_shot_drop"] >= 0.3 for w in novel) >= 3),
        "H3_worlds_supported": [w for w in novel if summary[w]["transfer_supported"]],
        "H3_count": sum(summary[w]["transfer_supported"] for w in novel),
    }
    return {"experiment_id": "CSP-P8-E001", "title": "Zero-Shot and Few-Shot Transfer Across Novel Worlds",
            "seeds": seeds, "source_steps": source_steps, "budgets": budgets, "source_in_dist_r2_mean": round(in_dist, 4),
            "per_seed": rows, "summary": summary, "preregistered_checks": checks}


# ---------------------------------------------------------------------------
# CSP-P8-E002: does Phase 5's continuous-property channel repair the appearance failure?
# ---------------------------------------------------------------------------
# E001 found the one clear Phase 8 failure: never-seen type codes (appearance) gave strongly
# NEGATIVE zero-shot R2 and negative transfer, consistent with Phase 5 (index-like codes
# cannot generalize). Phase 5 E007 found a continuous `properties` channel does. E002
# tests that remedy in the Phase 8 setting. Hypothesis recorded BEFORE running:
#   H4  with the type code dropped and a 1-D property channel used instead, zero-shot R2 on
#       'appearance_prop_same' (new codes, same property values) is >= 0.8 in >= 4/5 seeds;
#   H5  fine-tuning from the source beats scratch by >= 0.1 at >= 2 of 4 budgets there, and
#       the same holds (reported, not required) for 'appearance_prop_new' (new codes AND new
#       property values 0.8/0.2).
# Limitation stated up front: properties are informative about the outcome by construction.


def make_prop_world(name: str) -> WorldConfig:
    cfg = make_world("source")
    tpl = lambda t, d, p: {"entity_type": t, "hidden_state_delta": d, "is_interactive": True, "is_blocking": False, "properties": [p]}
    if name == "source_prop":
        cfg.entity_type_pool = [tpl(1, 10.0, 0.9), tpl(2, -10.0, 0.1)]
    elif name == "appearance_prop_same":
        cfg.entity_type_pool = [tpl(7, 10.0, 0.9), tpl(8, -10.0, 0.1)]
    elif name == "appearance_prop_new":
        cfg.entity_type_pool = [tpl(7, 10.0, 0.8), tpl(8, -10.0, 0.2)]
    else:
        raise ValueError(name)
    return cfg


def _collect_prop(cfg: WorldConfig, seed: int, steps: int):
    from experiments.phase5_experiments import _dataset_to_property_arrays
    return _dataset_to_property_arrays(_collect(cfg, seed=seed, steps=steps))


def run_experiment_e002(seeds: List[int] = [42, 101, 7, 2026, 555], source_steps: int = 3000,
                        budgets: List[int] = [50, 150, 500, 1500], test_steps: int = 3000) -> Dict[str, Any]:
    """CSP-P8-E002: appearance transfer with a continuous property channel."""
    novel = ["appearance_prop_same", "appearance_prop_new"]
    rows = []
    for s in seeds:
        Xs, Ys = _collect_prop(make_prop_world("source_prop"), seed=s, steps=source_steps)
        src = _train(Xs, Ys, seed=s)
        ctrl = _train(Xs, np.random.default_rng(s).permutation(Ys), seed=s)
        Xt, Yt = _collect_prop(make_prop_world("source_prop"), seed=s + 5000, steps=test_steps)
        row = {"seed": s, "source_in_dist_r2": round(_r2(src, Xt, Yt), 4), "worlds": {}}
        for w in novel:
            cfg = make_prop_world(w)
            Xw, Yw = _collect_prop(cfg, seed=s + 6000, steps=test_steps)
            entry = {"zero_shot_r2": round(_r2(src, Xw, Yw), 4), "budgets": {}}
            for B in budgets:
                Xb, Yb = _collect_prop(cfg, seed=s + 100 + B, steps=B)
                entry["budgets"][str(B)] = {"scratch": round(_r2(_train(Xb, Yb, seed=s), Xw, Yw), 4),
                                            "finetune": round(_r2(_finetune(src, Xb, Yb, seed=s), Xw, Yw), 4),
                                            "control_shuffled_pretrain": round(_r2(_finetune(ctrl, Xb, Yb, seed=s), Xw, Yw), 4)}
            row["worlds"][w] = entry
        rows.append(row)
        print("E002 seed", s, row["source_in_dist_r2"], {w: row["worlds"][w]["zero_shot_r2"] for w in novel}, flush=True)
    summary = {}
    for w in novel:
        zs = [r["worlds"][w]["zero_shot_r2"] for r in rows]
        per_b, ok_b = {}, 0
        for B in budgets:
            g = lambda k: float(np.mean([r["worlds"][w]["budgets"][str(B)][k] for r in rows]))
            sc, ft, ct = g("scratch"), g("finetune"), g("control_shuffled_pretrain")
            ok = (ft - sc >= 0.1) and (ft - ct >= 0.05); ok_b += int(ok)
            per_b[str(B)] = {"scratch": round(sc, 4), "finetune": round(ft, 4), "control": round(ct, 4), "criterion_met": bool(ok)}
        summary[w] = {"zero_shot_r2_mean": round(float(np.mean(zs)), 4), "zero_shot_r2_min": round(float(np.min(zs)), 4),
                      "seeds_zero_shot_ge_0.8": sum(z >= 0.8 for z in zs), "budgets": per_b, "transfer_supported": bool(ok_b >= 2)}
    checks = {"H4_met": bool(summary["appearance_prop_same"]["seeds_zero_shot_ge_0.8"] >= 4),
              "H5_same_met": summary["appearance_prop_same"]["transfer_supported"],
              "H5_new_met_reported": summary["appearance_prop_new"]["transfer_supported"]}
    return {"experiment_id": "CSP-P8-E002", "title": "Appearance Transfer With a Continuous Property Channel", "seeds": seeds,
            "budgets": budgets, "source_in_dist_r2_mean": round(float(np.mean([r["source_in_dist_r2"] for r in rows])), 4),
            "per_seed": rows, "summary": summary, "preregistered_checks": checks}


def run_all_phase8_experiments(output_dir: str = "logs") -> Dict[str, Any]:
    import os
    print("Running CSP-P8-E001 ...")
    e001 = run_experiment_e001()
    all_results = {"phase": "Phase 8 -- Novel Worlds", "experiments": {"CSP-P8-E001": e001}}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "phase8_all_experiments_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    return all_results


if __name__ == "__main__":
    run_all_phase8_experiments()
