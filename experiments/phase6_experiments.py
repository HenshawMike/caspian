"""Experimental Suite for Project Caspian (Phase 6 — Rule Change & Adaptation).

Addresses Section 34 of the Development Plan: change the world's rules mid-stream
(an entity that used to increase internal_state now decreases it) and ask whether
Caspian can detect that its existing model is wrong and adapt -- measuring detection
time, adaptation speed, catastrophic forgetting, transfer, old-model persistence, and
new-model learning, exactly as Section 34 lists.

This phase is a genuine architectural departure from Phases 1-5: every prior phase
trained a model offline (collect a dataset, fit to convergence, evaluate). Rule
change detection is meaningless for an offline-only agent -- there's no "during" to
measure. So Phase 6 earns one new piece of machinery, learning/online_trainer.py's
OnlineAdaptationTrainer: a streaming trainer that updates on a bounded sliding-window
replay buffer after every single transition, and reports each transition's PRE-update
error (the "surprise" signal detection/adaptation are measured from). This is the
simplest mechanism that could possibly answer the phase's question; nothing fancier
(elastic weight consolidation, explicit regime detection, meta-learned adaptation
rates) is assumed up front, per Section 41 ("never increase complexity simply
because the current experiment failed").

Design, shared across this suite:
- A 4-type world (configs/phase6_world_base.json, identical in structure to Phase 5's
  v2 world: codes 1/3 = +10, codes 2/4 = -10), used both as the PRE-change world and,
  with one type's hidden_state_delta sign flipped (mirroring Phase 5 E006's
  intervention design exactly), as the POST-change world.
- CSP-P6-E001 (Offline Baseline): train a model to convergence on the pre-change world
  with the existing offline PredictiveTrainer -- this is "Caspian's existing model"
  that Phase 6 is about to invalidate.
- CSP-P6-E002 (Detection & Adaptation): hand that pre-trained model to an
  OnlineAdaptationTrainer, stream interactions from the pre-change world for a while
  (confirming the online trainer agrees the old model is fine), then switch to the
  post-change world mid-stream and measure detection time and adaptation time on the
  entity whose rule changed.
- CSP-P6-E003 (Catastrophic Forgetting): does adapting to the changed type damage the
  model's accuracy on the OTHER, unchanged types?
- CSP-P6-E004 (Old-Model Persistence Control): a frozen checkpoint of the pre-change
  model, taken before any online adaptation, is never touched by the online updates
  and still predicts the OLD rule when evaluated against post-change data -- the
  control that proves the online model's improvement is real weight adaptation, not
  an artifact of shared mutable state (same logic as Phase 5 E006's cosmetic control).
- CSP-P6-E005 (Transfer): after the model has already adapted once (type 2's rule
  change), a SECOND, independent type's rule is flipped. Does detection/adaptation
  happen faster the second time (evidence the agent "learned how to adapt"), or
  equally slowly (evidence each rule change is independent, no meta-adaptation)?
"""

import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from models.mlp import PredictiveMLP
from models.optimizers import Adam
from learning.dataset import encode_state_action
from learning.collector import TrajectoryCollector
from learning.trainer import PredictiveTrainer
from learning.online_trainer import OnlineAdaptationTrainer, detect_and_measure_adaptation
from evaluation.metrics import compute_all_metrics

CONFIG_DIR = Path(__file__).parent.parent / "configs"
MOVEMENT_ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.NOOP]


def _load_world(name: str) -> WorldConfig:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return WorldConfig.from_dict(json.load(f))


def _flip_type(base: WorldConfig, flip_type: int) -> WorldConfig:
    """Same world, except `flip_type`'s hidden_state_delta sign is reversed -- the
    actual environmental "rule change" (Section 34's example: X used to increase
    energy, now it decreases it). Mirrors Phase 5 E006's intervention construction.
    """
    out = copy.deepcopy(base)
    new_pool = []
    for template in out.entity_type_pool:
        t = dict(template)
        if int(t["entity_type"]) == flip_type:
            t["hidden_state_delta"] = -float(t["hidden_state_delta"])
        new_pool.append(t)
    out.entity_type_pool = new_pool
    return out


def _train_offline_baseline(world_config: WorldConfig, seed: int, num_steps: int = 6000) -> Tuple[PredictiveMLP, Dict[str, Any]]:
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed)
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)
    X_train, Y_train = train_ds.to_numpy_arrays()
    X_val, Y_val = val_ds.to_numpy_arrays()

    model = PredictiveMLP(input_dim=X_train.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit((X_train, Y_train), val_data=(X_val, Y_val))
    val_metrics = compute_all_metrics(model.forward(X_val), Y_val)
    return model, {"epochs_run": history["epochs_run"], "final_train_loss": history["final_train_loss"], "val_metrics": val_metrics}


def run_experiment_e001(seed: int = 42) -> Dict[str, Any]:
    """CSP-P6-E001: Offline Baseline -- train Caspian's "existing model" on the
    stationary (pre-change) world, to convergence, the normal offline way. This is
    the model whose correctness Phase 6 is about to invalidate.
    """
    pre_world = _load_world("phase6_world_base.json")
    model, info = _train_offline_baseline(pre_world, seed)
    return {
        "experiment_id": "CSP-P6-E001",
        "title": "Offline Baseline: Caspian's Existing Model Before Any Rule Change",
        "seed": seed,
        "num_parameters": model.num_parameters,
        **info,
    }


def _run_online_stream(
    model: PredictiveMLP,
    world_pre: WorldConfig,
    world_post: WorldConfig,
    pre_steps: int,
    post_steps: int,
    seed: int,
    buffer_size: int = 32,
    lr: float = 0.01,
    interaction_prob: float = 0.4,
) -> Tuple[PredictiveMLP, List[Dict[str, Any]]]:
    """Run the online adaptation loop: `pre_steps` env steps on `world_pre` (confirms
    the online trainer's error is already low -- the model agrees the old rule holds),
    then switch to `world_post` for `post_steps` more env steps, all on the SAME
    OnlineAdaptationTrainer instance (same buffer, same running weights). Returns the
    final adapted model plus a per-step record of regime / entity_type / interaction /
    pre-update squared error.

    IMPORTANT: the online trainer only observes (buffers + gradient-updates on) actual
    INTERACTION events, not every environment step. An earlier version called
    online.observe() on every step, including plain movement -- whose target is always
    just -step_penalty, already solved by the offline baseline, and which vastly
    outnumbers real interaction events under a mostly-random policy. That flooded the
    small sliding-window buffer with easy majority-class samples and washed out the
    rare interaction signal almost immediately, causing the model to drift towards
    predicting "no interaction" even with ZERO rule change -- confirmed by a direct
    diagnostic run showing elevated, consistent type-2 error within the first ~400
    steps of the UNCHANGED pre-change world. Movement steps carry no information about
    "the rule" (Section 34's X -> energy relationship is defined only by interaction
    outcomes), so restricting adaptation to interaction events is not cherry-picking
    data, it's the only data that is actually informative for this question -- and it
    makes the buffer's composition naturally reflect the mix of entity types the agent
    has recently interacted with, which is exactly what the catastrophic-forgetting
    experiment (E003) needs it to do.
    """
    rng = random.Random(seed)
    optimizer = Adam(params=model.get_params(), lr=lr, weight_decay=1e-4)
    online = OnlineAdaptationTrainer(model, optimizer=optimizer, buffer_size=buffer_size)

    world = GridWorld(config=world_pre)
    obs = world.get_observation()
    records: List[Dict[str, Any]] = []
    total_steps = pre_steps + post_steps

    for step in range(total_steps):
        if step == pre_steps:
            world.config = world_post
            world.reset(seed=seed + 1)
            obs = world.get_observation()

        act = Action.INTERACT if rng.random() < interaction_prob else rng.choice(MOVEMENT_ACTIONS)
        next_obs, delta, done, info = world.step(act)

        interacted = bool(info.get("interaction", False))
        error = None
        if interacted:
            x = encode_state_action(obs, act)
            y = np.array([delta], dtype=np.float64)
            error = online.observe(x, y)

        records.append({
            "step": step,
            "regime": "pre" if step < pre_steps else "post",
            "interacted": interacted,
            "entity_type": obs.entities[0].entity_type if obs.entities else None,
            "squared_error": error,
        })

        if done:
            world.reset(seed=rng.randint(0, 100_000))
            obs = world.get_observation()
        else:
            obs = next_obs

    return model, records


def run_experiment_e002(seed: int = 42, flip_type: int = 2, pre_steps: int = 3000, post_steps: int = 3000) -> Dict[str, Any]:
    """CSP-P6-E002: Detection & Adaptation. Entity type `flip_type`'s hidden_state_delta
    sign is reversed mid-stream (Section 34's example exactly: X used to increase
    internal_state, now it decreases it). Measures how many type-`flip_type`
    interaction events pass before the online model's error signals the change
    (detection), and how many more before it recovers to pre-change accuracy
    (adaptation).
    """
    pre_world = _load_world("phase6_world_base.json")
    post_world = _flip_type(pre_world, flip_type)

    baseline_model, baseline_info = _train_offline_baseline(pre_world, seed)
    # A fresh copy via to_dict/from_dict: the online loop mutates its model in place,
    # so this adapted model is deliberately a SEPARATE object from baseline_model,
    # which E004 keeps frozen as the persistence control.
    online_model = PredictiveMLP.from_dict(baseline_model.to_dict())

    adapted_model, records = _run_online_stream(
        online_model, pre_world, post_world, pre_steps=pre_steps, post_steps=post_steps, seed=seed,
    )

    flip_events = [r for r in records if r["interacted"] and r["entity_type"] == flip_type]
    measurement = detect_and_measure_adaptation(flip_events)

    return {
        "experiment_id": "CSP-P6-E002",
        "title": "Detection & Adaptation: Does Caspian Notice and Correct a Rule Change?",
        "seed": seed,
        "flip_type": flip_type,
        "pre_steps": pre_steps,
        "post_steps": post_steps,
        "offline_baseline_val_r2": baseline_info["val_metrics"]["r2_score"],
        "num_flip_type_interaction_events": len(flip_events),
        **measurement,
        "detection_and_adaptation_both_occurred": (
            measurement.get("detection_event_index") is not None and measurement.get("adaptation_event_index") is not None
        ),
    }, adapted_model, baseline_model, pre_world, post_world, records


def run_experiment_e002_stability(seeds: List[int] = [42, 101, 7, 2026, 555]) -> Dict[str, Any]:
    """Multi-seed hardening of CSP-P6-E002, the same way Phase 5 checked single-seed
    results before trusting them (E003, E004-STABILITY, E007-STABILITY). A single run
    is not evidence by itself (Section 41: "never interpret a single run as definitive
    evidence") -- does detection/adaptation reproduce across independent offline-training
    + online-stream runs, or was seed=42 a favorable draw?
    """
    per_seed = []
    for s in seeds:
        e002, _, _, _, _, _ = run_experiment_e002(seed=s)
        per_seed.append({
            "seed": s,
            "num_post_events": e002["num_post_events"],
            "events_to_detect": e002["events_to_detect"],
            "events_to_fully_adapt": e002["events_to_fully_adapt"],
            "detection_and_adaptation_both_occurred": e002["detection_and_adaptation_both_occurred"],
            "recovered_to_baseline_by_end": e002["recovered_to_baseline_by_end"],
        })

    both_rate = 100.0 * sum(r["detection_and_adaptation_both_occurred"] for r in per_seed) / len(per_seed)
    detect_vals = [r["events_to_detect"] for r in per_seed if r["events_to_detect"] is not None]
    adapt_vals = [r["events_to_fully_adapt"] for r in per_seed if r["events_to_fully_adapt"] is not None]

    return {
        "experiment_id": "CSP-P6-E002-STABILITY",
        "title": "Stability Check: Does Detection & Adaptation Reproduce Across Seeds?",
        "seeds": seeds,
        "per_seed_results": per_seed,
        "detection_and_adaptation_both_occurred_rate_pct": round(both_rate, 1),
        "mean_events_to_detect": round(float(np.mean(detect_vals)), 2) if detect_vals else None,
        "mean_events_to_fully_adapt": round(float(np.mean(adapt_vals)), 2) if adapt_vals else None,
        "std_events_to_fully_adapt": round(float(np.std(adapt_vals)), 2) if adapt_vals else None,
    }


def run_experiment_e003(seed: int = 42, flip_type: int = 2, pre_steps: int = 3000, post_steps: int = 3000) -> Dict[str, Any]:
    """CSP-P6-E003: Catastrophic Forgetting. After adapting online to `flip_type`'s rule
    change, is the model's accuracy on the OTHER (unchanged) types still intact, or did
    the sliding-window updates -- which only ever see recent data, dominated by
    `flip_type` once its rule changes -- damage what was learned about types that never
    changed? Compared against the SAME offline baseline's accuracy on those types
    before any online adaptation happened.
    """
    e002_result = run_experiment_e002(seed=seed, flip_type=flip_type, pre_steps=pre_steps, post_steps=post_steps)
    e002, adapted_model, baseline_model, pre_world, post_world, records = e002_result

    other_types = [t for t in [1, 2, 3, 4] if t != flip_type]

    def eval_on_type(model: PredictiveMLP, t: int) -> Dict[str, Any]:
        cfg = copy.deepcopy(pre_world)
        template = next(tt for tt in pre_world.entity_type_pool if int(tt["entity_type"]) == t)
        cfg.entity_type_pool = [dict(template)]
        collector = TrajectoryCollector(world_config=cfg)
        dataset = collector.collect_exploratory_dataset(num_steps=800, interaction_prob=0.5, seed=seed)
        interacted = [tr for tr in dataset.transitions if tr.info.get("interaction", False)]
        X = np.stack([encode_state_action(tr.obs, tr.action) for tr in interacted], axis=0)
        Y = np.stack([tr.to_target_vector() for tr in interacted], axis=0)
        return compute_all_metrics(model.forward(X), Y)

    before_after = {}
    for t in other_types:
        before = eval_on_type(baseline_model, t)
        after = eval_on_type(adapted_model, t)
        before_after[f"type_{t}"] = {
            "mse_before_adaptation": before["mse"],
            "mse_after_adaptation": after["mse"],
            "mse_increase_pct": round(100.0 * (after["mse"] - before["mse"]) / (before["mse"] + 1e-10), 2),
        }

    mean_increase_pct = float(np.mean([v["mse_increase_pct"] for v in before_after.values()]))

    return {
        "experiment_id": "CSP-P6-E003",
        "title": "Catastrophic Forgetting: Does Adapting to the Rule Change Damage Unrelated Types?",
        "seed": seed,
        "flip_type": flip_type,
        "unchanged_types_evaluated": other_types,
        "per_type_results": before_after,
        "mean_mse_increase_pct_on_unchanged_types": round(mean_increase_pct, 2),
        "catastrophic_forgetting_occurred": mean_increase_pct > 50.0,
    }


def run_experiment_e004(seed: int = 42, flip_type: int = 2, pre_steps: int = 3000, post_steps: int = 3000) -> Dict[str, Any]:
    """CSP-P6-E004: Old-Model Persistence Control. A frozen checkpoint of the
    pre-change model (never touched by the online updates) should still predict the
    OLD rule when evaluated on post-change data -- i.e. it should now be WRONG, by
    construction, since the environment changed and it didn't. This is the control
    proving E002's error reduction is genuine weight adaptation in a distinct model
    object, not some shared-state artifact silently updating "baseline_model" too.
    """
    e002_result = run_experiment_e002(seed=seed, flip_type=flip_type, pre_steps=pre_steps, post_steps=post_steps)
    e002, adapted_model, baseline_model, pre_world, post_world, records = e002_result

    def eval_on_post_change_flip_type(model: PredictiveMLP) -> Dict[str, Any]:
        cfg = copy.deepcopy(post_world)
        template = next(tt for tt in post_world.entity_type_pool if int(tt["entity_type"]) == flip_type)
        cfg.entity_type_pool = [dict(template)]
        collector = TrajectoryCollector(world_config=cfg)
        dataset = collector.collect_exploratory_dataset(num_steps=800, interaction_prob=0.5, seed=seed + 999)
        interacted = [tr for tr in dataset.transitions if tr.info.get("interaction", False)]
        X = np.stack([encode_state_action(tr.obs, tr.action) for tr in interacted], axis=0)
        Y = np.stack([tr.to_target_vector() for tr in interacted], axis=0)
        preds = model.forward(X)
        return {
            **compute_all_metrics(preds, Y),
            "mean_predicted_sign_matches_new_rule": float(np.mean(np.sign(preds.flatten()) == np.sign(Y.flatten()))),
        }

    frozen_result = eval_on_post_change_flip_type(baseline_model)
    adapted_result = eval_on_post_change_flip_type(adapted_model)

    return {
        "experiment_id": "CSP-P6-E004",
        "title": "Old-Model Persistence Control: The Frozen Checkpoint Must Still Be Wrong",
        "seed": seed,
        "flip_type": flip_type,
        "frozen_pre_change_checkpoint_on_new_rule": frozen_result,
        "online_adapted_model_on_new_rule": adapted_result,
        "frozen_checkpoint_still_wrong_as_expected": frozen_result["mean_predicted_sign_matches_new_rule"] < 0.3,
        "adapted_model_now_correct": adapted_result["mean_predicted_sign_matches_new_rule"] > 0.7,
        "control_valid": (
            frozen_result["mean_predicted_sign_matches_new_rule"] < 0.3
            and adapted_result["mean_predicted_sign_matches_new_rule"] > 0.7
        ),
    }


def run_experiment_e005(seed: int = 42, first_flip_type: int = 2, second_flip_type: int = 4, pre_steps: int = 3000, post_steps: int = 3000) -> Dict[str, Any]:
    """CSP-P6-E005: Transfer. After the model has already adapted once (type
    `first_flip_type`'s rule change, as in E002), a SECOND, independent type's rule is
    flipped. Does detection/adaptation happen faster the second time around -- evidence
    the agent "learned how to adapt" in some general sense -- or equally slowly,
    meaning each rule change is handled independently with no meta-adaptation? No
    outcome is assumed; this experiment exists to find out, and reports whichever
    answer the numbers give (Section 41: never optimize only for a desired result).
    """
    pre_world = _load_world("phase6_world_base.json")
    first_post_world = _flip_type(pre_world, first_flip_type)
    second_post_world = _flip_type(first_post_world, second_flip_type)

    baseline_model, _ = _train_offline_baseline(pre_world, seed)
    online_model = PredictiveMLP.from_dict(baseline_model.to_dict())

    # First rule change (identical in structure to E002).
    _, records_1 = _run_online_stream(
        online_model, pre_world, first_post_world, pre_steps=pre_steps, post_steps=post_steps, seed=seed,
    )
    flip1_events = [r for r in records_1 if r["interacted"] and r["entity_type"] == first_flip_type]
    measurement_1 = detect_and_measure_adaptation(flip1_events)

    # Second rule change, continuing the SAME online model/buffer (now already adapted
    # once), switching straight from first_post_world to second_post_world -- no fresh
    # "pre" period here, since the model's current error ON SECOND_FLIP_TYPE while still
    # under first_post_world's (unchanged, for that type) rule serves as its baseline.
    rng = random.Random(seed + 1)
    optimizer_2 = Adam(params=online_model.get_params(), lr=0.01, weight_decay=1e-4)
    online_2 = OnlineAdaptationTrainer(online_model, optimizer=optimizer_2, buffer_size=32)
    world = GridWorld(config=first_post_world)
    obs = world.get_observation()
    records_2: List[Dict[str, Any]] = []
    total_steps_2 = pre_steps + post_steps
    for step in range(total_steps_2):
        if step == pre_steps:
            world.config = second_post_world
            world.reset(seed=seed + 2)
            obs = world.get_observation()
        act = Action.INTERACT if rng.random() < 0.4 else rng.choice(MOVEMENT_ACTIONS)
        next_obs, delta, done, info = world.step(act)
        interacted = bool(info.get("interaction", False))
        error = None
        if interacted:
            x = encode_state_action(obs, act)
            y = np.array([delta], dtype=np.float64)
            error = online_2.observe(x, y)
        records_2.append({
            "step": step, "regime": "pre" if step < pre_steps else "post",
            "interacted": interacted,
            "entity_type": obs.entities[0].entity_type if obs.entities else None,
            "squared_error": error,
        })
        if done:
            world.reset(seed=rng.randint(0, 100_000))
            obs = world.get_observation()
        else:
            obs = next_obs

    flip2_events = [r for r in records_2 if r["interacted"] and r["entity_type"] == second_flip_type]
    measurement_2 = detect_and_measure_adaptation(flip2_events)

    def faster(a, b):
        return (a is not None and b is not None and a < b)

    return {
        "experiment_id": "CSP-P6-E005",
        "title": "Transfer: Is the Second Rule Change Detected/Adapted Faster Than the First?",
        "seed": seed,
        "first_flip_type": first_flip_type,
        "second_flip_type": second_flip_type,
        "first_change_measurement": measurement_1,
        "second_change_measurement": measurement_2,
        "detection_faster_second_time": faster(measurement_2.get("events_to_detect"), measurement_1.get("events_to_detect")),
        "adaptation_faster_second_time": faster(measurement_2.get("events_to_fully_adapt"), measurement_1.get("events_to_fully_adapt")),
        "transfer_evidence": (
            "POSITIVE" if faster(measurement_2.get("events_to_detect"), measurement_1.get("events_to_detect"))
            and faster(measurement_2.get("events_to_fully_adapt"), measurement_1.get("events_to_fully_adapt"))
            else "NONE_OR_MIXED"
        ),
    }


def run_all_phase6_experiments(output_dir: str = "logs") -> Dict[str, Any]:
    """Execute all Phase 6 experiments sequentially and persist results."""
    import os

    print("Running CSP-P6-E001 (Offline Baseline)...")
    e001 = run_experiment_e001()

    print("Running CSP-P6-E002 (Detection & Adaptation)...")
    e002_full = run_experiment_e002()
    e002 = e002_full[0]

    print("Running CSP-P6-E002 Stability (multi-seed hardening)...")
    e002_stability = run_experiment_e002_stability()

    print("Running CSP-P6-E003 (Catastrophic Forgetting)...")
    e003 = run_experiment_e003()

    print("Running CSP-P6-E004 (Old-Model Persistence Control)...")
    e004 = run_experiment_e004()

    print("Running CSP-P6-E005 (Transfer)...")
    e005 = run_experiment_e005()

    all_results = {
        "phase": "Phase 6 — Rule Change & Adaptation",
        "experiments": {
            "CSP-P6-E001": e001,
            "CSP-P6-E002": e002,
            "CSP-P6-E002-STABILITY": e002_stability,
            "CSP-P6-E003": e003,
            "CSP-P6-E004": e004,
            "CSP-P6-E005": e005,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase6_all_experiments_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Phase 6 experiments completed and saved to: {out_file}")
    return all_results


if __name__ == "__main__":
    run_all_phase6_experiments()
