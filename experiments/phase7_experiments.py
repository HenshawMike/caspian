"""Experimental Suite for Project Caspian (Phase 7 — Causal Experimentation).

Addresses Section 35 of the Development Plan: move beyond observation. Every prior
phase's training data came from a PASSIVE policy (random movement, random INTERACT
attempts -- learning/collector.py's collect_exploratory_dataset) -- Caspian never
chose what to approach or when to repeat an interaction; it just wandered and
whatever happened, happened. Section 35 asks whether an agent that actively chooses
its interventions -- approaching a specific entity, interacting, moving away,
returning, repeating, deliberately varying the circumstances of an interaction --
can discover causal structure that a passive observer cannot, or can discover it
with fewer interactions.

Three experiments, each operationalizing one piece of Section 35's list
(approach/interact/move away/repeat/compare outcomes) as a concrete, measurable test:

- CSP-P7-E001 (Repeated-Intervention Consistency): literally "repeat interaction,
  compare outcomes." The agent deliberately navigates to the same entity, interacts,
  moves away, returns, and interacts again, across many independent trials. Does
  repeating the SAME intervention under matched conditions produce a reproducible
  outcome (the signature of a real causal effect), or noisy/inconsistent ones?

- CSP-P7-E002 (Confounded Observation vs. Active Intervention -- the central
  question): a PASSIVE collector confined to a small region of the grid builds a
  dataset where entity_type and POSITION are perfectly confounded (type A only ever
  seen near one corner, type B only ever seen near another) purely as a side effect
  of where it happened to wander -- nothing deceptive is built into the environment,
  this is what observation-only data collection produces by construction. Does a
  model trained on it latch onto the spurious position cue, failing when evaluated on
  a world where the SAME types appear in SWAPPED positions? An ACTIVE agent that
  deliberately visits each type in multiple different positions (an intervention a
  passive wanderer confined to one area never performs) breaks the confound by
  construction. This is Section 35's actual research question, directly tested: can
  selecting interventions recover the true causal structure where passive observation
  cannot?

- CSP-P7-E003 (Active vs. Passive Sample Efficiency): in the ordinary, unconfounded
  4-type world, does an agent that deliberately targets whichever entity type it has
  interacted with LEAST so far (the simplest possible "experiment selection" policy --
  earn complexity: no Bayesian uncertainty estimation or ensemble disagreement is
  assumed necessary without evidence) reach a given prediction accuracy using fewer
  total interactions than an agent that interacts with whatever it happens to bump
  into under a random policy?
"""

import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from environment.entities import Entity
from models.mlp import PredictiveMLP
from learning.dataset import encode_state_action, ExperienceDataset, Transition
from learning.collector import TrajectoryCollector
from learning.trainer import PredictiveTrainer
from evaluation.metrics import compute_all_metrics

CONFIG_DIR = Path(__file__).parent.parent / "configs"
MOVEMENT_ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]


def _load_world(name: str) -> WorldConfig:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return WorldConfig.from_dict(json.load(f))


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_towards(agent_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Action:
    """Greedy single-step action moving the agent one cell closer to target_pos
    (x-axis first, then y-axis). This is the "approach X" primitive Section 35 asks
    for -- the agent deliberately navigates to a chosen entity rather than wandering.
    """
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    if dx > 0:
        return Action.RIGHT
    if dx < 0:
        return Action.LEFT
    if dy > 0:
        return Action.UP
    if dy < 0:
        return Action.DOWN
    return Action.NOOP


def _step_away(agent_pos: Tuple[int, int], target_pos: Tuple[int, int], grid_w: int, grid_h: int) -> Action:
    """Greedy single-step action moving the agent one cell FARTHER from target_pos,
    staying in bounds -- the "move away from X" primitive."""
    candidates = []
    for act, (dx, dy) in [(Action.RIGHT, (1, 0)), (Action.LEFT, (-1, 0)), (Action.UP, (0, 1)), (Action.DOWN, (0, -1))]:
        nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
        if 0 <= nx < grid_w and 0 <= ny < grid_h:
            candidates.append((act, _manhattan((nx, ny), target_pos)))
    if not candidates:
        return Action.NOOP
    return max(candidates, key=lambda c: c[1])[0]


def _approach_and_interact(world: GridWorld, target_pos: Tuple[int, int], max_steps: int = 10) -> Optional[Tuple[Any, float, Dict[str, Any]]]:
    """Navigate the agent to within interaction range of target_pos, then INTERACT.
    Returns (obs_at_interact_time, state_delta, info) or None if it couldn't get there
    within max_steps (shouldn't happen on this 5x5 grid, but guarded defensively).
    """
    obs = world.get_observation()
    radius = world.config.interaction_radius
    for _ in range(max_steps):
        if _manhattan(world._agent_pos, target_pos) <= radius:
            interact_obs = obs
            next_obs, delta, done, info = world.step(Action.INTERACT)
            return interact_obs, delta, info
        act = _step_towards(world._agent_pos, target_pos)
        obs, _, done, _ = world.step(act)
        if done:
            return None
    return None


def run_experiment_e001(seed: int = 42, flip_target: int = 1, num_trials: int = 40) -> Dict[str, Any]:
    """CSP-P7-E001: Repeated-Intervention Consistency. Across `num_trials` independent
    episodes (fresh reset each time, so internal_state never clamps and trials are
    independent), the agent: approaches the entity, interacts (outcome 1), moves away
    for several steps, returns, interacts again (outcome 2). If this is a real causal
    effect, outcome 1 and outcome 2 should match; if the environment or the agent's
    control logic were noisy/non-deterministic in some hidden way, they would differ.
    """
    world_config = _load_world("phase7_world_single_type.json")
    rng = random.Random(seed)
    first_outcomes, second_outcomes, mismatches = [], [], 0

    for trial in range(num_trials):
        world = GridWorld(config=world_config)
        world.reset(seed=rng.randint(0, 100_000))
        entity_pos = world._entities[0].position

        result1 = _approach_and_interact(world, entity_pos)
        if result1 is None:
            continue
        _, delta1, info1 = result1
        if not info1.get("interaction", False):
            continue

        # Move away for a few steps, then return and interact again.
        for _ in range(3):
            act = _step_away(world._agent_pos, entity_pos, world_config.grid_width, world_config.grid_height)
            world.step(act)

        result2 = _approach_and_interact(world, entity_pos)
        if result2 is None:
            continue
        _, delta2, info2 = result2
        if not info2.get("interaction", False):
            continue

        first_outcomes.append(delta1)
        second_outcomes.append(delta2)
        if abs(delta1 - delta2) > 1e-9:
            mismatches += 1

    n = len(first_outcomes)
    diffs = [abs(a - b) for a, b in zip(first_outcomes, second_outcomes)]

    return {
        "experiment_id": "CSP-P7-E001",
        "title": "Repeated-Intervention Consistency: Does the Same Intervention Give the Same Outcome?",
        "seed": seed,
        "num_trials_attempted": num_trials,
        "num_trials_completed": n,
        "mean_first_outcome": round(float(np.mean(first_outcomes)), 4) if n else None,
        "mean_second_outcome": round(float(np.mean(second_outcomes)), 4) if n else None,
        "mean_absolute_difference": round(float(np.mean(diffs)), 6) if n else None,
        "max_absolute_difference": round(float(np.max(diffs)), 6) if n else None,
        "num_mismatches": mismatches,
        "reproducibility_rate_pct": round(100.0 * (n - mismatches) / n, 2) if n else None,
        "intervention_is_reproducible": (mismatches == 0 and n > 0),
    }


def _confounded_type_configs() -> Tuple[WorldConfig, WorldConfig]:
    """Two single-type worlds whose entity is confined to disjoint corners of the grid
    -- type 1 only ever appears near (0,0)-(1,1), type 3 only ever appears near
    (3,3)-(4,4). A PASSIVE collector run separately on each and concatenated produces a
    dataset where entity_type and position are perfectly confounded, purely as a side
    effect of where each collector happened to look -- nothing adversarial, just what
    observation-only data collection naturally yields when exploration is localized.
    """
    base = _load_world("phase7_world_base.json")

    type1_near = copy.deepcopy(base)
    type1_near.initial_entities = [Entity(entity_id=1, entity_type=1, position=(1, 1), hidden_state_delta=10.0)]
    type1_near.allowed_entity_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    type1_near.randomize_entity_positions = True
    type1_near.initial_agent_pos = (0, 0)

    type3_far = copy.deepcopy(base)
    type3_far.initial_entities = [Entity(entity_id=1, entity_type=3, position=(3, 3), hidden_state_delta=-10.0)]
    type3_far.allowed_entity_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
    type3_far.randomize_entity_positions = True
    type3_far.initial_agent_pos = (4, 4)

    return type1_near, type3_far


def _swapped_type_configs() -> Tuple[WorldConfig, WorldConfig]:
    """Same two types, SWAPPED to the opposite corner from CSP-P7-E002's confounded
    training distribution -- the test of whether a model relied on the true cause
    (entity_type) or the spurious one (position).
    """
    base = _load_world("phase7_world_base.json")

    type1_far = copy.deepcopy(base)
    type1_far.initial_entities = [Entity(entity_id=1, entity_type=1, position=(3, 3), hidden_state_delta=10.0)]
    type1_far.allowed_entity_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
    type1_far.randomize_entity_positions = True
    type1_far.initial_agent_pos = (4, 4)

    type3_near = copy.deepcopy(base)
    type3_near.initial_entities = [Entity(entity_id=1, entity_type=3, position=(1, 1), hidden_state_delta=-10.0)]
    type3_near.allowed_entity_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    type3_near.randomize_entity_positions = True
    type3_near.initial_agent_pos = (0, 0)

    return type1_far, type3_near


def _collect_passive(world_config: WorldConfig, seed: int, num_steps: int) -> ExperienceDataset:
    collector = TrajectoryCollector(world_config=world_config)
    return collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.4, seed=seed)


def _collect_active_varied_positions(entity_type: int, hidden_state_delta: float, corners: List[Tuple[int, int]], seed: int, num_steps_per_corner: int) -> ExperienceDataset:
    """The ACTIVE counterpart to _confounded_type_configs(): the agent deliberately
    visits the SAME entity type in MULTIPLE different positions (every corner, not just
    the one it happens to be confined near), which is exactly the "approach X" +
    "deliberately vary the circumstances of the intervention" behavior a passive
    wanderer confined to one region cannot produce. Breaks the type/position confound
    by construction, through choice of what to sample, not through any change to the
    environment's own mechanics.
    """
    base = _load_world("phase7_world_base.json")
    combined = ExperienceDataset()
    for i, corner in enumerate(corners):
        cfg = copy.deepcopy(base)
        cfg.initial_entities = [Entity(entity_id=1, entity_type=entity_type, position=corner, hidden_state_delta=hidden_state_delta)]
        cfg.allowed_entity_positions = [corner]
        cfg.randomize_entity_positions = False
        cfg.initial_agent_pos = corner
        ds = _collect_passive(cfg, seed=seed + i, num_steps=num_steps_per_corner)
        combined.add_transitions(ds.transitions)
    return combined


def _train_on_dataset(dataset: ExperienceDataset, seed: int) -> Tuple[PredictiveMLP, Dict[str, Any]]:
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)
    X_train, Y_train = train_ds.to_numpy_arrays()
    X_val, Y_val = val_ds.to_numpy_arrays()
    model = PredictiveMLP(input_dim=X_train.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit((X_train, Y_train), val_data=(X_val, Y_val))
    val_metrics = compute_all_metrics(model.forward(X_val), Y_val) if len(X_val) else None
    return model, {"epochs_run": history["epochs_run"], "val_metrics": val_metrics, "num_train_samples": len(X_train)}


def _eval_on_world(model: PredictiveMLP, world_config: WorldConfig, seed: int, num_steps: int = 800) -> Dict[str, Any]:
    ds = _collect_passive(world_config, seed=seed, num_steps=num_steps)
    interacted = [t for t in ds.transitions if t.info.get("interaction", False)]
    X = np.stack([encode_state_action(t.obs, t.action) for t in interacted], axis=0)
    Y = np.stack([t.to_target_vector() for t in interacted], axis=0)
    return compute_all_metrics(model.forward(X), Y)


def run_experiment_e002(seed: int = 42, steps_per_region: int = 3000) -> Dict[str, Any]:
    """CSP-P7-E002: Confounded Observation vs. Active Intervention -- Section 35's
    central research question, directly tested.

    Data volume is EQUAL across arms: passive = 2 types x steps_per_region steps; active =
    2 types x 4 corners x steps_per_region/4 steps. (An earlier run gave the active arm
    2x the steps, which would have confounded "better data" with "more data"; corrected.)
    """
    type1_near, type3_far = _confounded_type_configs()

    passive_ds = ExperienceDataset()
    passive_ds.add_transitions(_collect_passive(type1_near, seed=seed, num_steps=steps_per_region).transitions)
    passive_ds.add_transitions(_collect_passive(type3_far, seed=seed + 1, num_steps=steps_per_region).transitions)
    passive_model, passive_info = _train_on_dataset(passive_ds, seed=seed)

    corners = [(0, 0), (4, 4), (0, 4), (4, 0)]
    active_ds = ExperienceDataset()
    active_ds.add_transitions(_collect_active_varied_positions(1, 10.0, corners, seed=seed, num_steps_per_corner=steps_per_region // 4).transitions)
    active_ds.add_transitions(_collect_active_varied_positions(3, -10.0, corners, seed=seed + 10, num_steps_per_corner=steps_per_region // 4).transitions)
    active_model, active_info = _train_on_dataset(active_ds, seed=seed)

    type1_far, type3_near = _swapped_type_configs()

    def eval_both(model):
        return {
            "type1_near_matched": _eval_on_world(model, type1_near, seed=seed + 100),
            "type3_far_matched": _eval_on_world(model, type3_far, seed=seed + 101),
            "type1_far_swapped": _eval_on_world(model, type1_far, seed=seed + 102),
            "type3_near_swapped": _eval_on_world(model, type3_near, seed=seed + 103),
        }

    passive_eval = eval_both(passive_model)
    active_eval = eval_both(active_model)

    def matched_vs_swapped_r2(ev):
        matched_r2 = float(np.mean([ev["type1_near_matched"]["r2_score"], ev["type3_far_matched"]["r2_score"]]))
        swapped_r2 = float(np.mean([ev["type1_far_swapped"]["r2_score"], ev["type3_near_swapped"]["r2_score"]]))
        return matched_r2, swapped_r2

    passive_matched_r2, passive_swapped_r2 = matched_vs_swapped_r2(passive_eval)
    active_matched_r2, active_swapped_r2 = matched_vs_swapped_r2(active_eval)

    return {
        "experiment_id": "CSP-P7-E002",
        "title": "Confounded Observation vs. Active Intervention: Discovering True Causal Structure",
        "seed": seed,
        "passive_training": {"num_train_samples": passive_info["num_train_samples"], "val_metrics": passive_info["val_metrics"]},
        "active_training": {"num_train_samples": active_info["num_train_samples"], "val_metrics": active_info["val_metrics"]},
        "passive_model_eval": passive_eval,
        "active_model_eval": active_eval,
        "passive_matched_position_r2": round(passive_matched_r2, 4),
        "passive_swapped_position_r2": round(passive_swapped_r2, 4),
        "active_matched_position_r2": round(active_matched_r2, 4),
        "active_swapped_position_r2": round(active_swapped_r2, 4),
        "passive_model_relied_on_spurious_position": bool((passive_matched_r2 - passive_swapped_r2) > 0.3),
        "active_model_robust_to_position_swap": bool((active_matched_r2 - active_swapped_r2) < 0.15),
        "active_intervention_recovered_true_cause_where_passive_failed": bool(
            ((passive_matched_r2 - passive_swapped_r2) > 0.3) and ((active_matched_r2 - active_swapped_r2) < 0.15)
        ),
    }


def run_experiment_e002_stability(seeds: List[int] = [42, 101, 7, 2026, 555]) -> Dict[str, Any]:
    """Multi-seed hardening of CSP-P7-E002 (Section 41: a single run is not evidence)."""
    per_seed = []
    for s in seeds:
        e = run_experiment_e002(seed=s)
        per_seed.append({
            "seed": s,
            "passive_matched_r2": e["passive_matched_position_r2"],
            "passive_swapped_r2": e["passive_swapped_position_r2"],
            "active_matched_r2": e["active_matched_position_r2"],
            "active_swapped_r2": e["active_swapped_position_r2"],
            "recovered_true_cause": e["active_intervention_recovered_true_cause_where_passive_failed"],
        })
    return {
        "experiment_id": "CSP-P7-E002-STABILITY",
        "title": "Stability Check: Does the Confound Result Reproduce Across Seeds?",
        "seeds": seeds,
        "per_seed_results": per_seed,
        "reproduced_rate_pct": round(100.0 * sum(r["recovered_true_cause"] for r in per_seed) / len(per_seed), 1),
        "mean_passive_swap_drop": round(float(np.mean([r["passive_matched_r2"] - r["passive_swapped_r2"] for r in per_seed])), 4),
        "mean_active_swap_drop": round(float(np.mean([r["active_matched_r2"] - r["active_swapped_r2"] for r in per_seed])), 4),
    }


def _collect_active_budgeted(world_config: WorldConfig, num_steps: int, seed: int) -> ExperienceDataset:
    """ACTIVE collection under an exact env-step budget. Each episode the agent
    deliberately runs a small experiment loop on the entity the episode presents:
    approach it, interact, move away, return, interact again (Section 35's
    approach / interact / move away / repeat vocabulary), until the episode ends.
    Every environment step counts against `num_steps` -- no step is free -- so this
    is directly comparable to the passive policy at the same budget. The agent does
    NOT choose which entity type appears (the environment resamples type on reset;
    the semantic firewall gives the agent no handle on it), only how it spends the
    steps once an entity is present.
    """
    rng = random.Random(seed)
    combined = ExperienceDataset()
    steps_used = 0
    while steps_used < num_steps:
        world = GridWorld(config=world_config)
        world.reset(seed=rng.randint(0, 100_000))
        if not world._entities:
            continue
        entity = world._entities[0]
        obs = world.get_observation()
        away_remaining = 0
        done = False
        while not done and steps_used < num_steps:
            dist = _manhattan(world._agent_pos, entity.position)
            if away_remaining > 0:
                act = _step_away(world._agent_pos, entity.position, world_config.grid_width, world_config.grid_height)
                away_remaining -= 1
            elif dist <= world_config.interaction_radius:
                act = Action.INTERACT
                away_remaining = 2  # after each interaction, step away, then come back and repeat
            else:
                act = _step_towards(world._agent_pos, entity.position)
            next_obs, delta, done, info = world.step(act)
            combined.add(Transition(obs=obs, action=act, state_delta=delta, next_obs=next_obs, done=done, info=info))
            obs = next_obs
            steps_used += 1
    return combined


def _interaction_xy(ds: ExperienceDataset) -> Tuple[np.ndarray, np.ndarray]:
    inter = [t for t in ds.transitions if t.info.get("interaction", False)]
    X = np.stack([encode_state_action(t.obs, t.action) for t in inter], axis=0)
    Y = np.stack([t.to_target_vector() for t in inter], axis=0)
    return X, Y


def _train_xy(ds: ExperienceDataset, seed: int) -> PredictiveMLP:
    """Train on ALL collected transitions (no validation split carved from the same
    stream -- evaluation happens on an independent common test set instead, so
    near-duplicate transitions from one episode can never leak across train/val)."""
    X, Y = ds.to_numpy_arrays()
    model = PredictiveMLP(input_dim=X.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=100, early_stopping_patience=None, seed=seed)
    trainer.fit((X, Y))
    return model


def run_experiment_e003(seeds: List[int] = [42, 101, 7, 2026, 555], budgets: List[int] = [250, 500, 1000, 2000, 4000]) -> Dict[str, Any]:
    """CSP-P7-E003: Active vs. Passive Sample Efficiency at EQUAL environment-step
    budgets, in the ordinary unconfounded 4-type world, evaluated on an INDEPENDENT
    common test set (collected by the passive policy, with its own seed) for both
    models. Two test slices: interaction events only (does the model know what
    entities do?) and all transitions (does it also cover everything else the passive
    distribution contains, e.g. INTERACT attempted far from any entity?). An earlier
    smoke version of this experiment counted repeated within-episode interactions as
    independent samples and validated each model on its own stream; it reported a
    spurious ~95% step saving. Replaced by this equal-budget, shared-test-set design.
    """
    world_config = _load_world("phase6_world_base.json")
    per_budget: Dict[str, Any] = {}

    for B in budgets:
        rows = []
        for s in seeds:
            test_ds = _collect_passive(world_config, seed=s + 5000, num_steps=3000)
            Xi, Yi = _interaction_xy(test_ds)
            Xa, Ya = test_ds.to_numpy_arrays()

            passive_ds = _collect_passive(world_config, seed=s, num_steps=B)
            active_ds = _collect_active_budgeted(world_config, num_steps=B, seed=s)

            row = {"seed": s}
            for name, ds in [("passive", passive_ds), ("active", active_ds)]:
                model = _train_xy(ds, seed=s)
                row[f"{name}_interaction_r2"] = compute_all_metrics(model.forward(Xi), Yi)["r2_score"]
                row[f"{name}_all_r2"] = compute_all_metrics(model.forward(Xa), Ya)["r2_score"]
                row[f"{name}_num_interactions_collected"] = sum(1 for t in ds.transitions if t.info.get("interaction", False))
            rows.append(row)

        def agg(key):
            v = [r[key] for r in rows]
            return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}

        per_budget[str(B)] = {
            "per_seed": rows,
            "passive_interaction_r2": agg("passive_interaction_r2"),
            "active_interaction_r2": agg("active_interaction_r2"),
            "passive_all_r2": agg("passive_all_r2"),
            "active_all_r2": agg("active_all_r2"),
            "passive_mean_interactions": round(float(np.mean([r["passive_num_interactions_collected"] for r in rows])), 1),
            "active_mean_interactions": round(float(np.mean([r["active_num_interactions_collected"] for r in rows])), 1),
        }

    wins_interaction = sum(per_budget[str(B)]["active_interaction_r2"]["mean"] > per_budget[str(B)]["passive_interaction_r2"]["mean"] for B in budgets)
    wins_all = sum(per_budget[str(B)]["active_all_r2"]["mean"] > per_budget[str(B)]["passive_all_r2"]["mean"] for B in budgets)

    return {
        "experiment_id": "CSP-P7-E003",
        "title": "Active vs. Passive Sample Efficiency at Equal Environment-Step Budgets",
        "seeds": seeds,
        "budgets": budgets,
        "per_budget": per_budget,
        "budgets_where_active_better_on_interaction_events": f"{wins_interaction}/{len(budgets)}",
        "budgets_where_active_better_on_all_transitions": f"{wins_all}/{len(budgets)}",
    }


# ---------------------------------------------------------------------------
# CSP-P7-E004: autonomous (model-driven) selection of where to intervene
# ---------------------------------------------------------------------------

_E004_CORNERS = [(0, 0), (4, 4), (0, 4), (4, 0)]
_E004_TYPES = {1: 10.0, 3: -10.0}


def _site_config(entity_type: int, corner: Tuple[int, int]) -> WorldConfig:
    cfg = copy.deepcopy(_load_world("phase7_world_base.json"))
    cfg.initial_entities = [Entity(entity_id=1, entity_type=entity_type, position=corner, hidden_state_delta=_E004_TYPES[entity_type])]
    cfg.allowed_entity_positions = [corner]
    cfg.randomize_entity_positions = False
    cfg.initial_agent_pos = corner
    return cfg


def _all_sites() -> List[Tuple[int, Tuple[int, int]]]:
    return [(t, c) for t in _E004_TYPES for c in _E004_CORNERS]


def _fit_quick(ds: ExperienceDataset, seed: int, epochs: int) -> PredictiveMLP:
    X, Y = ds.to_numpy_arrays()
    model = PredictiveMLP(input_dim=X.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    PredictiveTrainer(model=model, batch_size=32, epochs=epochs, early_stopping_patience=None, seed=seed).fit((X, Y))
    return model


def _ensemble_disagreement(ds: ExperienceDataset, probe_X: np.ndarray, seed: int, members: int = 3) -> float:
    preds = [_fit_quick(ds, seed + 31 * m, epochs=40).forward(probe_X).reshape(-1) for m in range(members)]
    return float(np.mean(np.var(np.stack(preds, axis=0), axis=0)))


def _run_e004_arm(arm: str, seed: int, budget: int, seed_block: int = 200, probe_steps: int = 30, block: int = 220) -> Tuple[ExperienceDataset, List[Tuple[int, Tuple[int, int]]]]:
    """Build a dataset under an EXACT env-step budget. All arms start with the same
    naturally confounded seed data (type1@(0,0), type3@(4,4)). Arms differ only in how
    the remaining steps are allocated across candidate sites (type x corner blocks):
      confounded : never leaves the confounded sites (observation-only baseline)
      random     : uniformly random site per block
      roundrobin : hand-designed coverage (cycle sites)  [E002-style]
      count      : site with the fewest samples so far (simple non-model heuristic)
      uncertainty: site whose cached probe inputs the model-ensemble disagrees on most.
                   Probe steps (30/site) ARE counted against the budget and their data is used.
    """
    rng = random.Random(seed)
    sites = _all_sites()
    seed_sites = [(1, (0, 0)), (3, (4, 4))]
    ds = ExperienceDataset()
    counts = {s: 0 for s in sites}
    chosen: List[Tuple[int, Tuple[int, int]]] = []
    used = 0

    def collect(site, n, sd):
        nonlocal used
        d = _collect_passive(_site_config(*site), seed=sd, num_steps=n)
        ds.add_transitions(d.transitions)
        counts[site] += n
        used += n
        return d

    for i, st in enumerate(seed_sites):
        collect(st, seed_block, seed * 7 + i)

    probes: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}
    if arm == "uncertainty":
        for i, st in enumerate(sites):
            d = collect(st, probe_steps, seed * 13 + i)  # probe data are real, counted observations
            inter = [t for t in d.transitions if t.info.get("interaction", False)]
            if inter:
                probes[st] = np.stack([encode_state_action(t.obs, t.action) for t in inter], axis=0)
            else:  # no interaction happened in the probe: fall back to all transitions at that site
                probes[st] = d.to_numpy_arrays()[0]
    rr = 0
    k = 0
    while used < budget:
        n = min(block, budget - used)
        if arm == "confounded":
            site = seed_sites[k % 2]
        elif arm == "random":
            site = rng.choice(sites)
        elif arm == "roundrobin":
            site = sites[rr % len(sites)]; rr += 1
        elif arm == "count":
            site = min(sites, key=lambda s: (counts[s], sites.index(s)))
        elif arm == "uncertainty":
            scores = {s: _ensemble_disagreement(ds, probes[s], seed + k) for s in sites}
            site = max(scores, key=scores.get)
        else:
            raise ValueError(arm)
        chosen.append(site)
        collect(site, n, seed * 101 + k)
        k += 1
    return ds, chosen


def run_experiment_e004(seeds: List[int] = [42, 101, 7, 2026, 555], budget: int = 2400) -> Dict[str, Any]:
    """CSP-P7-E004: can Caspian itself decide where to intervene, using only its own
    uncertainty, and thereby break the position/type confound? Equal env-step budgets,
    evaluated on the SWAPPED worlds (type1 far, type3 near) and the matched worlds.
    Limitation recorded up front: the agent chooses among pre-defined candidate sites
    (type x corner); the semantic firewall is respected (type codes are neutral numbers)
    but site menu construction is still experimenter-provided.
    """
    arms = ["confounded", "random", "roundrobin", "count", "uncertainty"]
    t1_far, t3_near = _swapped_type_configs()
    t1_near, t3_far = _confounded_type_configs()
    rows = []
    for s in seeds:
        row = {"seed": s}
        for arm in arms:
            ds, chosen = _run_e004_arm(arm, s, budget)
            model = _train_xy(ds, seed=s)
            sw = [_eval_on_world(model, c, seed=s + 900 + i)["r2_score"] for i, c in enumerate([t1_far, t3_near])]
            ma = [_eval_on_world(model, c, seed=s + 950 + i)["r2_score"] for i, c in enumerate([t1_near, t3_far])]
            row[arm] = {
                "swapped_r2": round(float(np.mean(sw)), 4),
                "matched_r2": round(float(np.mean(ma)), 4),
                "sites_covered": len(set(chosen)),
                "swapped_sites_chosen": sum(1 for (t, c) in chosen if (t == 1 and c in [(3, 3), (3, 4), (4, 3), (4, 4)]) or (t == 3 and c in [(0, 0), (0, 1), (1, 0), (1, 1)])),
            }
        rows.append(row)
    summary = {}
    for arm in arms:
        v = [r[arm]["swapped_r2"] for r in rows]
        m = [r[arm]["matched_r2"] for r in rows]
        summary[arm] = {
            "swapped_r2_mean": round(float(np.mean(v)), 4), "swapped_r2_std": round(float(np.std(v)), 4),
            "matched_r2_mean": round(float(np.mean(m)), 4),
            "seeds_swapped_r2_above_0.9": sum(x > 0.9 for x in v),
        }
    return {
        "experiment_id": "CSP-P7-E004",
        "title": "Autonomous Uncertainty-Driven Intervention Selection vs Baselines (equal budget)",
        "seeds": seeds, "budget": budget, "arms": arms,
        "per_seed": rows, "summary": summary,
    }


# ---------------------------------------------------------------------------
# CSP-P7-E005: strengthened uncertainty signal + mixed targeted/broad collection
# (a follow-up to E004's negative result; E004 itself is left untouched)
# ---------------------------------------------------------------------------
# Changes vs E004, each motivated by E004's logged failure (the uncertainty policy
# covered only 4-5 of 8 sites per seed, and E004 retrained the ensemble once PER SITE
# per round, which also capped how large an ensemble was affordable):
#   * one 6-member x 80-epoch ensemble trained once per round, scoring all sites
#   * a pre-declared diversity-discounted variant (score / (1 + steps already spent at site / block))
#   * a mixed variant that sends 25% of every block to broad passive exploration
#     (full unconfounded 4-type world) to protect coverage (E003's weakness)
# Arms were declared before running; no arm was added or dropped after seeing results.


def _eval_all_on_world(model: PredictiveMLP, world_config: WorldConfig, seed: int, num_steps: int = 800) -> float:
    ds = _collect_passive(world_config, seed=seed, num_steps=num_steps)
    X, Y = ds.to_numpy_arrays()
    return compute_all_metrics(model.forward(X), Y)["r2_score"]


def _run_e005_arm(arm: str, seed: int, budget: int, seed_block: int = 200, probe_steps: int = 30, block: int = 220,
                  members: int = 6, epochs: int = 80, mix_frac: float = 0.25, broad_world: str = "phase6_world_base.json"):
    rng = random.Random(seed)
    sites = _all_sites()
    seed_sites = [(1, (0, 0)), (3, (4, 4))]
    broad_cfg = _load_world(broad_world)
    ds = ExperienceDataset()
    counts = {s: 0 for s in sites}
    chosen: List[Tuple[int, Tuple[int, int]]] = []
    state = {"used": 0, "broad": 0}

    def collect(site, n, sd):
        d = _collect_passive(_site_config(*site), seed=sd, num_steps=n)
        ds.add_transitions(d.transitions)
        counts[site] += n
        state["used"] += n
        return d

    def collect_broad(n, sd):
        d = _collect_passive(broad_cfg, seed=sd, num_steps=n)
        ds.add_transitions(d.transitions)
        state["used"] += n
        state["broad"] += n

    for i, st in enumerate(seed_sites):
        collect(st, seed_block, seed * 7 + i)

    use_unc = arm.startswith("uncertainty")
    probes: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}
    if use_unc:
        for i, st in enumerate(sites):
            d = collect(st, probe_steps, seed * 13 + i)  # probe data are real, counted observations
            inter = [t for t in d.transitions if t.info.get("interaction", False)]
            probes[st] = np.stack([encode_state_action(t.obs, t.action) for t in inter], axis=0) if inter else d.to_numpy_arrays()[0]

    mix = arm.endswith("_mix")
    k = 0
    while state["used"] < budget:
        n = min(block, budget - state["used"])
        if arm.startswith("count"):
            site = min(sites, key=lambda s: (counts[s], sites.index(s)))
        else:
            X, Y = ds.to_numpy_arrays()
            preds = []
            for m in range(members):
                mdl = _fit_quick(ds, seed + 31 * m + 1000 * k, epochs)
                preds.append({st: mdl.forward(probes[st]).reshape(-1) for st in sites})
            scores = {st: float(np.mean(np.var(np.stack([p[st] for p in preds], axis=0), axis=0))) for st in sites}
            if arm.startswith("uncertainty_disc"):
                scores = {st: v / (1.0 + counts[st] / block) for st, v in scores.items()}
            site = max(scores, key=scores.get)
        chosen.append(site)
        n_broad = int(n * mix_frac) if mix else 0
        collect(site, n - n_broad, seed * 101 + k)
        if n_broad:
            collect_broad(n_broad, seed * 211 + k)
        k += 1
    return ds, chosen


def run_experiment_e005(seeds: List[int] = [42, 101, 7, 2026, 555], budget: int = 2400,
                        arms: Optional[List[str]] = None, broad_world: str = "phase6_world_base.json",
                        experiment_id: str = "CSP-P7-E005") -> Dict[str, Any]:
    """CSP-P7-E005: does a strengthened uncertainty signal (and/or mixed broad
    exploration) let Caspian's own selection match or beat the trivial count heuristic?
    Same seed data, budget, and swapped-world test as E004. Extra metric: R2 on ALL
    transitions of the swapped worlds (coverage), the cost E003 exposed for targeted
    collection."""
    arms = arms or ["count", "count_mix", "uncertainty_v2", "uncertainty_disc", "uncertainty_v2_mix"]
    t1_far, t3_near = _swapped_type_configs()
    t1_near, t3_far = _confounded_type_configs()
    swapped_cells = {(1, c) for c in [(3, 3), (3, 4), (4, 3), (4, 4)]} | {(3, c) for c in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    rows = []
    for s in seeds:
        row = {"seed": s}
        for arm in arms:
            ds, chosen = _run_e005_arm(arm, s, budget, broad_world=broad_world)
            model = _train_xy(ds, seed=s)
            sw = [_eval_on_world(model, c, seed=s + 900 + i)["r2_score"] for i, c in enumerate([t1_far, t3_near])]
            ma = [_eval_on_world(model, c, seed=s + 950 + i)["r2_score"] for i, c in enumerate([t1_near, t3_far])]
            sw_all = [_eval_all_on_world(model, c, seed=s + 980 + i) for i, c in enumerate([t1_far, t3_near])]
            top = max([chosen.count(x) for x in set(chosen)]) / max(1, len(chosen))
            row[arm] = {
                "swapped_r2": round(float(np.mean(sw)), 4),
                "matched_r2": round(float(np.mean(ma)), 4),
                "swapped_all_transitions_r2": round(float(np.mean(sw_all)), 4),
                "sites_covered": len(set(chosen)),
                "max_site_share": round(top, 3),
            }
        rows.append(row)
        print("E005 seed", s, {a: (row[a]["swapped_r2"], row[a]["sites_covered"]) for a in arms}, flush=True)
    summary = {}
    for arm in arms:
        g = lambda k: [r[arm][k] for r in rows]
        summary[arm] = {
            "swapped_r2_mean": round(float(np.mean(g("swapped_r2"))), 4), "swapped_r2_std": round(float(np.std(g("swapped_r2"))), 4),
            "swapped_r2_min": round(float(np.min(g("swapped_r2"))), 4),
            "matched_r2_mean": round(float(np.mean(g("matched_r2"))), 4),
            "swapped_all_transitions_r2_mean": round(float(np.mean(g("swapped_all_transitions_r2"))), 4),
            "mean_sites_covered": round(float(np.mean(g("sites_covered"))), 2),
            "seeds_swapped_r2_above_0.9": sum(x > 0.9 for x in g("swapped_r2")),
        }
    return {"experiment_id": experiment_id, "broad_world": broad_world, "title": "Strengthened Uncertainty Signal and Mixed Broad Exploration (follow-up to E004)",
            "seeds": seeds, "budget": budget, "arms": arms, "per_seed": rows, "summary": summary}


# ---------------------------------------------------------------------------
# CSP-P7-E006 / E007: noise robustness and a harder, larger confound
# (pre-registered success criteria are stored in each result under "preregistered")
# ---------------------------------------------------------------------------


def _clean_site_config(entity_type: int, delta: float, corner: Tuple[int, int], noise_std: float = 0.0) -> WorldConfig:
    """Single-entity site world with UNCLAMPED internal_state (wide bounds, start at 0) so that
    the interaction effect is linear and the ceiling-clamping confound found in E001 cannot
    recur; optional Gaussian outcome noise on interactions only."""
    cfg = copy.deepcopy(_load_world("phase7_world_base.json"))
    cfg.initial_entities = [Entity(entity_id=1, entity_type=entity_type, position=corner, hidden_state_delta=delta)]
    cfg.allowed_entity_positions = [corner]
    cfg.randomize_entity_positions = False
    cfg.initial_agent_pos = corner
    cfg.initial_internal_state = 0.0
    cfg.min_internal_state = -1000.0
    cfg.max_internal_state = 1000.0
    cfg.outcome_noise_std = noise_std
    return cfg


def _pooled_r2(model: PredictiveMLP, world_configs: List[WorldConfig], seed: int, num_steps: int = 800) -> float:
    """R2 pooled across several single-type test worlds. (Scoring each single-type world
    separately gives constant targets under unclamped linear outcomes, so R2 is degenerate
    -- caught in the first E006b/E007 run, where every score printed exactly 0.0.)"""
    Xs, Ys = [], []
    for i, cfg in enumerate(world_configs):
        X, Y = _interaction_xy(_collect_passive(cfg, seed=seed + i, num_steps=num_steps))
        Xs.append(X); Ys.append(Y)
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    return compute_all_metrics(model.forward(X), Y)["r2_score"]


def run_experiment_e006a(seed: int = 42, noise_levels: List[float] = [0.0, 5.0, 10.0, 20.0],
                         repeats: List[int] = [1, 2, 4, 8, 16], trials: int = 200) -> Dict[str, Any]:
    """CSP-P7-E006a: Repeated interventions under outcome noise. In each trial the agent runs
    the Section-35 loop (approach, interact, step away, return, interact ...) r times on one
    unknown entity and averages the r outcomes, then decides whether the entity is beneficial
    (mean outcome above the no-interaction baseline of -step_penalty) or harmful. Entities are
    type 1 (true outcome +9) and type 3 (true outcome -11), 50/50.
    PRE-REGISTERED: (i) at sigma=0 every r gives 100% decision accuracy; (ii) at sigma=10,
    accuracy at r=16 exceeds accuracy at r=1 by >= 10 points; (iii) the std of the repeat-mean
    shrinks roughly as 1/sqrt(r) (log-log slope within -0.5 +/- 0.15)."""
    rng = random.Random(seed)
    results = {}
    for sigma in noise_levels:
        per_r = {}
        for r in repeats:
            correct, means_by_type = 0, {1: [], 3: []}
            for t in range(trials):
                etype, delta = (1, 10.0) if rng.random() < 0.5 else (3, -10.0)
                cfg = _clean_site_config(etype, delta, (2, 2), sigma)
                cfg.allowed_entity_positions = None
                cfg.initial_agent_pos = (0, 0)
                cfg.max_timesteps = 200
                w = GridWorld(config=cfg)
                w.reset(seed=rng.randint(0, 10**6))
                pos = w._entities[0].position
                outs = []
                for _ in range(r):
                    res = _approach_and_interact(w, pos, max_steps=12)
                    if res is None or not res[2].get("interaction", False):
                        break
                    outs.append(res[1])
                    for _ in range(2):
                        w.step(_step_away(w._agent_pos, pos, cfg.grid_width, cfg.grid_height))
                if len(outs) < r:
                    continue
                m = float(np.mean(outs))
                means_by_type[etype].append(m)
                correct += int((m > -cfg.step_penalty) == (delta > 0))
            n = len(means_by_type[1]) + len(means_by_type[3])
            sd = float(np.mean([np.std(means_by_type[1]), np.std(means_by_type[3])])) if n > 3 else None
            per_r[str(r)] = {"decision_accuracy_pct": round(100.0 * correct / max(1, n), 2), "std_of_repeat_mean": round(sd, 4) if sd is not None else None, "n": n}
        results[str(sigma)] = per_r
    # log-log slope of std vs r at the largest noise level
    big = results[str(max(noise_levels))]
    xs = np.log([int(r) for r in repeats]); ys = np.log([big[str(r)]["std_of_repeat_mean"] for r in repeats])
    slope = float(np.polyfit(xs, ys, 1)[0])
    s10 = results.get("10.0")
    checks = {
        "i_sigma0_all_100pct": all(v["decision_accuracy_pct"] == 100.0 for v in results["0.0"].values()),
        "ii_sigma10_gain_ge_10pts": (s10["16"]["decision_accuracy_pct"] - s10["1"]["decision_accuracy_pct"] >= 10.0) if s10 else None,
        "iii_loglog_slope": round(slope, 3),
        "iii_slope_within_tolerance": abs(slope + 0.5) <= 0.15,
    }
    return {"experiment_id": "CSP-P7-E006a", "title": "Repeated Interventions Under Outcome Noise", "seed": seed, "trials_per_cell": trials,
            "noise_levels": noise_levels, "repeats": repeats, "results": results, "preregistered_checks": checks,
            "preregistered": "see docstring: (i) sigma0 100%; (ii) sigma10 r16-r1 >= +10 pts; (iii) std slope -0.5 +/- 0.15"}


def run_experiment_e006b(seeds: List[int] = [42, 101, 7, 2026, 555], noise_levels: List[float] = [0.0, 3.0, 6.0],
                         steps_per_region: int = 3000) -> Dict[str, Any]:
    """CSP-P7-E006b: the E002 confound test UNDER outcome noise. Training data are noisy; the
    model is scored against the NOISE-FREE truth on swapped-position worlds. Equal data volume.
    PRE-REGISTERED: at each sigma, active (varied-position) swapped R2 >= 0.8 in >= 4/5 seeds and
    mean active swapped R2 exceeds mean passive swapped R2 by >= 0.3."""
    rows = []
    near1 = [(0, 0), (0, 1), (1, 0), (1, 1)]; far = [(3, 3), (3, 4), (4, 3), (4, 4)]
    corners = [(0, 0), (4, 4), (0, 4), (4, 0)]

    def noiseless_world(t, d, region):
        c = _clean_site_config(t, d, region[0], 0.0)
        c.allowed_entity_positions = region; c.randomize_entity_positions = True
        return c

    for sigma in noise_levels:
        for s in seeds:
            c1n = noiseless_world(1, 10.0, near1); c3f = noiseless_world(3, -10.0, far)
            c1f = noiseless_world(1, 10.0, far); c3n = noiseless_world(3, -10.0, near1)
            n1 = _clean_site_config(1, 10.0, (1, 1), sigma); n1.allowed_entity_positions = near1; n1.randomize_entity_positions = True
            n3 = _clean_site_config(3, -10.0, (3, 3), sigma); n3.allowed_entity_positions = far; n3.randomize_entity_positions = True; n3.initial_agent_pos = (4, 4)
            passive = ExperienceDataset()
            passive.add_transitions(_collect_passive(n1, seed=s, num_steps=steps_per_region).transitions)
            passive.add_transitions(_collect_passive(n3, seed=s + 1, num_steps=steps_per_region).transitions)
            active = ExperienceDataset()
            for et, d, off in [(1, 10.0, 0), (3, -10.0, 10)]:
                for i, cn in enumerate(corners):
                    active.add_transitions(_collect_passive(_clean_site_config(et, d, cn, sigma), seed=s + off + i, num_steps=steps_per_region // 4).transitions)
            row = {"noise": sigma, "seed": s}
            for name, ds in [("passive", passive), ("active", active)]:
                m, _ = _train_on_dataset(ds, seed=s)
                row[f"{name}_matched_r2"] = round(float(_pooled_r2(m, [c1n, c3f], seed=s + 700)), 4)
                row[f"{name}_swapped_r2"] = round(float(_pooled_r2(m, [c1f, c3n], seed=s + 800)), 4)
            rows.append(row)
        print("E006b sigma", sigma, "done", flush=True)
    summary = {}
    for sigma in noise_levels:
        r = [x for x in rows if x["noise"] == sigma]
        a = [x["active_swapped_r2"] for x in r]; p = [x["passive_swapped_r2"] for x in r]
        summary[str(sigma)] = {"active_swapped_r2_mean": round(float(np.mean(a)), 4), "passive_swapped_r2_mean": round(float(np.mean(p)), 4),
                               "active_seeds_ge_0.8": sum(v >= 0.8 for v in a), "gap": round(float(np.mean(a) - np.mean(p)), 4),
                               "prereg_met": bool(sum(v >= 0.8 for v in a) >= 4 and np.mean(a) - np.mean(p) >= 0.3)}
    return {"experiment_id": "CSP-P7-E006b", "title": "Confound Recovery Under Outcome Noise (scored vs noise-free truth)",
            "seeds": seeds, "noise_levels": noise_levels, "steps_per_region": steps_per_region, "per_run": rows, "summary": summary,
            "preregistered": "per sigma: active swapped R2 >= 0.8 in >= 4/5 seeds AND mean(active)-mean(passive) >= 0.3"}


_E007_TYPES = {1: 6.0, 2: -6.0, 3: 3.0, 4: -3.0}
_E007_CORNERS = [(0, 0), (0, 4), (4, 4), (4, 0)]


def _e007_sites():
    return [(t, c) for t in _E007_TYPES for c in _E007_CORNERS]


def _e007_collect(site, n, seed, noise=0.0):
    t, c = site
    return _collect_passive(_clean_site_config(t, _E007_TYPES[t], c, noise), seed=seed, num_steps=n)


def _e007_arm(arm: str, seed: int, budget: int, seed_block: int = 300, block: int = 300, noise: float = 0.0):
    """4 types, 16 (type x corner) candidate sites. Seed data: type k only at corner k (a perfect
    4-way type/position confound). Arms spend the rest of the budget: confounded (never leaves),
    random site, count (least-sampled site)."""
    rng = random.Random(seed)
    sites = _e007_sites()
    seed_sites = [(t, _E007_CORNERS[i]) for i, t in enumerate(_E007_TYPES)]
    ds = ExperienceDataset(); counts = {s_: 0 for s_ in sites}; used = 0
    for i, st in enumerate(seed_sites):
        ds.add_transitions(_e007_collect(st, seed_block, seed * 7 + i, noise).transitions); counts[st] += seed_block; used += seed_block
    k = 0
    while used < budget:
        n = min(block, budget - used)
        if arm == "confounded": site = seed_sites[k % 4]
        elif arm == "random": site = rng.choice(sites)
        elif arm == "count": site = min(sites, key=lambda s_: (counts[s_], sites.index(s_)))
        else: raise ValueError(arm)
        ds.add_transitions(_e007_collect(site, n, seed * 101 + k, noise).transitions); counts[site] += n; used += n; k += 1
    return ds


def run_experiment_e007(seeds: List[int] = [42, 101, 7, 2026, 555], budget: int = 4800, noise_levels: List[float] = [0.0, 3.0]) -> Dict[str, Any]:
    """CSP-P7-E007: a harder confound. 4 types (effects +6, -6, +3, -3), a perfect 4-way
    type/corner confound in the seed data, 16 candidate sites, and test worlds where every type
    is moved to a DIFFERENT corner (cyclic shift). Scored vs noise-free truth, equal budgets.
    PRE-REGISTERED: count and random each reach swapped R2 >= 0.8 in >= 4/5 seeds at both noise
    levels, and exceed 'confounded' by >= 0.5 on average."""
    arms = ["confounded", "random", "count"]
    rows = []
    for sigma in noise_levels:
        for s in seeds:
            row = {"noise": sigma, "seed": s}
            for arm in arms:
                ds = _e007_arm(arm, s, budget, noise=sigma)
                model = _train_xy(ds, seed=s)
                sw_cfgs = [_clean_site_config(t, _E007_TYPES[t], _E007_CORNERS[(i + 1) % 4], 0.0) for i, t in enumerate(_E007_TYPES)]
                ma_cfgs = [_clean_site_config(t, _E007_TYPES[t], _E007_CORNERS[i], 0.0) for i, t in enumerate(_E007_TYPES)]
                row[arm] = {"swapped_r2": round(float(_pooled_r2(model, sw_cfgs, seed=s + 600)), 4), "matched_r2": round(float(_pooled_r2(model, ma_cfgs, seed=s + 650)), 4)}
            rows.append(row)
        print("E007 sigma", sigma, "done", flush=True)
    summary = {}
    for sigma in noise_levels:
        r = [x for x in rows if x["noise"] == sigma]
        summary[str(sigma)] = {a: {"swapped_r2_mean": round(float(np.mean([x[a]["swapped_r2"] for x in r])), 4),
                                   "swapped_r2_min": round(float(np.min([x[a]["swapped_r2"] for x in r])), 4),
                                   "seeds_ge_0.8": sum(x[a]["swapped_r2"] >= 0.8 for x in r),
                                   "matched_r2_mean": round(float(np.mean([x[a]["matched_r2"] for x in r])), 4)} for a in arms}
        c = summary[str(sigma)]
        summary[str(sigma)]["prereg_met"] = bool(all(c[a]["seeds_ge_0.8"] >= 4 and c[a]["swapped_r2_mean"] - c["confounded"]["swapped_r2_mean"] >= 0.5 for a in ["random", "count"]))
    return {"experiment_id": "CSP-P7-E007", "title": "Harder Confound: 4 Types, 16 Sites, Cyclic-Shift Test, With and Without Noise",
            "seeds": seeds, "budget": budget, "noise_levels": noise_levels, "per_run": rows, "summary": summary,
            "preregistered": "random & count: swapped R2 >= 0.8 in >=4/5 seeds at each sigma AND >= confounded + 0.5"}


def run_all_phase7_experiments(output_dir: str = "logs") -> Dict[str, Any]:
    """Execute all Phase 7 experiments sequentially and persist results."""
    import os

    print("Running CSP-P7-E001 (Repeated-Intervention Consistency)...")
    e001 = run_experiment_e001()

    print("Running CSP-P7-E002 (Confounded Observation vs. Active Intervention)...")
    e002 = run_experiment_e002()

    print("Running CSP-P7-E002 Stability (multi-seed hardening)...")
    e002_stability = run_experiment_e002_stability()

    print("Running CSP-P7-E003 (Active vs. Passive Sample Efficiency, equal budgets)...")
    e003 = run_experiment_e003()

    print("Running CSP-P7-E004 (Autonomous intervention selection)...")
    e004 = run_experiment_e004()

    print("Running CSP-P7-E005 (Strengthened uncertainty + mixed exploration)...")
    e005 = run_experiment_e005()

    all_results = {
        "phase": "Phase 7 — Causal Experimentation",
        "experiments": {
            "CSP-P7-E001": e001,
            "CSP-P7-E002": e002,
            "CSP-P7-E002-STABILITY": e002_stability,
            "CSP-P7-E003": e003,
            "CSP-P7-E004": e004,
            "CSP-P7-E005": e005,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase7_all_experiments_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Phase 7 experiments completed and saved to: {out_file}")
    return all_results


if __name__ == "__main__":
    run_all_phase7_experiments()
