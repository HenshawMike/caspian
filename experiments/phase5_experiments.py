"""Experimental Suite for Project Caspian (Phase 5 — Emergent Categories).

Addresses Sections 32-33 of the Development Plan. Caspian is never given entity
categories like "beneficial" or "harmful" -- entity_type is just a neutral
numeric code (Section 8, the semantic firewall). This suite asks whether the
model's *internal representation* nonetheless organizes entities by their
functional consequence (the sign of the interaction outcome), and tests that
against several of Section 33's measurable criteria rather than asserting it:

- CSP-P5-E001 (Predictiveness): does entity_type actually carry information the
  model needs? (ablation: zero it out, measure the damage.)
- CSP-P5-E002 (Generalization + the core clustering test): do NOVEL entity_type
  codes, never seen in training, land in the correct functional region of the
  latent space -- including one deliberately adjacent (in raw code) to the
  WRONG function, so raw code-proximity cannot explain a positive result?
  Also checks the raw input space shows no such structure, to rule out "the
  inputs already looked similar" as the explanation. Folds in a lightweight
  behavioral-relevance check (does the model's own prediction sign match the
  novel type's true function).
- CSP-P5-E003 (Stability): does the clustering result reproduce across
  independently-initialized training runs, or is it a one-seed artifact?
- CSP-P5-E004 (Generalization, follow-up): with a one-hot entity_type encoding
  (removing the ordinality confound raw-scalar codes introduce), can the
  network discover shared function across four TRAINED types it is never told
  are related?
- CSP-P5-E005 (Compression): does the latent code represent function more
  *compactly* than the raw input does -- i.e. is within-function variance
  small relative to between-function variance, more so than in the raw input,
  and does the latent space actually use fewer effective dimensions than its
  nominal width?
- CSP-P5-E006 (Intervention Sensitivity): if we intervene on the actual
  environmental property that causes a type's outcome (flip its hidden
  hidden_state_delta sign) and retrain, does the type's latent representation
  move to the opposite functional cluster? And, as a control, does changing
  something cosmetic/non-causal (entity_id only) leave it where it was?
- CSP-P5-E007 (True Zero-Shot Generalization, follow-up to E002/E003/E004):
  E002/E003/E004 all found the same structural limit -- no fixed-vocabulary
  category encoding (raw-scalar or one-hot) can generalize to an entity_type
  never seen in training, because the model's only handle on "type" is an
  opaque index with zero learned weight for an unseen code. E007 tests
  whether that limit is specific to index-based encodings by dropping
  entity_type from the input entirely and replacing it with a neutral
  CONTINUOUS property with a consistent hidden rule relating it to outcome.
  Both E004 and E007 have multi-seed stability companions
  (run_experiment_e004_stability, run_experiment_e007_stability), hardening
  results that were originally reported on a single seed.

All six of Section 33's measurable category-discovery criteria
(Predictiveness, Stability, Generalization, Compression, Intervention
Sensitivity, Behavioral Relevance) are now covered by at least one experiment
in this suite, each with an executed, logged result. Behavioral Relevance is
folded into E002/E007 (does the model's own prediction sign match a novel
type's true function) rather than given its own experiment ID, since it
falls out of those tests for free.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.world import WorldConfig
from environment.actions import Action
from models.mlp import PredictiveMLP
from learning.collector import TrajectoryCollector
from learning.trainer import PredictiveTrainer
from evaluation.metrics import compute_all_metrics, compute_cosine_similarity

CONFIG_DIR = Path(__file__).parent.parent / "configs"
MAX_ENTITIES = 5
ENTITY_TYPE_INDICES = [6 + i * 5 + 1 for i in range(MAX_ENTITIES)]  # per-slot offset of entity_type in the 37-dim vector

# --- One-hot entity_type encoding (Phase 5 v2 fix) ---
# Kept local to this file rather than added to the shared encode_state_action() used by
# every other phase: this is a targeted fix for the ordinality problem CSP-P5-E002/E003
# diagnosed, not a change the rest of the project has asked for or been re-validated
# against.
TYPE_VOCAB = [1, 2, 3, 4]


def _encode_onehot_type(obs, action, type_vocab=TYPE_VOCAB, max_entities: int = MAX_ENTITIES, num_actions: int = 6) -> np.ndarray:
    """Same information as encode_state_action(), except entity_type is one-hot (no
    ordinal relationship between codes) instead of a raw scalar.
    """
    from learning.dataset import one_hot_encode_action

    type_to_idx = {t: i for i, t in enumerate(type_vocab)}
    v = len(type_vocab)

    global_vec = [
        float(obs.agent_position[0]), float(obs.agent_position[1]),
        1.0 if obs.collision else 0.0, float(obs.internal_state), float(obs.timestep),
        float(obs.previous_action if obs.previous_action is not None else -1.0),
    ]

    entity_block: List[float] = []
    for i in range(max_entities):
        if i < len(obs.entities):
            e = obs.entities[i]
            onehot = [0.0] * v
            if e.entity_type in type_to_idx:
                onehot[type_to_idx[e.entity_type]] = 1.0
            entity_block.extend([float(e.entity_id)] + onehot + [float(e.relative_x), float(e.relative_y), float(e.distance)])
        else:
            entity_block.extend([0.0] * (1 + v + 3))

    act_vec = one_hot_encode_action(action, num_actions=num_actions)
    return np.concatenate([np.array(global_vec, dtype=np.float64), np.array(entity_block, dtype=np.float64), act_vec])


def _load_world(name: str) -> WorldConfig:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return WorldConfig.from_dict(json.load(f))


def _zero_entity_type(X: np.ndarray) -> np.ndarray:
    """Ablation: strip entity_type from the input, leaving position/distance/action intact."""
    X_ablated = X.copy()
    X_ablated[:, ENTITY_TYPE_INDICES] = 0.0
    return X_ablated


def _train_model(world_config: WorldConfig, seed: int, num_steps: int = 6000, ablate_type: bool = False):
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed)
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)
    X_train, Y_train = train_ds.to_numpy_arrays()
    X_val, Y_val = val_ds.to_numpy_arrays()

    if ablate_type:
        X_train, X_val = _zero_entity_type(X_train), _zero_entity_type(X_val)

    model = PredictiveMLP(input_dim=X_train.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit((X_train, Y_train), val_data=(X_val, Y_val))
    return model, history, (X_val, Y_val)


def run_experiment_e001(seed: int = 42) -> Dict[str, Any]:
    """CSP-P5-E001: Predictiveness -- does entity_type carry information the model needs?"""
    train_world = _load_world("phase5_world_train.json")

    full_model, full_history, (X_val, Y_val) = _train_model(train_world, seed, ablate_type=False)
    full_metrics = compute_all_metrics(full_model.forward(X_val), Y_val)

    ablated_model, ablated_history, (X_val_abl, Y_val_abl) = _train_model(train_world, seed, ablate_type=True)
    ablated_metrics = compute_all_metrics(ablated_model.forward(X_val_abl), Y_val_abl)

    return {
        "experiment_id": "CSP-P5-E001",
        "title": "Predictiveness: Does entity_type Carry Necessary Information?",
        "seed": seed,
        "full_model_metrics": full_metrics,
        "type_ablated_model_metrics": ablated_metrics,
        "mse_increase_pct_from_ablation": round(
            100.0 * (ablated_metrics["mse"] - full_metrics["mse"]) / (full_metrics["mse"] + 1e-10), 2
        ),
        "type_is_necessary": ablated_metrics["mse"] > full_metrics["mse"] * 1.5,
    }


def _dataset_to_onehot_arrays(dataset) -> "Tuple[np.ndarray, np.ndarray]":
    """Rebuild (X, Y) from a collected ExperienceDataset using the one-hot type encoding,
    instead of the dataset's own to_numpy_arrays() (which uses the raw-scalar encoding).
    """
    x_list = [_encode_onehot_type(t.obs, t.action) for t in dataset.transitions]
    y_list = [t.to_target_vector() for t in dataset.transitions]
    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)


def _train_model_onehot(world_config: WorldConfig, seed: int, num_steps: int = 6000):
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed)
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)
    X_train, Y_train = _dataset_to_onehot_arrays(train_ds)
    X_val, Y_val = _dataset_to_onehot_arrays(val_ds)

    model = PredictiveMLP(input_dim=X_train.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit((X_train, Y_train), val_data=(X_val, Y_val))
    return model, history, (X_val, Y_val)


def _collect_latent_states_onehot_by_type(model: PredictiveMLP, world_config: WorldConfig, seed: int, num_samples: int = 300):
    """One-hot-encoding counterpart to _collect_latent_states_by_type()."""
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_samples * 4, interaction_prob=0.5, seed=seed)
    X, Y = _dataset_to_onehot_arrays(dataset)

    interact_idx = int(Action.INTERACT)
    action_onehot = X[:, -6:]
    is_interact = np.argmax(action_onehot, axis=1) == interact_idx
    X_interact, Y_interact = X[is_interact], Y[is_interact]

    latents = model.get_latent_state(X_interact)
    return X_interact, latents, Y_interact


def run_experiment_e004(seed: int = 42) -> Dict[str, Any]:
    """CSP-P5-E004: Does one-hot entity_type (no ordinal relationship between codes)
    let the network discover shared function across nominally-different trained types,
    where the raw-scalar encoding could not?

    Redesigned test (see Section 4 of the report): instead of a truly novel, never-seen
    code -- which a one-hot/embedding encoding cannot generalize to by construction, since
    an untrained one-hot column carries no information at all -- all four types (two
    beneficial, two harmful) are now present during training, each in its own disjoint
    one-hot slot. The network is never told which pairs share a function; it can only
    discover that from interaction outcomes. Both encodings are trained on the IDENTICAL
    world/data distribution for a direct, controlled comparison.
    """
    train_world = _load_world("phase5_world_train_v2.json")

    # Single-type configs for isolated per-type latent collection (same pattern as E002).
    import copy
    type_configs = {}
    for t, delta in [(1, 10.0), (2, -10.0), (3, 10.0), (4, -10.0)]:
        cfg = copy.deepcopy(train_world)
        cfg.entity_type_pool = [{"entity_type": t, "hidden_state_delta": delta, "is_interactive": True, "is_blocking": False}]
        type_configs[t] = cfg

    PAIR_SPEC = [
        ("type1_vs_type3_SAME_FUNC", 1, 3), ("type2_vs_type4_SAME_FUNC", 2, 4),
        ("type1_vs_type2_DIFF_FUNC", 1, 2), ("type2_vs_type3_DIFF_FUNC", 2, 3),
        ("type3_vs_type4_DIFF_FUNC", 3, 4), ("type1_vs_type4_DIFF_FUNC", 1, 4),
    ]

    def sims_for(latent_means):
        return {name: round(compute_cosine_similarity(latent_means[a], latent_means[b]), 4) for name, a, b in PAIR_SPEC}

    def euclid_for(latent_means):
        return {name: round(float(np.linalg.norm(latent_means[a] - latent_means[b])), 4) for name, a, b in PAIR_SPEC}

    # --- Raw-scalar encoding, trained on the SAME v2 world (control) ---
    raw_model, raw_history, _ = _train_model(train_world, seed)
    raw_means = {}
    for t, cfg in type_configs.items():
        _, L, _ = _collect_latent_states_by_type(raw_model, cfg, seed)
        raw_means[t] = np.mean(L, axis=0)
    raw_sims = sims_for(raw_means)
    raw_same = float(np.mean([raw_sims[k] for k in raw_sims if "SAME_FUNC" in k]))
    raw_diff = float(np.mean([raw_sims[k] for k in raw_sims if "DIFF_FUNC" in k]))
    raw_eucl = euclid_for(raw_means)
    raw_eucl_same = float(np.mean([raw_eucl[k] for k in raw_eucl if "SAME_FUNC" in k]))
    raw_eucl_diff = float(np.mean([raw_eucl[k] for k in raw_eucl if "DIFF_FUNC" in k]))

    # --- One-hot encoding, trained on the IDENTICAL world/data distribution ---
    onehot_model, onehot_history, _ = _train_model_onehot(train_world, seed)
    onehot_means = {}
    for t, cfg in type_configs.items():
        _, L, _ = _collect_latent_states_onehot_by_type(onehot_model, cfg, seed)
        onehot_means[t] = np.mean(L, axis=0)
    onehot_sims = sims_for(onehot_means)
    onehot_same = float(np.mean([onehot_sims[k] for k in onehot_sims if "SAME_FUNC" in k]))
    onehot_diff = float(np.mean([onehot_sims[k] for k in onehot_sims if "DIFF_FUNC" in k]))
    onehot_eucl = euclid_for(onehot_means)
    onehot_eucl_same = float(np.mean([onehot_eucl[k] for k in onehot_eucl if "SAME_FUNC" in k]))
    onehot_eucl_diff = float(np.mean([onehot_eucl[k] for k in onehot_eucl if "DIFF_FUNC" in k]))

    return {
        "experiment_id": "CSP-P5-E004",
        "title": "One-Hot Encoding Fix: Can the Network Discover Shared Function Across Trained Types?",
        "seed": seed,
        "raw_scalar_control": {
            "cosine_similarities": raw_sims,
            "cosine_same_function_avg": round(raw_same, 4),
            "cosine_diff_function_avg": round(raw_diff, 4),
            "euclidean_distances": raw_eucl,
            "euclidean_same_function_avg": round(raw_eucl_same, 4),
            "euclidean_diff_function_avg": round(raw_eucl_diff, 4),
            "euclidean_diff_to_same_ratio": round(raw_eucl_diff / (raw_eucl_same + 1e-10), 2),
            "functional_clustering_euclidean": raw_eucl_same < raw_eucl_diff,
        },
        "onehot_fix": {
            "cosine_similarities": onehot_sims,
            "cosine_same_function_avg": round(onehot_same, 4),
            "cosine_diff_function_avg": round(onehot_diff, 4),
            "euclidean_distances": onehot_eucl,
            "euclidean_same_function_avg": round(onehot_eucl_same, 4),
            "euclidean_diff_function_avg": round(onehot_eucl_diff, 4),
            "euclidean_diff_to_same_ratio": round(onehot_eucl_diff / (onehot_eucl_same + 1e-10), 2),
            "functional_clustering_euclidean": onehot_eucl_same < onehot_eucl_diff,
        },
        "onehot_training_history": {"epochs_run": onehot_history["epochs_run"], "final_train_loss": onehot_history["final_train_loss"]},
    }


def _collect_latent_states_by_type(model: PredictiveMLP, world_config: WorldConfig, seed: int, num_samples: int = 300):
    """Collect (raw_input, latent_state, true_delta) triples for INTERACT actions on this
    world's entity type, at varied agent/entity positions.
    """
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_samples * 4, interaction_prob=0.5, seed=seed)
    X, Y = dataset.to_numpy_arrays()

    # Keep only INTERACT-action rows (last 6 cols are the one-hot action; INTERACT index per Action enum).
    interact_idx = int(Action.INTERACT)
    action_onehot = X[:, -6:]
    is_interact = np.argmax(action_onehot, axis=1) == interact_idx
    X_interact, Y_interact = X[is_interact], Y[is_interact]

    latents = model.get_latent_state(X_interact)
    return X_interact, latents, Y_interact


def run_experiment_e002(seed: int = 42) -> Dict[str, Any]:
    """CSP-P5-E002: Generalization -- do novel entity-type codes land in the correct
    functional region of latent space, and does the raw input alone already show this
    (it shouldn't -- raw vectors for different types differ in exactly one scalar)?
    """
    train_world = _load_world("phase5_world_train.json")
    novel_beneficial_world = _load_world("phase5_world_novel_beneficial.json")  # code=3, +10 (like type 1)
    novel_harmful_world = _load_world("phase5_world_novel_harmful.json")  # code=4, -10 (like type 2)

    model, history, _ = _train_model(train_world, seed)

    # Collect latent + raw representations for each of the 4 types (2 trained, 2 never seen).
    def collect_for(world_config):
        return _collect_latent_states_by_type(model, world_config, seed)

    # Trained-pool worlds only ever contain ONE of the two types per episode (sampled from
    # entity_type_pool), so we can't isolate type 1 vs type 2 from phase5_world_train.json
    # directly -- build single-type worlds mirroring the training templates for this analysis.
    type1_world = _load_world("phase5_world_novel_beneficial.json")  # reuse structure, override type/delta below
    import copy
    type1_cfg = copy.deepcopy(type1_world)
    type1_cfg.entity_type_pool = [{"entity_type": 1, "hidden_state_delta": 10.0, "is_interactive": True, "is_blocking": False}]
    type2_cfg = copy.deepcopy(type1_world)
    type2_cfg.entity_type_pool = [{"entity_type": 2, "hidden_state_delta": -10.0, "is_interactive": True, "is_blocking": False}]

    X1, L1, Y1 = collect_for(type1_cfg)          # trained, beneficial
    X2, L2, Y2 = collect_for(type2_cfg)          # trained, harmful
    X3, L3, Y3 = collect_for(novel_beneficial_world)  # NOVEL code, beneficial (adjacent code to type 2!)
    X4, L4, Y4 = collect_for(novel_harmful_world)     # NOVEL code, harmful (far code from type 2)

    def mean_vec(arr):
        return np.mean(arr, axis=0) if len(arr) > 0 else None

    latent_means = {"type1": mean_vec(L1), "type2": mean_vec(L2), "type3_novel": mean_vec(L3), "type4_novel": mean_vec(L4)}
    raw_means = {"type1": mean_vec(X1), "type2": mean_vec(X2), "type3_novel": mean_vec(X3), "type4_novel": mean_vec(X4)}

    def sims(means):
        pairs = [
            ("type1_vs_type3_novel_SAME_FUNC", "type1", "type3_novel"),
            ("type2_vs_type4_novel_SAME_FUNC", "type2", "type4_novel"),
            ("type1_vs_type2_DIFF_FUNC_adjacent_code", "type1", "type2"),
            ("type2_vs_type3_novel_DIFF_FUNC_adjacent_code", "type2", "type3_novel"),
            ("type3_novel_vs_type4_novel_DIFF_FUNC_adjacent_code", "type3_novel", "type4_novel"),
            ("type1_vs_type4_novel_DIFF_FUNC", "type1", "type4_novel"),
        ]
        return {name: round(compute_cosine_similarity(means[a], means[b]), 4) for name, a, b in pairs}

    latent_sims = sims(latent_means)
    raw_sims = sims(raw_means)

    same_func_keys = [k for k in latent_sims if "SAME_FUNC" in k]
    diff_func_keys = [k for k in latent_sims if "DIFF_FUNC" in k]
    latent_same_avg = float(np.mean([latent_sims[k] for k in same_func_keys]))
    latent_diff_avg = float(np.mean([latent_sims[k] for k in diff_func_keys]))
    raw_same_avg = float(np.mean([raw_sims[k] for k in same_func_keys]))
    raw_diff_avg = float(np.mean([raw_sims[k] for k in diff_func_keys]))

    # Lightweight behavioral-relevance check: does the model's own PREDICTED sign match
    # the true function for the novel types it never trained on?
    pred3 = model.forward(X3)
    pred4 = model.forward(X4)
    behavioral_check = {
        "novel_beneficial_predicted_positive_rate": round(float(np.mean(pred3 > 0)), 3),
        "novel_harmful_predicted_negative_rate": round(float(np.mean(pred4 < 0)), 3),
    }

    return {
        "experiment_id": "CSP-P5-E002",
        "title": "Generalization: Do Novel Entity-Type Codes Cluster by Function, Not by Code?",
        "seed": seed,
        "latent_space_similarities": latent_sims,
        "raw_input_similarities": raw_sims,
        "latent_same_function_avg_similarity": round(latent_same_avg, 4),
        "latent_diff_function_avg_similarity": round(latent_diff_avg, 4),
        "raw_same_function_avg_similarity": round(raw_same_avg, 4),
        "raw_diff_function_avg_similarity": round(raw_diff_avg, 4),
        "functional_clustering_in_latent_space": latent_same_avg > latent_diff_avg,
        "raw_input_shows_no_functional_structure": abs(raw_same_avg - raw_diff_avg) < 0.05,
        "behavioral_relevance_check": behavioral_check,
        "training_history": {"epochs_run": history["epochs_run"], "final_train_loss": history["final_train_loss"]},
    }


def run_experiment_e003(seeds: List[int] = [42, 101, 7, 2026, 555]) -> Dict[str, Any]:
    """CSP-P5-E003: Stability -- does the clustering result reproduce across independent
    training runs, or is it a one-seed artifact?
    """
    per_seed = []
    for s in seeds:
        e002 = run_experiment_e002(seed=s)
        per_seed.append({
            "seed": s,
            "latent_same_function_avg_similarity": e002["latent_same_function_avg_similarity"],
            "latent_diff_function_avg_similarity": e002["latent_diff_function_avg_similarity"],
            "functional_clustering_in_latent_space": e002["functional_clustering_in_latent_space"],
        })

    clustering_rate = 100.0 * sum(r["functional_clustering_in_latent_space"] for r in per_seed) / len(per_seed)

    return {
        "experiment_id": "CSP-P5-E003",
        "title": "Stability Across Independent Training Runs",
        "seeds": seeds,
        "per_seed_results": per_seed,
        "clustering_reproduced_rate_pct": round(clustering_rate, 1),
        "mean_same_function_similarity": round(float(np.mean([r["latent_same_function_avg_similarity"] for r in per_seed])), 4),
        "mean_diff_function_similarity": round(float(np.mean([r["latent_diff_function_avg_similarity"] for r in per_seed])), 4),
    }


def _compute_variance_ratio(samples_by_type: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Fisher-style compactness: between-type variance divided by (mean) within-type
    variance, averaged over dimensions. Higher means samples sharing a function sit in a
    tighter cluster relative to how far apart the functional groups are -- i.e. function
    is represented more efficiently, not just "present somewhere" in the vector.
    """
    type_means = {t: np.mean(S, axis=0) for t, S in samples_by_type.items()}

    within_per_type = [float(np.mean(np.var(S, axis=0))) for S in samples_by_type.values()]
    within_var = float(np.mean(within_per_type))

    between_var = float(np.mean(np.var(np.stack(list(type_means.values())), axis=0)))

    return {
        "within_type_variance": round(within_var, 6),
        "between_type_variance": round(between_var, 6),
        "compression_ratio": round(between_var / (within_var + 1e-10), 4),
    }


def _participation_ratio(samples: np.ndarray) -> float:
    """Effective number of dimensions carrying variance: (sum of eigenvalues)^2 / (sum of
    squared eigenvalues) of the sample covariance matrix. A representation that collapses
    most of its variance onto a few directions has a participation ratio well below its
    nominal dimensionality -- i.e. it is compressed, not just smaller on paper.
    """
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s1, s2 = float(np.sum(eigvals)), float(np.sum(eigvals ** 2))
    return (s1 ** 2) / (s2 + 1e-10)


def run_experiment_e005(seed: int = 42) -> Dict[str, Any]:
    """CSP-P5-E005: Compression -- is function represented more compactly in the latent
    code than in the raw input? Two independent measures, both on the one-hot model/world
    (the encoding E004 showed actually discovers shared function):

    1. Compression ratio (between-function / within-function variance): should be higher
       in latent space than in the raw input if the model is summarizing "what function
       this is" rather than just carrying the input's own structure forward.
    2. Participation ratio (effective dimensionality): the latent code should use fewer
       effective dimensions, relative to its nominal width, than the raw input does if it
       has actually compressed the input into a smaller functional description.
    """
    import copy

    train_world = _load_world("phase5_world_train_v2.json")
    model, history, _ = _train_model_onehot(train_world, seed)

    type_configs = {}
    for t, delta in [(1, 10.0), (2, -10.0), (3, 10.0), (4, -10.0)]:
        cfg = copy.deepcopy(train_world)
        cfg.entity_type_pool = [{"entity_type": t, "hidden_state_delta": delta, "is_interactive": True, "is_blocking": False}]
        type_configs[t] = cfg

    latent_by_type, raw_by_type = {}, {}
    for t, cfg in type_configs.items():
        X, L, _ = _collect_latent_states_onehot_by_type(model, cfg, seed, num_samples=300)
        latent_by_type[t] = L
        raw_by_type[t] = X

    latent_stats = _compute_variance_ratio(latent_by_type)
    raw_stats = _compute_variance_ratio(raw_by_type)

    all_latent = np.concatenate(list(latent_by_type.values()), axis=0)
    all_raw = np.concatenate(list(raw_by_type.values()), axis=0)
    latent_pr = _participation_ratio(all_latent)
    raw_pr = _participation_ratio(all_raw)

    return {
        "experiment_id": "CSP-P5-E005",
        "title": "Compression: Is Function Represented More Compactly in the Latent Code Than in the Raw Input?",
        "seed": seed,
        "latent_space": {
            **latent_stats,
            "nominal_dim": int(all_latent.shape[1]),
            "participation_ratio": round(latent_pr, 2),
            "relative_effective_dim": round(latent_pr / all_latent.shape[1], 4),
        },
        "raw_input_space": {
            **raw_stats,
            "nominal_dim": int(all_raw.shape[1]),
            "participation_ratio": round(raw_pr, 2),
            "relative_effective_dim": round(raw_pr / all_raw.shape[1], 4),
        },
        "compression_ratio_improvement_factor": round(latent_stats["compression_ratio"] / (raw_stats["compression_ratio"] + 1e-10), 2),
        "latent_shows_greater_functional_compression": latent_stats["compression_ratio"] > raw_stats["compression_ratio"],
        "training_history": {"epochs_run": history["epochs_run"], "final_train_loss": history["final_train_loss"]},
    }


def _build_intervention_world(flip_type: int = 2) -> WorldConfig:
    """Same world as phase5_world_train_v2, except `flip_type`'s hidden_state_delta sign
    is reversed -- a genuine intervention on the environmental property that determines
    its function (Section 33), holding its type code and every other property fixed.
    """
    base = _load_world("phase5_world_train_v2.json")
    new_pool = []
    for template in base.entity_type_pool:
        t = dict(template)
        if int(t["entity_type"]) == flip_type:
            t["hidden_state_delta"] = -float(t["hidden_state_delta"])
        new_pool.append(t)
    base.entity_type_pool = new_pool
    return base


def _build_cosmetic_control_world(new_entity_id: int = 99) -> WorldConfig:
    """Same world as phase5_world_train_v2, except the physical entity's entity_id is
    renamed. entity_id is arbitrary bookkeeping (Section 8 semantic firewall) with no
    causal bearing on any type's function -- the control for E006: a change to the world
    config that should NOT move the functional clustering, unlike the real intervention.
    """
    import dataclasses

    base = _load_world("phase5_world_train_v2.json")
    base.initial_entities = [dataclasses.replace(e, entity_id=new_entity_id) for e in base.initial_entities]
    return base


def _single_type_config(base_world: WorldConfig, entity_type: int, hidden_state_delta: float) -> WorldConfig:
    import copy

    cfg = copy.deepcopy(base_world)
    cfg.entity_type_pool = [{
        "entity_type": entity_type, "hidden_state_delta": hidden_state_delta,
        "is_interactive": True, "is_blocking": False,
    }]
    return cfg


def run_experiment_e006(seed: int = 42) -> Dict[str, Any]:
    """CSP-P5-E006: Intervention Sensitivity -- if we intervene on the actual causal
    property behind type 2's function (flip its hidden_state_delta sign, making it
    beneficial like types 1/3 instead of harmful like type 4) and retrain, does type 2's
    latent representation move to the opposite functional cluster? And, as a control,
    does a cosmetic, non-causal change (renaming entity_id, function unchanged) leave it
    where it was? Without the control, "the representation changed" would be a weak
    claim -- retraining on ANY modified world might reshuffle it for unrelated reasons.
    """
    baseline_world = _load_world("phase5_world_train_v2.json")
    intervention_world = _build_intervention_world(flip_type=2)       # type 2 now +10 (beneficial)
    control_world = _build_cosmetic_control_world(new_entity_id=99)   # type 2 still -10 (harmful)

    baseline_model, baseline_hist, _ = _train_model_onehot(baseline_world, seed)
    intervention_model, intervention_hist, _ = _train_model_onehot(intervention_world, seed)
    control_model, control_hist, _ = _train_model_onehot(control_world, seed)

    def type_means(model, world, delta_by_type):
        means = {}
        for t, delta in delta_by_type.items():
            cfg = _single_type_config(world, t, delta)
            _, L, _ = _collect_latent_states_onehot_by_type(model, cfg, seed)
            means[t] = np.mean(L, axis=0)
        return means

    baseline_means = type_means(baseline_model, baseline_world, {1: 10.0, 2: -10.0, 3: 10.0, 4: -10.0})
    intervention_means = type_means(intervention_model, intervention_world, {1: 10.0, 2: 10.0, 3: 10.0, 4: -10.0})
    control_means = type_means(control_model, control_world, {1: 10.0, 2: -10.0, 3: 10.0, 4: -10.0})

    def type2_cluster_distances(means):
        return {
            "to_beneficial_cluster_type1": float(np.linalg.norm(means[2] - means[1])),
            "to_beneficial_cluster_type3": float(np.linalg.norm(means[2] - means[3])),
            "to_harmful_cluster_type4": float(np.linalg.norm(means[2] - means[4])),
        }

    def closer_to_beneficial(d):
        beneficial_avg = (d["to_beneficial_cluster_type1"] + d["to_beneficial_cluster_type3"]) / 2.0
        return bool(beneficial_avg < d["to_harmful_cluster_type4"])

    baseline_d = type2_cluster_distances(baseline_means)
    intervention_d = type2_cluster_distances(intervention_means)
    control_d = type2_cluster_distances(control_means)
    baseline_closer, intervention_closer, control_closer = (
        closer_to_beneficial(baseline_d), closer_to_beneficial(intervention_d), closer_to_beneficial(control_d),
    )

    return {
        "experiment_id": "CSP-P5-E006",
        "title": "Intervention Sensitivity: Does Changing the Causal Property Move the Representation?",
        "seed": seed,
        "baseline": {"type2_distances": {k: round(v, 4) for k, v in baseline_d.items()}, "type2_closer_to_beneficial": baseline_closer},
        "real_intervention_flip_hidden_state_delta": {"type2_distances": {k: round(v, 4) for k, v in intervention_d.items()}, "type2_closer_to_beneficial": intervention_closer},
        "cosmetic_control_rename_entity_id": {"type2_distances": {k: round(v, 4) for k, v in control_d.items()}, "type2_closer_to_beneficial": control_closer},
        "intervention_moved_representation": (not baseline_closer) and intervention_closer,
        "cosmetic_control_left_representation_unchanged": control_closer == baseline_closer,
        "training_histories": {
            "baseline_epochs_run": baseline_hist["epochs_run"],
            "intervention_epochs_run": intervention_hist["epochs_run"],
            "control_epochs_run": control_hist["epochs_run"],
        },
    }


PROPERTY_BOUNDARY = 0.5  # the hidden rule the network must discover: delta sign = sign(property_0 - boundary)


def _encode_property_type(obs, action, num_actions: int = 6, max_entities: int = MAX_ENTITIES, num_properties: int = 1) -> np.ndarray:
    """Property-based counterpart to _encode_onehot_type(): entity_type (the closed-vocabulary
    code that cannot, by construction, carry any information for a never-trained code) is
    DROPPED entirely and replaced with the entity's `properties` vector -- neutral continuous
    numbers with no human semantic label attached (Section 8 semantic firewall still holds:
    these are not called "beneficial"/"harmful" anywhere, just property_0, property_1, ...).

    This is the encoding used by CSP-P5-E007 to test whether true zero-shot generalization to
    a never-trained entity_type is achievable once the thing the network conditions on is a
    continuous quantity with a learnable function to the outcome, rather than an opaque index.
    """
    from learning.dataset import one_hot_encode_action

    global_vec = [
        float(obs.agent_position[0]), float(obs.agent_position[1]),
        1.0 if obs.collision else 0.0, float(obs.internal_state), float(obs.timestep),
        float(obs.previous_action if obs.previous_action is not None else -1.0),
    ]

    entity_block: List[float] = []
    for i in range(max_entities):
        if i < len(obs.entities):
            e = obs.entities[i]
            props = list(e.properties) if e.properties else [0.0] * num_properties
            props = (props + [0.0] * num_properties)[:num_properties]
            entity_block.extend([float(e.entity_id)] + [float(p) for p in props] + [float(e.relative_x), float(e.relative_y), float(e.distance)])
        else:
            entity_block.extend([0.0] * (1 + num_properties + 3))

    act_vec = one_hot_encode_action(action, num_actions=num_actions)
    return np.concatenate([np.array(global_vec, dtype=np.float64), np.array(entity_block, dtype=np.float64), act_vec])


def _dataset_to_property_arrays(dataset) -> "Tuple[np.ndarray, np.ndarray]":
    x_list = [_encode_property_type(t.obs, t.action) for t in dataset.transitions]
    y_list = [t.to_target_vector() for t in dataset.transitions]
    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)


def _train_model_property(world_config: WorldConfig, seed: int, num_steps: int = 6000):
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_steps, interaction_prob=0.3, seed=seed)
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)
    X_train, Y_train = _dataset_to_property_arrays(train_ds)
    X_val, Y_val = _dataset_to_property_arrays(val_ds)

    model = PredictiveMLP(input_dim=X_train.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    trainer = PredictiveTrainer(model=model, batch_size=32, epochs=150, early_stopping_patience=25, seed=seed)
    history = trainer.fit((X_train, Y_train), val_data=(X_val, Y_val))
    return model, history, (X_val, Y_val)


def _property_single_type_config(base_world: WorldConfig, entity_type: int, hidden_state_delta: float, property_0: float) -> WorldConfig:
    import copy

    cfg = copy.deepcopy(base_world)
    cfg.entity_type_pool = [{
        "entity_type": entity_type, "hidden_state_delta": hidden_state_delta,
        "is_interactive": True, "is_blocking": False, "properties": [property_0],
    }]
    return cfg


def _collect_latent_and_pred_property(model: PredictiveMLP, world_config: WorldConfig, seed: int, num_samples: int = 300):
    """Unlike _collect_latent_states_by_type()/_collect_latent_states_onehot_by_type() (which
    filter on action == INTERACT only), this filters on info["interaction"] -- whether an
    interaction actually occurred (agent within interaction_radius of the entity) -- not just
    whether INTERACT was the chosen action. collect_exploratory_dataset() picks INTERACT
    irrespective of proximity, so an action-only filter mixes in "attempted interact, nothing
    in range" rows whose true target is just -step_penalty, i.e. always negative regardless of
    the entity's actual function -- which would bias any per-type behavioral-sign check toward
    "predicted negative" for exactly the reason that has nothing to do with what the model
    learned about property_0. Needed here because E007 reads predicted SIGN per type, where
    E002/E004's mixed-type-pool MSE/cosine comparisons are far less sensitive to this noise.
    """
    collector = TrajectoryCollector(world_config=world_config)
    dataset = collector.collect_exploratory_dataset(num_steps=num_samples * 4, interaction_prob=0.5, seed=seed)
    interacted = [t for t in dataset.transitions if t.info.get("interaction", False)]

    x_list = [_encode_property_type(t.obs, t.action) for t in interacted]
    y_list = [t.to_target_vector() for t in interacted]
    X_interact = np.stack(x_list, axis=0)
    Y_interact = np.stack(y_list, axis=0)

    latents = model.get_latent_state(X_interact)
    preds = model.forward(X_interact)
    return X_interact, latents, Y_interact, preds


def run_experiment_e007(seed: int = 42) -> Dict[str, Any]:
    """CSP-P5-E007: True Zero-Shot Generalization via Property-Based Encoding.

    E002/E003/E004's conclusion was that no fixed-vocabulary category encoding (raw-scalar
    OR one-hot) can generalize to a genuinely never-trained entity_type, because the model's
    only handle on "type" is an opaque index/column with zero learned weight for a code it
    never saw. This experiment asks whether that limit is specific to index-based encodings,
    or fundamental to category-based reasoning in general, by giving the model something
    categorically different to condition on: a neutral CONTINUOUS property (property_0 in
    [0, 1]) with a consistent hidden rule relating it to outcome
    (delta = +10 if property_0 >= 0.5 else -10), and entity_type itself dropped from the
    input entirely.

    Four types are trained, straddling the boundary: {0.9: +10, 0.7: +10, 0.3: -10, 0.1: -10}.
    Five entity_type codes -- 5 through 9 -- are then tested, each with a property_0 value
    NEVER seen during training at all (not even the entity_type code is reused): two
    interior interpolations, one hard near-boundary case on each side, and two
    extrapolations beyond the trained property range. If the network discovered the
    continuous rule rather than memorizing four (code -> outcome) pairs, these should
    predict correctly despite being entirely unseen combinations.
    """
    train_world = _load_world("phase5_world_property_train.json")
    model, history, (X_val, Y_val) = _train_model_property(train_world, seed)
    val_metrics = compute_all_metrics(model.forward(X_val), Y_val)

    trained_types = {1: (10.0, 0.9), 2: (-10.0, 0.1), 3: (10.0, 0.7), 4: (-10.0, 0.3)}
    novel_types = {
        5: (10.0, 0.6, "interior_interpolation"),
        6: (10.0, 0.55, "near_boundary_hard"),
        7: (-10.0, 0.45, "near_boundary_hard"),
        8: (10.0, 0.95, "extrapolation_beyond_trained_max"),
        9: (-10.0, 0.05, "extrapolation_beyond_trained_min"),
    }

    latent_means, pred_sign_correct_rate = {}, {}
    for t, (delta, prop) in trained_types.items():
        cfg = _property_single_type_config(train_world, t, delta, prop)
        _, L, Y, preds = _collect_latent_and_pred_property(model, cfg, seed)
        latent_means[t] = np.mean(L, axis=0)
        pred_sign_correct_rate[t] = float(np.mean(np.sign(preds.flatten()) == np.sign(delta)))

    novel_results = {}
    for t, (delta, prop, kind) in novel_types.items():
        cfg = _property_single_type_config(train_world, t, delta, prop)
        _, L, Y, preds = _collect_latent_and_pred_property(model, cfg, seed)
        latent_means[t] = np.mean(L, axis=0)
        correct_rate = float(np.mean(np.sign(preds.flatten()) == np.sign(delta)))
        pred_sign_correct_rate[t] = correct_rate
        novel_results[f"type{t}_prop{prop}_{kind}"] = {
            "property_0": prop,
            "true_delta_sign": "positive" if delta > 0 else "negative",
            "predicted_sign_correct_rate": round(correct_rate, 4),
            "mean_prediction": round(float(np.mean(preds)), 4),
        }

    novel_overall_accuracy = float(np.mean([v["predicted_sign_correct_rate"] for v in novel_results.values()]))

    # Clustering check: do the two never-trained near-boundary-hard cases (closest to
    # memorized-threshold failure) land nearer their TRUE functional cluster's trained types
    # than the opposite one, in latent space?
    beneficial_trained = np.mean([latent_means[1], latent_means[3]], axis=0)
    harmful_trained = np.mean([latent_means[2], latent_means[4]], axis=0)
    clustering_check = {}
    for t, (delta, prop, kind) in novel_types.items():
        d_beneficial = float(np.linalg.norm(latent_means[t] - beneficial_trained))
        d_harmful = float(np.linalg.norm(latent_means[t] - harmful_trained))
        correct_cluster = (d_beneficial < d_harmful) if delta > 0 else (d_harmful < d_beneficial)
        clustering_check[f"type{t}_prop{prop}_{kind}"] = {
            "distance_to_beneficial_trained_cluster": round(d_beneficial, 4),
            "distance_to_harmful_trained_cluster": round(d_harmful, 4),
            "landed_in_correct_functional_cluster": correct_cluster,
        }
    clustering_accuracy = float(np.mean([v["landed_in_correct_functional_cluster"] for v in clustering_check.values()]))

    return {
        "experiment_id": "CSP-P5-E007",
        "title": "True Zero-Shot Generalization via Property-Based (Continuous) Encoding",
        "seed": seed,
        "property_boundary_rule": f"delta = +10 if property_0 >= {PROPERTY_BOUNDARY} else -10 (never given to the model as a rule; must be discovered from interaction outcomes)",
        "trained_types_property_0": {str(t): p for t, (_, p) in trained_types.items()},
        "trained_val_metrics": val_metrics,
        "trained_type_behavioral_accuracy": {str(t): round(r, 4) for t, r in pred_sign_correct_rate.items() if t in trained_types},
        "novel_type_results": novel_results,
        "novel_type_overall_behavioral_accuracy": round(novel_overall_accuracy, 4),
        "novel_type_clustering_check": clustering_check,
        "novel_type_clustering_accuracy": round(clustering_accuracy, 4),
        "true_zero_shot_generalization_achieved": novel_overall_accuracy >= 0.8 and clustering_accuracy >= 0.8,
        "training_history": {"epochs_run": history["epochs_run"], "final_train_loss": history["final_train_loss"]},
    }


def run_experiment_e007_stability(seeds: List[int] = [42, 101, 7, 2026, 555]) -> Dict[str, Any]:
    """Multi-seed hardening of CSP-P5-E007, the same way E003 checked E002 and
    run_experiment_e004_stability checked E004: is the property encoding's zero-shot result a
    one-seed artifact, or does it reproduce across independent training runs?
    """
    per_seed = []
    for s in seeds:
        e007 = run_experiment_e007(seed=s)
        per_seed.append({
            "seed": s,
            "novel_type_overall_behavioral_accuracy": e007["novel_type_overall_behavioral_accuracy"],
            "novel_type_clustering_accuracy": e007["novel_type_clustering_accuracy"],
            "true_zero_shot_generalization_achieved": e007["true_zero_shot_generalization_achieved"],
        })

    return {
        "experiment_id": "CSP-P5-E007-STABILITY",
        "title": "Stability Check: Does E007's Property-Based Zero-Shot Result Reproduce Across Seeds?",
        "seeds": seeds,
        "per_seed_results": per_seed,
        "mean_novel_behavioral_accuracy": round(float(np.mean([r["novel_type_overall_behavioral_accuracy"] for r in per_seed])), 4),
        "std_novel_behavioral_accuracy": round(float(np.std([r["novel_type_overall_behavioral_accuracy"] for r in per_seed])), 4),
        "mean_novel_clustering_accuracy": round(float(np.mean([r["novel_type_clustering_accuracy"] for r in per_seed])), 4),
        "achieved_rate_pct": round(100.0 * sum(r["true_zero_shot_generalization_achieved"] for r in per_seed) / len(per_seed), 1),
    }


def run_experiment_e004_stability(seeds: List[int] = [42, 101, 7, 2026, 555]) -> Dict[str, Any]:
    """Multi-seed hardening of CSP-P5-E004 (originally reported on seed=42 only): does the
    one-hot-fix's functional clustering reproduce across independent training runs, the same
    way E003 checked E002's result for the raw-scalar/held-out-code design?
    """
    per_seed = []
    for s in seeds:
        e004 = run_experiment_e004(seed=s)
        per_seed.append({
            "seed": s,
            "raw_euclidean_diff_to_same_ratio": e004["raw_scalar_control"]["euclidean_diff_to_same_ratio"],
            "onehot_euclidean_diff_to_same_ratio": e004["onehot_fix"]["euclidean_diff_to_same_ratio"],
            "raw_clustering": e004["raw_scalar_control"]["functional_clustering_euclidean"],
            "onehot_clustering": e004["onehot_fix"]["functional_clustering_euclidean"],
        })

    raw_rate = 100.0 * sum(r["raw_clustering"] for r in per_seed) / len(per_seed)
    onehot_rate = 100.0 * sum(r["onehot_clustering"] for r in per_seed) / len(per_seed)

    return {
        "experiment_id": "CSP-P5-E004-STABILITY",
        "title": "Stability Check: Does E004's Functional Clustering Reproduce Across Seeds?",
        "seeds": seeds,
        "per_seed_results": per_seed,
        "raw_scalar_clustering_reproduced_rate_pct": round(raw_rate, 1),
        "onehot_clustering_reproduced_rate_pct": round(onehot_rate, 1),
        "mean_raw_diff_to_same_ratio": round(float(np.mean([r["raw_euclidean_diff_to_same_ratio"] for r in per_seed])), 2),
        "mean_onehot_diff_to_same_ratio": round(float(np.mean([r["onehot_euclidean_diff_to_same_ratio"] for r in per_seed])), 2),
    }


def run_all_phase5_experiments(output_dir: str = "logs") -> Dict[str, Any]:
    """Execute all Phase 5 experiments sequentially and persist results."""
    import os

    print("Running CSP-P5-E001 (Predictiveness)...")
    e001 = run_experiment_e001()

    print("Running CSP-P5-E002 (Generalization / Core Clustering Test)...")
    e002 = run_experiment_e002()

    print("Running CSP-P5-E003 (Stability Across Seeds)...")
    e003 = run_experiment_e003()

    print("Running CSP-P5-E004 (One-Hot Encoding Fix)...")
    e004 = run_experiment_e004()

    print("Running CSP-P5-E004 Stability (multi-seed hardening)...")
    e004_stability = run_experiment_e004_stability()

    print("Running CSP-P5-E005 (Compression)...")
    e005 = run_experiment_e005()

    print("Running CSP-P5-E006 (Intervention Sensitivity)...")
    e006 = run_experiment_e006()

    print("Running CSP-P5-E007 (Property-Based Zero-Shot Generalization)...")
    e007 = run_experiment_e007()

    print("Running CSP-P5-E007 Stability (multi-seed hardening)...")
    e007_stability = run_experiment_e007_stability()

    all_results = {
        "phase": "Phase 5 — Emergent Categories",
        "experiments": {
            "CSP-P5-E001": e001,
            "CSP-P5-E002": e002,
            "CSP-P5-E003": e003,
            "CSP-P5-E004": e004,
            "CSP-P5-E004-STABILITY": e004_stability,
            "CSP-P5-E005": e005,
            "CSP-P5-E006": e006,
            "CSP-P5-E007": e007,
            "CSP-P5-E007-STABILITY": e007_stability,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase5_all_experiments_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Phase 5 experiments completed and saved to: {out_file}")
    return all_results


if __name__ == "__main__":
    run_all_phase5_experiments()
