"""Experimental Suite for Project Caspian (Phase 10 -- Ontology Discovery & Compositional Generalization).

Stage 10.1  CSP-P10-E001  Mechanism measurement: what do the hidden layers of passive vs
                          intervention-trained models actually encode?

PRE-REGISTERED (written before any run; see logs/phase10/ for raw per-seed output):

  Setup: the two Phase 7 confound conditions, 10 seeds each.
    clamped  = the E002 setup (clamped world, 3000 steps/region; passive confounded vs active varied-position)
    noise3   = the E006b setup at sigma=3 (clean unclamped world, noisy outcomes)
  Representation measurement uses a BALANCED probe set: interaction events from the four
  type x region cells (matched AND swapped worlds), equalized to the same count per cell, so type and
  region are statistically independent in the probe data. Hidden activations are measured on it.
  eta2 = share of total hidden-activation variance (summed over units) explained by the type main
  effect, the region main effect, their interaction (balanced two-way ANOVA decomposition).

  M1  (representation differs)  In >= 8/10 seeds in each condition, for the LAST hidden layer:
        region-eta2(passive) > region-eta2(active)  AND  type-eta2(active) > type-eta2(passive).
  M2  (representation predicts behavior)  Pooling the 20 trained models of a condition, the Spearman
        correlation between swapped-position R2 and the type/region eta2 ratio is >= +0.6.
  M3  (training, not architecture)  Untrained networks with the same architecture and the arm's own
        input normalizer show |passive - active| differences in type-eta2 smaller than half of the
        trained differences (mean over seeds).
  M4  (probes, descriptive)  Cross-validated probe balanced accuracy for type and region from the last
        hidden layer, reported against a shuffled-label control, an untrained-network control and the
        raw-input upper bound. No threshold: probe accuracy is evidence of ACCESSIBLE information, not of
        causal use. Causal use is assessed only through behavior (swapped-position R2, M2).
  Reproduction gate: the swapped-position R2 recomputed here must equal the value logged by Phase 7
        (E002-STABILITY for 'clamped', E006b for 'noise3') for seeds that exist in those logs.
"""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learning.dataset import ExperienceDataset, encode_state_action
from models.mlp import PredictiveMLP
from experiments import phase7_experiments as p7

SEEDS = [42, 101, 7, 2026, 555, 1, 2, 3, 4, 5]
CKPT_DIR = Path("logs/phase10/checkpoints")
CORNERS = [(0, 0), (4, 4), (0, 4), (4, 0)]


# ----------------------------------------------------------------------------- data / models

def _build_condition(cond: str, seed: int, steps_per_region: int = 3000):
    """Return passive_ds, active_ds and the four probe worlds as [(cfg, type_label, region_label)]."""
    if cond == "clamped":
        t1n, t3f = p7._confounded_type_configs(); t1f, t3n = p7._swapped_type_configs()
        passive = ExperienceDataset()
        passive.add_transitions(p7._collect_passive(t1n, seed=seed, num_steps=steps_per_region).transitions)
        passive.add_transitions(p7._collect_passive(t3f, seed=seed + 1, num_steps=steps_per_region).transitions)
        active = ExperienceDataset()
        active.add_transitions(p7._collect_active_varied_positions(1, 10.0, CORNERS, seed=seed, num_steps_per_corner=steps_per_region // 4).transitions)
        active.add_transitions(p7._collect_active_varied_positions(3, -10.0, CORNERS, seed=seed + 10, num_steps_per_corner=steps_per_region // 4).transitions)
        cells = [(t1n, 0, 0), (t3f, 1, 1), (t1f, 0, 1), (t3n, 1, 0)]  # (cfg, type label, region label: 0 near, 1 far)
        swapped = [t1f, t3n]
        return passive, active, cells, swapped
    if cond == "noise3":
        sigma = 3.0
        near1 = [(0, 0), (0, 1), (1, 0), (1, 1)]; far = [(3, 3), (3, 4), (4, 3), (4, 4)]

        def region_world(t, d, region, noise):
            c = p7._clean_site_config(t, d, region[0], noise)
            c.allowed_entity_positions = region; c.randomize_entity_positions = True
            return c
        # Must match Phase 7 E006b exactly (reproduction gate): entity/agent anchor corners (1,1) and (3,3), agent start (4,4) for type 3.
        # (First Phase 10 run used region[0] as the anchor for type 1, which changed the passive arm's agent start -> gate failed.)
        n1 = p7._clean_site_config(1, 10.0, (1, 1), sigma); n1.allowed_entity_positions = near1; n1.randomize_entity_positions = True
        n3 = p7._clean_site_config(3, -10.0, (3, 3), sigma); n3.allowed_entity_positions = far; n3.randomize_entity_positions = True; n3.initial_agent_pos = (4, 4)
        passive = ExperienceDataset()
        passive.add_transitions(p7._collect_passive(n1, seed=seed, num_steps=steps_per_region).transitions)
        passive.add_transitions(p7._collect_passive(n3, seed=seed + 1, num_steps=steps_per_region).transitions)
        active = ExperienceDataset()
        for et, d, off in [(1, 10.0, 0), (3, -10.0, 10)]:
            for i, cn in enumerate(CORNERS):
                active.add_transitions(p7._collect_passive(p7._clean_site_config(et, d, cn, sigma), seed=seed + off + i, num_steps=steps_per_region // 4).transitions)
        c1n, c3f, c1f, c3n = region_world(1, 10.0, near1, 0.0), region_world(3, -10.0, far, 0.0), region_world(1, 10.0, far, 0.0), region_world(3, -10.0, near1, 0.0)
        cells = [(c1n, 0, 0), (c3f, 1, 1), (c1f, 0, 1), (c3n, 1, 0)]
        return passive, active, cells, [c1f, c3n]
    raise ValueError(cond)


def _probe_set(cells, seed: int, per_cell: int = 150, steps: int = 3000):
    """Balanced interaction events over the type x region cells (equal count per cell)."""
    pools = []
    for i, (cfg, ty, rg) in enumerate(cells):
        ds = p7._collect_passive(cfg, seed=seed + 7000 + i, num_steps=steps)
        X, _ = p7._interaction_xy(ds)
        pools.append((X, ty, rg))
    n = min(min(len(p[0]) for p in pools), per_cell)
    rng = np.random.default_rng(seed)
    Xs, T, R = [], [], []
    for X, ty, rg in pools:
        idx = rng.choice(len(X), size=n, replace=False)
        Xs.append(X[idx]); T += [ty] * n; R += [rg] * n
    return np.concatenate(Xs), np.array(T), np.array(R), n


def _hidden_layers(model: PredictiveMLP, X: np.ndarray) -> List[np.ndarray]:
    curr = model.normalizer.transform(X) if model.normalizer.is_fit else X
    outs = []
    for i in range(len(model.layers) - 1):
        curr = model.hidden_act.forward(model.layers[i].forward(curr))
        outs.append(curr)
    return outs


def _swapped_r2(cond: str, model, swapped, seed: int) -> float:
    if cond == "clamped":
        return float(np.mean([p7._eval_on_world(model, c, seed=seed + 102 + i)["r2_score"] for i, c in enumerate(swapped)]))
    return float(p7._pooled_r2(model, swapped, seed=seed + 800))


# ----------------------------------------------------------------------------- measures

def eta_squared(H: np.ndarray, T: np.ndarray, R: np.ndarray) -> Dict[str, float]:
    """Balanced two-way ANOVA decomposition of hidden-activation variance, summed over units."""
    tot = ((H - H.mean(0)) ** 2).sum()
    if tot <= 1e-12:
        return {"type": 0.0, "region": 0.0, "interaction": 0.0, "residual": 1.0}
    gm = H.mean(0)
    mt = {t: H[T == t].mean(0) for t in (0, 1)}; mr = {r: H[R == r].mean(0) for r in (0, 1)}
    ss_t = sum((T == t).sum() * ((mt[t] - gm) ** 2).sum() for t in (0, 1))
    ss_r = sum((R == r).sum() * ((mr[r] - gm) ** 2).sum() for r in (0, 1))
    ss_cells = 0.0
    for t in (0, 1):
        for r in (0, 1):
            m = (T == t) & (R == r)
            ss_cells += m.sum() * ((H[m].mean(0) - gm) ** 2).sum()
    ss_i = max(ss_cells - ss_t - ss_r, 0.0)
    res = max(tot - ss_cells, 0.0)
    return {"type": float(ss_t / tot), "region": float(ss_r / tot), "interaction": float(ss_i / tot), "residual": float(res / tot)}


def probe_accuracy(H: np.ndarray, y: np.ndarray, seed: int, shuffle: bool = False) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rng = np.random.default_rng(seed)
    yy = rng.permutation(y) if shuffle else y
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, H, yy, cv=cv, scoring="balanced_accuracy")))


def _untrained_like(model_ref: PredictiveMLP, data_X: np.ndarray, seed: int) -> PredictiveMLP:
    m = PredictiveMLP(input_dim=model_ref.input_dim, hidden_dims=model_ref.hidden_dims, output_dim=1, hidden_activation="relu", seed=seed + 999)
    m.fit_normalizer(data_X)
    return m


def _measure(model, X_probe, T, R, seed):
    hs = _hidden_layers(model, X_probe)
    out = {}
    for li, H in enumerate(hs):
        out[f"layer{li + 1}"] = {"eta2": eta_squared(H, T, R)}
    last = hs[-1]
    out["probe_last_layer"] = {"type_acc": probe_accuracy(last, T, seed), "region_acc": probe_accuracy(last, R, seed),
                               "type_acc_shuffled": probe_accuracy(last, T, seed, shuffle=True), "region_acc_shuffled": probe_accuracy(last, R, seed, shuffle=True)}
    return out


def run_experiment_e001(conditions: List[str] = ("clamped", "noise3"), seeds: List[int] = SEEDS, save_checkpoints: bool = True) -> Dict[str, Any]:
    """CSP-P10-E001: mechanism measurement. See module docstring for the pre-registered hypotheses."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    per_run = []
    for cond in conditions:
        for s in seeds:
            passive_ds, active_ds, cells, swapped = _build_condition(cond, s)
            X_probe, T, R, n_cell = _probe_set(cells, s)
            Xp, _ = passive_ds.to_numpy_arrays(); Xa, _ = active_ds.to_numpy_arrays()
            row = {"condition": cond, "seed": s, "probe_events_per_cell": int(n_cell), "arms": {}}
            for arm, ds, Xtrain in [("passive", passive_ds, Xp), ("active", active_ds, Xa)]:
                model, info = p7._train_on_dataset(ds, seed=s)
                if save_checkpoints:
                    (CKPT_DIR / f"{cond}_{arm}_seed{s}.json").write_text(json.dumps(model.to_dict()))
                entry = {"num_train_samples": info["num_train_samples"], "swapped_r2": round(_swapped_r2(cond, model, swapped, s), 4)}
                entry["trained"] = _measure(model, X_probe, T, R, s)
                entry["untrained"] = _measure(_untrained_like(model, Xtrain, s), X_probe, T, R, s)
                entry["raw_input_probe"] = {"type_acc": probe_accuracy(model.normalizer.transform(X_probe), T, s), "region_acc": probe_accuracy(model.normalizer.transform(X_probe), R, s)}
                row["arms"][arm] = entry
            per_run.append(row)
            print("E001", cond, "seed", s, {a: row["arms"][a]["swapped_r2"] for a in row["arms"]}, flush=True)
    return {"experiment_id": "CSP-P10-E001", "title": "Mechanism Measurement: Passive vs Intervention-Trained Representations",
            "seeds": seeds, "conditions": list(conditions), "per_run": per_run, "summary": summarize(per_run, conditions)}


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if r == r else 0.0


def summarize(per_run, conditions) -> Dict[str, Any]:
    out = {}
    for cond in conditions:
        rows = [r for r in per_run if r["condition"] == cond]
        g = lambda arm, kind, key, layer="layer2": [r["arms"][arm][kind][layer]["eta2"][key] for r in rows]
        last = "layer%d" % (len(rows[0]["arms"]["passive"]["trained"]) - 1)  # layers + probe entry
        g = lambda arm, kind, key: [r["arms"][arm][kind][last]["eta2"][key] for r in rows]
        t_p, t_a = np.array(g("passive", "trained", "type")), np.array(g("active", "trained", "type"))
        r_p, r_a = np.array(g("passive", "trained", "region")), np.array(g("active", "trained", "region"))
        m1_votes = int(np.sum((r_p > r_a) & (t_a > t_p)))
        swp = [r["arms"][a]["swapped_r2"] for r in rows for a in ("passive", "active")]
        ratio = [r["arms"][a]["trained"][last]["eta2"]["type"] / max(r["arms"][a]["trained"][last]["eta2"]["region"], 1e-6) for r in rows for a in ("passive", "active")]
        m2 = _spearman(swp, ratio)
        ut_p, ut_a = np.array(g("passive", "untrained", "type")), np.array(g("active", "untrained", "type"))
        d_trained, d_untrained = float(np.mean(np.abs(t_a - t_p))), float(np.mean(np.abs(ut_a - ut_p)))
        pr = lambda arm, kind, key: float(np.mean([r["arms"][arm][kind]["probe_last_layer"][key] if kind != "raw" else r["arms"][arm]["raw_input_probe"][key] for r in rows]))
        out[cond] = {
            "n_seeds": len(rows), "last_layer": last,
            "eta2_means": {arm: {k: round(float(np.mean(g(arm, "trained", k))), 4) for k in ("type", "region", "interaction", "residual")} for arm in ("passive", "active")},
            "eta2_untrained_means": {arm: {k: round(float(np.mean(g(arm, "untrained", k))), 4) for k in ("type", "region")} for arm in ("passive", "active")},
            "swapped_r2_means": {arm: round(float(np.mean([r["arms"][arm]["swapped_r2"] for r in rows])), 4) for arm in ("passive", "active")},
            "M1_votes": m1_votes, "M1_met": bool(m1_votes >= 8),
            "M2_spearman_swappedR2_vs_type_region_ratio": round(m2, 4), "M2_met": bool(m2 >= 0.6),
            "M3_mean_abs_type_eta2_diff_trained": round(d_trained, 4), "M3_mean_abs_type_eta2_diff_untrained": round(d_untrained, 4), "M3_met": bool(d_untrained < 0.5 * d_trained),
            "M4_probe_means": {arm: {"type_acc": round(pr(arm, "trained", "type_acc"), 4), "region_acc": round(pr(arm, "trained", "region_acc"), 4),
                                     "type_acc_shuffled": round(pr(arm, "trained", "type_acc_shuffled"), 4), "region_acc_shuffled": round(pr(arm, "trained", "region_acc_shuffled"), 4),
                                     "type_acc_untrained": round(pr(arm, "untrained", "type_acc"), 4), "region_acc_untrained": round(pr(arm, "untrained", "region_acc"), 4),
                                     "type_acc_raw_input": round(pr(arm, "raw", "type_acc"), 4), "region_acc_raw_input": round(pr(arm, "raw", "region_acc"), 4)} for arm in ("passive", "active")},
        }
    return out



# =============================================================================================
# Stage 10.2-10.4  CSP-P10-E002: compositional world, passive vs intervention-driven data, held-out combinations
# =============================================================================================
# PRE-REGISTERED (written before any E002 run).
#
# World: one entity per episode on a 5x5 grid; each entity carries three observable neutral numeric property
# channels (a, b, c) in {0.2, 0.8}^3 (8 combinations) and an opaque type code that the encoder DROPS (index-like
# codes cannot generalize; Phase 5). The outcome of interacting is additive in the three hidden binary
# properties, delta = 6*A - 4*B + 2*C (variant 'interaction' adds +5*A*B, which breaks additivity). Agents are
# never told which channel matters or that they combine additively. Internal state is unclamped.
# Training combinations T = {000, 111, 100, 010, 001}; HELD-OUT combinations H = {110, 101, 011} never appear in
# any training transition of any arm (automated leakage test).
# Arms (equal env-step budgets, same MLP (32,16), same training recipe):
#   passive_natural   random walk in a world whose natural spawn mix is skewed: 000 and 111 each 43%,
#                     100/010/001 each 7%  (A,B,C co-occur: the properties are confounded in nature)
#   passive_uniform   same random walk, uniform spawn over T       (control: balance without agency)
#   coverage_count    blocks of 100 steps; the agent picks the training combination it has sampled least
#   one_factor        blocks of 100 steps cycling 000,100,010,001: vary ONE property from baseline, others fixed
# Baselines on each arm's data: global mean; nearest-training-combination memorization; additive linear
# regression on the property channels (structure oracle).
# Budgets 1000/2000/4000/8000 env steps; 10 seeds. Primary metric: R2 on interaction events of held-out combos.
#   C1 (primary)  at >= 2 of 4 budgets, the best intervention arm (coverage_count or one_factor, chosen BEFORE
#                 looking by the rule 'one_factor') beats passive_natural by >= 0.10 mean held-out R2
#                 with >= 8/10 seeds favoring it (paired).
#   C2 (distribution vs agency)  passive_uniform vs coverage_count differ by < 0.05 mean held-out R2 at every
#                 budget (i.e. any gain over passive_natural is explained by data balance, not by choosing).
#   C3 (negative control)  in the 'interaction' world no arm reaches held-out R2 >= 0.5 at the largest budget
#                 (the held-out test discriminates composition from memorization).
#   C4 (leakage)  zero held-out property vectors in any training set; held-out codes never trained.
# Descriptive (no threshold): in-distribution R2, baselines, per-combo errors.

COMBO_VALUES = (0.2, 0.8)
TRAIN_COMBOS = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
HELDOUT_COMBOS = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
_CODE_PERM = [4, 7, 1, 6, 3, 8, 2, 5]  # fixed arbitrary mapping combo-index -> neutral code (not encoded, kept for leakage tests)


def _combo_index(c) -> int:
    return c[0] * 4 + c[1] * 2 + c[2]


def combo_delta(c, variant: str = "additive") -> float:
    a, b, cc = c
    d = 6.0 * a - 4.0 * b + 2.0 * cc
    if variant == "interaction":
        d += 5.0 * a * b
    return d


def _template(c, variant):
    return {"entity_type": 100 + _CODE_PERM[_combo_index(c)], "hidden_state_delta": combo_delta(c, variant), "is_interactive": True,
            "is_blocking": False, "properties": [COMBO_VALUES[int(x)] for x in c]}


def make_comp_world(mix, variant: str = "additive"):
    """mix: list of (combo, integer weight). The pool is replicated so sampling probability is proportional."""
    from environment.entities import Entity
    cfg = copy.deepcopy(p7._load_world("phase7_world_base.json"))
    first = mix[0][0]
    cfg.initial_entities = [Entity(entity_id=1, entity_type=100 + _CODE_PERM[_combo_index(first)], position=(2, 2),
                                   hidden_state_delta=combo_delta(first, variant), properties=tuple(COMBO_VALUES[int(x)] for x in first))]
    cfg.entity_type_pool = [_template(c, variant) for c, w in mix for _ in range(int(w))]
    cfg.randomize_entity_positions = True
    cfg.allowed_entity_positions = None
    cfg.initial_agent_pos = (0, 0)
    cfg.initial_internal_state = 0.0
    cfg.min_internal_state = -1000.0
    cfg.max_internal_state = 1000.0
    return cfg


def _enc(ds):
    from experiments.phase5_experiments import _encode_property_type
    X = np.stack([_encode_property_type(t.obs, t.action, num_properties=3) for t in ds.transitions], axis=0)
    Y = np.stack([t.to_target_vector() for t in ds.transitions], axis=0)
    return X, Y


def _interaction_rows(ds):
    from experiments.phase5_experiments import _encode_property_type
    rows = [t for t in ds.transitions if t.info.get("interaction", False)]
    if not rows:
        return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 3))
    X = np.stack([_encode_property_type(t.obs, t.action, num_properties=3) for t in rows], axis=0)
    Y = np.stack([t.to_target_vector() for t in rows], axis=0)
    P = np.stack([np.array(t.obs.entities[0].properties, dtype=float) if t.obs.entities else np.zeros(3) for t in rows], axis=0)
    return X, Y, P


NATURAL_MIX = [((0, 0, 0), 6), ((1, 1, 1), 6), ((1, 0, 0), 1), ((0, 1, 0), 1), ((0, 0, 1), 1)]
UNIFORM_MIX = [(c, 1) for c in TRAIN_COMBOS]
ONE_FACTOR = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


def collect_arm(arm: str, seed: int, budget: int, variant: str = "additive", block: int = 100):
    ds = ExperienceDataset(); used = 0; k = 0
    if arm in ("passive_natural", "passive_uniform"):
        cfg = make_comp_world(NATURAL_MIX if arm == "passive_natural" else UNIFORM_MIX, variant)
        ds.add_transitions(p7._collect_passive(cfg, seed=seed, num_steps=budget).transitions)
        return ds
    counts = {c: 0 for c in TRAIN_COMBOS}
    while used < budget:
        n = min(block, budget - used)
        if arm == "coverage_count":
            combo = min(TRAIN_COMBOS, key=lambda c: (counts[c], TRAIN_COMBOS.index(c)))
        elif arm == "one_factor":
            combo = ONE_FACTOR[k % len(ONE_FACTOR)]
        else:
            raise ValueError(arm)
        ds.add_transitions(p7._collect_passive(make_comp_world([(combo, 1)], variant), seed=seed * 1000 + k, num_steps=n).transitions)
        counts[combo] = counts.get(combo, 0) + n; used += n; k += 1
    return ds


def leakage_report(ds) -> Dict[str, Any]:
    props = set()
    for t in ds.transitions:
        for e in t.obs.entities:
            props.add(tuple(round(float(x), 3) for x in e.properties))
    held = {tuple(COMBO_VALUES[int(x)] for x in c) for c in HELDOUT_COMBOS}
    codes_seen = {int(e.entity_type) for t in ds.transitions for e in t.obs.entities if hasattr(e, "entity_type")}
    held_codes = {100 + _CODE_PERM[_combo_index(c)] for c in HELDOUT_COMBOS}
    return {"heldout_property_vectors_in_training": sorted(props & held), "heldout_codes_in_training": sorted(codes_seen & held_codes)}


def _fit_mlp(X, Y, seed, epochs=100):
    from models.mlp import PredictiveMLP
    from learning.trainer import PredictiveTrainer
    m = PredictiveMLP(input_dim=X.shape[1], hidden_dims=(32, 16), output_dim=1, hidden_activation="relu", seed=seed)
    PredictiveTrainer(model=m, batch_size=32, epochs=epochs, early_stopping_patience=None, seed=seed).fit((X, Y))
    return m


def _r2_vec(pred, y) -> float:
    from evaluation.metrics import compute_all_metrics
    return float(compute_all_metrics(np.asarray(pred).reshape(-1, 1), np.asarray(y).reshape(-1, 1))["r2_score"])


def _baselines(train_P, train_Y, test_P, test_Y):
    """global mean, nearest-training-combination memorization, additive linear regression on property channels."""
    out = {}
    out["global_mean"] = _r2_vec(np.full(len(test_Y), float(np.mean(train_Y))), test_Y)
    keys = {}
    for p, y in zip(np.round(train_P, 3), train_Y.reshape(-1)):
        keys.setdefault(tuple(p), []).append(float(y))
    cm = {k: float(np.mean(v)) for k, v in keys.items()}
    def nearest(p):
        d = {k: np.abs(np.array(k) - p).sum() for k in cm}
        m = min(d.values()); return float(np.mean([cm[k] for k in cm if d[k] == m]))
    out["nearest_combo_memorization"] = _r2_vec(np.array([nearest(p) for p in np.round(test_P, 3)]), test_Y)
    A = np.hstack([train_P, np.ones((len(train_P), 1))]); w, *_ = np.linalg.lstsq(A, train_Y.reshape(-1), rcond=None)
    out["additive_linear_on_channels"] = _r2_vec(np.hstack([test_P, np.ones((len(test_P), 1))]) @ w, test_Y)
    return out


def _eval_sets(seed: int, variant: str):
    held = [p7._collect_passive(make_comp_world([(c, 1)], variant), seed=seed + 9000 + i, num_steps=2500) for i, c in enumerate(HELDOUT_COMBOS)]
    ind = [p7._collect_passive(make_comp_world([(c, 1)], variant), seed=seed + 8000 + i, num_steps=2500) for i, c in enumerate(TRAIN_COMBOS)]
    cat = lambda dss: tuple(np.concatenate(z) for z in zip(*[_interaction_rows(d) for d in dss]))
    return cat(held), cat(ind)


def run_experiment_e002(seeds: List[int] = SEEDS, budgets: List[int] = (1000, 2000, 4000, 8000), variants=("additive", "interaction"),
                        arms=("passive_natural", "passive_uniform", "coverage_count", "one_factor"), save_raw_seeds=(42,), verbose=True) -> Dict[str, Any]:
    raw_dir = Path("logs/phase10/raw"); raw_dir.mkdir(parents=True, exist_ok=True)
    rows, leaks = [], []
    for variant in variants:
        for s in seeds:
            (Xh, Yh, Ph), (Xi, Yi, Pi) = _eval_sets(s, variant)
            for B in budgets:
                for arm in arms:
                    ds = collect_arm(arm, s, B, variant)
                    leak = leakage_report(ds); leaks.append({"variant": variant, "seed": s, "budget": B, "arm": arm, **leak})
                    X, Y = _enc(ds)
                    model = _fit_mlp(X, Y, seed=s)
                    Xt, Yt, Pt = _interaction_rows(ds)
                    row = {"variant": variant, "seed": s, "budget": B, "arm": arm, "n_transitions": len(ds.transitions), "n_interaction_events": int(len(Xt)),
                           "heldout_r2": round(_r2_vec(model.forward(Xh), Yh), 4), "indist_r2": round(_r2_vec(model.forward(Xi), Yi), 4),
                           "heldout_per_combo_mae": {}, "baselines_heldout_r2": {k: round(v, 4) for k, v in _baselines(Pt, Yt, Ph, Yh).items()}}
                    pred_h = model.forward(Xh).reshape(-1)
                    for c in HELDOUT_COMBOS:
                        m = np.all(np.isclose(Ph, [COMBO_VALUES[int(x)] for x in c], atol=1e-3), axis=1)
                        row["heldout_per_combo_mae"][str(c)] = round(float(np.mean(np.abs(pred_h[m] - Yh.reshape(-1)[m]))) if m.any() else float("nan"), 4)
                    rows.append(row)
                    if B == max(budgets) and s in save_raw_seeds:
                        np.savez_compressed(raw_dir / f"e002_{variant}_{arm}_seed{s}_B{B}.npz", X=X.astype(np.float32), Y=Y.astype(np.float32))
                if verbose:
                    print("E002", variant, "seed", s, "budget", B, {a: [r["heldout_r2"] for r in rows if r["seed"] == s and r["budget"] == B and r["arm"] == a and r["variant"] == variant][0] for a in arms}, flush=True)
    return {"experiment_id": "CSP-P10-E002", "title": "Compositional World: Passive vs Intervention-Driven Data, Held-Out Property Combinations",
            "seeds": list(seeds), "budgets": list(budgets), "arms": list(arms), "variants": list(variants), "train_combos": TRAIN_COMBOS,
            "heldout_combos": HELDOUT_COMBOS, "per_run": rows, "leakage_checks": leaks, "summary": summarize_e002(rows, leaks, list(budgets), list(seeds))}


def summarize_e002(rows, leaks, budgets, seeds) -> Dict[str, Any]:
    def get(variant, arm, B, key="heldout_r2"):
        return np.array([next(r[key] for r in rows if r["variant"] == variant and r["arm"] == arm and r["budget"] == B and r["seed"] == s) for s in seeds])
    out = {"per_budget": {}}
    c1_budgets = 0
    for B in budgets:
        d = {}
        for arm in ("passive_natural", "passive_uniform", "coverage_count", "one_factor"):
            d[arm] = {"heldout_r2_mean": round(float(get("additive", arm, B).mean()), 4), "heldout_r2_std": round(float(get("additive", arm, B).std()), 4),
                      "indist_r2_mean": round(float(get("additive", arm, B, "indist_r2").mean()), 4)}
        diff = get("additive", "one_factor", B) - get("additive", "passive_natural", B)
        d["one_factor_minus_passive_natural"] = {"mean": round(float(diff.mean()), 4), "seeds_positive": int((diff > 0).sum())}
        ok = bool(diff.mean() >= 0.10 and (diff > 0).sum() >= 0.8 * len(seeds)); c1_budgets += int(ok); d["C1_criterion_met"] = ok
        dd = get("additive", "passive_uniform", B) - get("additive", "coverage_count", B)
        d["C2_uniform_minus_count_mean"] = round(float(dd.mean()), 4)
        d["baselines_heldout_r2_mean_by_arm"] = {arm: {k: round(float(np.mean([r["baselines_heldout_r2"][k] for r in rows if r["variant"] == "additive" and r["arm"] == arm and r["budget"] == B])), 4) for k in ("global_mean", "nearest_combo_memorization", "additive_linear_on_channels")} for arm in ("passive_natural", "coverage_count", "one_factor")}
        out["per_budget"][str(B)] = d
    out["C1_met"] = bool(c1_budgets >= 2); out["C1_budgets_meeting_criterion"] = c1_budgets
    out["C2_met"] = bool(all(abs(out["per_budget"][str(B)]["C2_uniform_minus_count_mean"]) < 0.05 for B in budgets))
    Bmax = max(budgets)
    inter = {arm: round(float(get("interaction", arm, Bmax).mean()), 4) for arm in ("passive_natural", "passive_uniform", "coverage_count", "one_factor")}
    out["interaction_world_heldout_r2_at_max_budget"] = inter
    out["C3_met"] = bool(all(v < 0.5 for v in inter.values()))
    out["C4_met"] = bool(all(not l["heldout_property_vectors_in_training"] and not l["heldout_codes_in_training"] for l in leaks))
    return out


def run_all_phase10_experiments(output_dir: str = "logs/phase10") -> Dict[str, Any]:
    import os
    from experiments.safe_io import safe_write_json
    res = {"phase": "Phase 10 -- Ontology Discovery & Compositional Generalization", "experiments": {}}
    if os.environ.get("CASPIAN_P10_ONLY") in (None, "", "e001"):
        res["experiments"]["CSP-P10-E001"] = run_experiment_e001()
        safe_write_json(os.path.join(output_dir, "phase10_e001_results.json"), res)
    if os.environ.get("CASPIAN_P10_ONLY") == "e002":
        res["experiments"]["CSP-P10-E002"] = run_experiment_e002()
        safe_write_json(os.path.join(output_dir, "phase10_e002_results.json"), res)
    return res


if __name__ == "__main__":
    run_all_phase10_experiments()
