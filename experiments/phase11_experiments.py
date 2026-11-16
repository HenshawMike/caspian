"""Phase 11: architectural inductive bias and compositional generalisation.

Reuses: environment.world (via phase7._collect_passive), models.mlp.PredictiveMLP, models.optimizers (Adam, MSELoss),
phase10.make_comp_world-style world construction (generalised here), phase5._encode_property_type (A1 only).
Differences from Phase 10 (documented in docs/phase11/): own training loop (PredictiveTrainer restores 'best' weights by
rebuilding PredictiveMLP layers, which breaks composite models and continued fine-tuning); interaction-event rows only for
the architecture-matched comparison; held-out RMSE as primary metric.
"""
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import experiments.phase7_experiments as p7
from learning.dataset import ExperienceDataset
from models.compositional import build_model
from models.mlp import PredictiveMLP
from models.optimizers import Adam, MSELoss

SEEDS = [42, 101, 7, 2026, 555, 1, 2, 3, 4, 5]
EXPLORATORY_SEEDS = [9001, 9002, 9003]
ARCHS = ["mlp_small", "mlp_large", "factorised", "additive", "additive_mlp"]
EPOCHS = 200  # fixed a priori for all architectures, NOT tuned (the exploratory run on seeds 9001-9003 only checked that the pipeline works; see docs/phase11/EXPERIMENT_REGISTRY.md)
LR = 0.01
STEP_OFFSET = -1.0  # world step_penalty (1.0) applied to every transition, including interactions


# --------------------------------------------------------------------------------------------------
# World specifications
# --------------------------------------------------------------------------------------------------
@dataclass
class WorldSpec:
    name: str
    # channel layout: list of (kind, values). kind = ("prop", prop_index) or ("nuisance", None)
    channels: List[Tuple[Optional[int], Tuple[float, ...]]]
    n_levels: Tuple[int, ...]                      # number of levels of each relevant property
    train_combos: List[Tuple[int, ...]]
    heldout_combos: List[Tuple[int, ...]]
    delta: Callable[[Tuple[int, ...], str], float]
    value_range: float                              # additive-world range of outcomes (nRMSE denominator)
    steps_per_combo: Tuple[int, int] = (667, 2667)  # (low, high) training env steps per training combination

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def all_combos(self):
        import itertools
        return list(itertools.product(*[range(n) for n in self.n_levels]))

    def nuisance_values(self):
        idx = [i for i, (p, _) in enumerate(self.channels) if p is None]
        import itertools
        return list(itertools.product(*[self.channels[i][1] for i in idx])) or [()]

    def channel_vector(self, combo, nuis=()):
        out, ni = [], 0
        for p, vals in self.channels:
            if p is None:
                out.append(float(nuis[ni])); ni += 1
            else:
                out.append(float(vals[combo[p]]))
        return tuple(out)

    def decode(self, P: np.ndarray) -> np.ndarray:
        """channel values -> level tuples (N, n_props) (analysis only; models never see this)."""
        cols = []
        for p in sorted({p for p, _ in self.channels if p is not None}):
            ci = [i for i, (q, _) in enumerate(self.channels) if q == p][0]
            vals = np.array(self.channels[ci][1])
            cols.append(np.argmin(np.abs(P[:, ci:ci + 1] - vals[None, :]), axis=1))
        return np.stack(cols, axis=1)

    def truth(self, P: np.ndarray, variant: str) -> np.ndarray:
        """Observed outcome of an interaction = rule(combo) + STEP_OFFSET (the environment's constant per-step cost,
        verified against the simulator in tests/test_phase11.py). Found during exploratory smoke tests: an earlier draft of this
        function omitted the offset and made every model look 1.0 wrong."""
        L = self.decode(P)
        return np.array([self.delta(tuple(int(x) for x in l), variant) + STEP_OFFSET for l in L], dtype=float)


def _w1_delta(c, variant):
    a, b, cc = c
    d = 6.0 * a - 4.0 * b + 2.0 * cc
    return d + (5.0 * a * b if variant == "interaction" else 0.0)


def _w2_delta(c, variant):
    lu, lv, lw = c
    d = 3.0 * lu - 4.0 * lv + float(lw) ** 2
    return d + (3.0 * (1 if (lu == 2 and lw >= 1) else 0) if variant == "interaction" else 0.0)


W1 = WorldSpec(
    name="W1",
    channels=[(0, (0.2, 0.8)), (1, (0.2, 0.8)), (2, (0.2, 0.8))],
    n_levels=(2, 2, 2),
    train_combos=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)],
    heldout_combos=[(1, 1, 0), (1, 1, 1)],
    delta=_w1_delta, value_range=12.0, steps_per_combo=(667, 2667))

# W2: different channel order (w, nuisance z, u, v), 3-level properties, non-linear (quadratic) effect of w, a distractor
# channel that never matters, and a threshold-type interaction. Held-out: u at its top level together with any w>0.
W2 = WorldSpec(
    name="W2",
    channels=[(2, (0.15, 0.55, 0.95)), (None, (0.30, 0.60)), (0, (0.10, 0.50, 0.90)), (1, (0.25, 0.75))],
    n_levels=(3, 2, 3),
    train_combos=[c for c in __import__("itertools").product(range(3), range(2), range(3)) if not (c[0] == 2 and c[2] >= 1)],
    heldout_combos=[c for c in __import__("itertools").product(range(3), range(2), range(3)) if (c[0] == 2 and c[2] >= 1)],
    delta=_w2_delta, value_range=14.0, steps_per_combo=(643, 2571))


# --------------------------------------------------------------------------------------------------
# Environment construction and data collection
# --------------------------------------------------------------------------------------------------
def make_world(spec: WorldSpec, combos: Sequence[Tuple[int, ...]], variant: str = "additive"):
    from environment.entities import Entity
    cfg = copy.deepcopy(p7._load_world("phase7_world_base.json"))
    pool = []
    for k, c in enumerate(combos):
        for nuis in spec.nuisance_values():
            pool.append({"entity_type": 100 + k, "hidden_state_delta": spec.delta(c, variant), "is_interactive": True,
                         "is_blocking": False, "properties": list(spec.channel_vector(c, nuis))})
    first = pool[0]
    cfg.initial_entities = [Entity(entity_id=1, entity_type=first["entity_type"], position=(2, 2),
                                   hidden_state_delta=first["hidden_state_delta"], properties=tuple(first["properties"]))]
    cfg.entity_type_pool = pool
    cfg.randomize_entity_positions = True
    cfg.allowed_entity_positions = None
    cfg.initial_agent_pos = (0, 0)
    cfg.initial_internal_state = 0.0
    cfg.min_internal_state = -1000.0
    cfg.max_internal_state = 1000.0
    return cfg


def interaction_rows(ds: ExperienceDataset, n_props: int):
    P, Y = [], []
    for t in ds.transitions:
        if t.info.get("interaction", False) and t.obs.entities:
            P.append(np.array(t.obs.entities[0].properties[:n_props], dtype=float))
            Y.append(float(t.to_target_vector().reshape(-1)[0]))
    if not P:
        return np.zeros((0, n_props)), np.zeros(0)
    return np.stack(P), np.array(Y)


def collect_train(spec: WorldSpec, seed: int, steps: int):
    """Uniform-over-training-combinations passive collection (balanced; Phase 10 showed balance, not agency, drives gains)."""
    cfg = make_world(spec, spec.train_combos, "additive")
    ds = p7._collect_passive(cfg, seed=seed, num_steps=steps)
    P, Y = interaction_rows(ds, spec.n_channels)
    return ds, P, Y


def collect_eval(spec: WorldSpec, seed: int, which: str, steps: int = 3000):
    combos = spec.heldout_combos if which == "heldout" else spec.train_combos
    base = 9000 if which == "heldout" else 8000
    Ps = []
    for i, c in enumerate(combos):
        ds = p7._collect_passive(make_world(spec, [c], "additive"), seed=seed + base + i, num_steps=steps)
        P, _ = interaction_rows(ds, spec.n_channels)
        Ps.append(P)
    return np.concatenate(Ps)


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------
def train_model(model, X, Y, seed: int, epochs: int = EPOCHS, lr: float = LR, batch_size: int = 32,
                optimizer: Optional[Adam] = None, refit_normalizer: bool = True, rng=None):
    """Same loop for every architecture: Adam(lr, wd 1e-4), MSE, batch 32, fixed epochs, NO best-weight restoring."""
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    if refit_normalizer and hasattr(model, "fit_normalizer"):
        model.fit_normalizer(X)
    opt = optimizer or Adam(params=model.get_params(), lr=lr, weight_decay=1e-4)
    loss_fn = MSELoss()
    rng = rng if rng is not None else np.random.default_rng(seed)
    n = len(X); losses = []
    for _ in range(epochs):
        idx = np.arange(n); rng.shuffle(idx)
        tot = 0.0
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            model.zero_grad()
            y = model.forward(X[b])
            tot += float(loss_fn.forward(y, Y[b])) * len(b)
            model.backward(loss_fn.gradient(y, Y[b]))
            opt.step()
        losses.append(tot / n)
    return opt, losses


def predict(model, P):
    return np.asarray(model.forward(P)).reshape(-1)


# --------------------------------------------------------------------------------------------------
# Metrics and baselines
# --------------------------------------------------------------------------------------------------
def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def r2(pred, y):
    y = np.asarray(y); ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((np.asarray(pred) - y) ** 2) / ss) if ss > 1e-12 else float("nan")


def baselines(spec: WorldSpec, Ptr, Ytr, Pte):
    """Return predictions for: memorisation (nearest training combination), additive-dummy regression (fit on train),
    structure oracle (analytical additive formula), per the directive."""
    out = {}
    keys = {}
    for p, y in zip(np.round(Ptr, 3), Ytr):
        keys.setdefault(tuple(p), []).append(y)
    cm = {k: float(np.mean(v)) for k, v in keys.items()}
    def nearest(p):
        d = {k: np.abs(np.array(k) - p).sum() for k in cm}; m = min(d.values())
        return float(np.mean([cm[k] for k in cm if d[k] == m]))
    out["memorisation"] = np.array([nearest(p) for p in np.round(Pte, 3)])
    # additive dummy regression over each channel's distinct values (identifiable from training data when every level is seen)
    def design(P):
        cols = [np.ones(len(P))]
        for ci in range(P.shape[1]):
            vals = sorted(set(np.round(Ptr[:, ci], 3)))[1:]
            for v in vals:
                cols.append((np.abs(P[:, ci] - v) < 1e-3).astype(float))
        return np.stack(cols, axis=1)
    w, *_ = np.linalg.lstsq(design(Ptr), Ytr, rcond=None)
    out["additive_fit"] = design(Pte) @ w
    out["oracle_additive_formula"] = spec.truth(Pte, "additive")
    return out


def per_combo_stats(spec, P, pred, y):
    L = spec.decode(P); res = {}
    for c in sorted({tuple(int(x) for x in l) for l in L}):
        m = np.all(L == np.array(c), axis=1)
        res["".join(map(str, c))] = {"mae": round(float(np.mean(np.abs(pred[m] - y[m]))), 4), "bias": round(float(np.mean(pred[m] - y[m])), 4), "n": int(m.sum())}
    return res


def pairwise_accuracy(spec, P, pred, variant, need_heldout: bool):
    """Behavioural test: among all pairs of distinct combinations with different true outcomes, how often does the model's mean
    prediction order them correctly? (a 'which would you interact with' choice). Pairs restricted to those touching a held-out combo."""
    L = spec.decode(P); held = {tuple(c) for c in spec.heldout_combos}
    combos = sorted({tuple(int(x) for x in l) for l in L})
    mp = {c: float(np.mean(pred[np.all(L == np.array(c), axis=1)])) for c in combos}
    ok = tot = 0
    for i, a in enumerate(combos):
        for b in combos[i + 1:]:
            if need_heldout and a not in held and b not in held:
                continue
            ta, tb = spec.delta(a, variant), spec.delta(b, variant)  # offset cancels
            if abs(ta - tb) < 1e-9:
                continue
            tot += 1; ok += int((mp[a] - mp[b]) * (ta - tb) > 0)
    return ok / tot if tot else float("nan")


def counterfactual_flip_error(spec, model, variant):
    """For every pair of neighbouring levels of one property (others fixed), compare predicted effect to the true effect of the
    world variant. Split into SEEN pairs (both endpoints training combinations) and UNSEEN (an endpoint is held out).
    Also returns the std across contexts of predicted effect per (property, step): low = a reusable effect."""
    held = {tuple(c) for c in spec.heldout_combos}
    nuis = spec.nuisance_values()[0]
    seen, unseen, eff = [], [], {}
    for c in spec.all_combos():
        for p in range(len(spec.n_levels)):
            if c[p] + 1 >= spec.n_levels[p]:
                continue
            c2 = list(c); c2[p] += 1; c2 = tuple(c2)
            d_true = spec.delta(c2, variant) - spec.delta(c, variant)  # offset cancels
            d_pred = float(predict(model, np.array([spec.channel_vector(c2, nuis)]))[0] - predict(model, np.array([spec.channel_vector(c, nuis)]))[0])
            err = abs(d_pred - d_true)
            (unseen if (c in held or c2 in held) else seen).append(err)
            eff.setdefault((p, c[p]), []).append(d_pred)
    cons = float(np.mean([np.std(v) for v in eff.values() if len(v) > 1]))
    return {"seen_mae": float(np.mean(seen)) if seen else float("nan"), "unseen_mae": float(np.mean(unseen)) if unseen else float("nan"),
            "effect_std_across_contexts": cons}


def marginal_eta2(H: np.ndarray, labels: np.ndarray) -> float:
    tot = float(np.sum((H - H.mean(0)) ** 2))
    if tot < 1e-12:
        return 0.0
    ss = sum(float(np.sum(labels == g)) * float(np.sum((H[labels == g].mean(0) - H.mean(0)) ** 2)) for g in np.unique(labels))
    return ss / tot


def ridge_probe_r2(Htr, ytr, Hte, yte, lam=1e-3, shuffle_seed=None):
    y = ytr.copy()
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
    A = np.hstack([Htr, np.ones((len(Htr), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return r2(np.hstack([Hte, np.ones((len(Hte), 1))]) @ w, yte)


def latent(model, P):
    if hasattr(model, "get_latent_state"):
        return np.asarray(model.get_latent_state(P))
    raise ValueError("no latent")


# --------------------------------------------------------------------------------------------------
# Confirmatory/exploratory run
# --------------------------------------------------------------------------------------------------
def run_cell(spec: WorldSpec, arch: str, seed: int, steps: int, cache: Dict[str, Any], raw_dir: Optional[Path] = None, epochs: int = EPOCHS, lat_store: Optional[Dict] = None):
    key = (spec.name, seed, steps)
    if key not in cache:
        ds, Ptr, Ytr = collect_train(spec, seed, steps)
        Ph = collect_eval(spec, seed, "heldout"); Pi = collect_eval(spec, seed, "indist", 2000)
        cache[key] = (Ptr, Ytr, Ph, Pi)
    Ptr, Ytr, Ph, Pi = cache[key]
    L = spec.decode(Ptr); held = {tuple(c) for c in spec.heldout_combos}
    leak = int(sum(tuple(int(x) for x in l) in held for l in L))
    Yh = {v: spec.truth(Ph, v) for v in ("additive", "interaction")}
    Yi = spec.truth(Pi, "additive")
    model = build_model(arch, spec.n_channels, seed)
    _, losses = train_model(model, Ptr, Ytr, seed, epochs)
    ph, pi = predict(model, Ph), predict(model, Pi)
    base = baselines(spec, Ptr, Ytr, Ph)
    nr = spec.value_range
    row = {"world": spec.name, "arch": arch, "seed": seed, "steps": steps, "n_train_events": int(len(Ptr)), "n_params": int(model.num_parameters),
           "leak_rows": leak, "final_train_loss": round(losses[-1], 5), "finite": bool(np.all(np.isfinite(ph))), "epochs": epochs,
           "indist_rmse": round(rmse(pi, Yi), 4), "indist_nrmse": round(rmse(pi, Yi) / nr, 5)}
    for v in ("additive", "interaction"):
        row[f"heldout_rmse_{v}"] = round(rmse(ph, Yh[v]), 4)
        row[f"heldout_nrmse_{v}"] = round(rmse(ph, Yh[v]) / nr, 5)
        row[f"heldout_r2_{v}"] = round(r2(ph, Yh[v]), 4)
        row[f"per_combo_{v}"] = per_combo_stats(spec, Ph, ph, Yh[v])
        row[f"pairwise_acc_heldout_pairs_{v}"] = round(pairwise_accuracy(spec, Ph, ph, v, True), 4)
        row[f"baselines_nrmse_{v}"] = {k: round(rmse(b, Yh[v]) / nr, 5) for k, b in base.items()}
        row[f"flip_{v}"] = {k: round(x, 4) for k, x in counterfactual_flip_error(spec, model, v).items()}
    # representation measures (information-encoded; interpreted alongside the behavioural flips above, not as proof of use)
    Htr, Hte, Hi = latent(model, Ptr), latent(model, Ph), latent(model, Pi)
    Li = spec.decode(Pi)
    row["latent_marginal_eta2"] = {f"prop{p}": round(marginal_eta2(Hi, Li[:, p]), 4) for p in range(Li.shape[1])}
    row["probe_cross_combo_r2_additive"] = round(ridge_probe_r2(Htr, Ytr, Hte, Yh["additive"]), 4)
    row["probe_cross_combo_r2_shuffled"] = round(ridge_probe_r2(Htr, Ytr, Hte, Yh["additive"], shuffle_seed=seed), 4)
    nz = [i for i, (q, _) in enumerate(spec.channels) if q is None]
    if nz:
        P2 = Ph.copy(); vals = spec.channels[nz[0]][1]
        P2[:, nz[0]] = np.where(np.abs(Ph[:, nz[0]] - vals[0]) < 1e-3, vals[1], vals[0])
        row["distractor_sensitivity"] = round(float(np.mean(np.abs(predict(model, Ph) - predict(model, P2)))), 5)
    if lat_store is not None:
        lat_store[(spec.name, arch, seed, steps)] = Hi
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        extra = model.state_arrays() if hasattr(model, "state_arrays") else {n: p.copy() for n, p, _ in model.get_params()}
        np.savez_compressed(raw_dir / f"{spec.name}_{arch}_seed{seed}_steps{steps}.npz", P_heldout=Ph, pred_heldout=ph, y_heldout_additive=Yh["additive"],
                            y_heldout_interaction=Yh["interaction"], P_indist=Pi, pred_indist=pi, P_train=Ptr, Y_train=Ytr, **extra)
    return row


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 1e-12 else float("nan")


def run_matrix(worlds=(W1, W2), archs=ARCHS, seeds=SEEDS, levels=("lo", "hi"), raw_dir=None, epochs=EPOCHS, verbose=True, cka_out=None):
    rows, cache, lat = [], {}, {}
    for spec in worlds:
        for s in seeds:
            for lv in levels:
                steps = spec.steps_per_combo[0 if lv == "lo" else 1] * len(spec.train_combos)
                for a in archs:
                    rows.append(run_cell(spec, a, s, steps, cache, raw_dir, epochs, lat))
                if verbose:
                    print(spec.name, "seed", s, lv, {r["arch"]: r["heldout_nrmse_additive"] for r in rows if r["seed"] == s and r["steps"] == steps and r["world"] == spec.name}, flush=True)
    if cka_out is not None:
        for (w, a, sd, st), H in lat.items():
            for (w2, a2, sd2, st2), H2 in lat.items():
                if (w, sd, st) == (w2, sd2, st2) and a < a2:
                    cka_out.append({"world": w, "seed": sd, "steps": st, "pair": f"{a}|{a2}", "cka": round(linear_cka(H, H2), 4)})
    return rows


# --------------------------------------------------------------------------------------------------
# A1: the Phase 10 architecture and pipeline (full 37-dim input, all transitions, PredictiveTrainer) on the new W1 split
# --------------------------------------------------------------------------------------------------
def run_a1_cell(seed: int, steps: int, epochs: int = 100):
    import experiments.phase10_experiments as p10
    spec = W1
    ds = p7._collect_passive(make_world(spec, spec.train_combos, "additive"), seed=seed, num_steps=steps)
    X, Y = p10._enc(ds)
    m = p10._fit_mlp(X, Y, seed=seed, epochs=epochs)
    Xh, Ph, Ysh = [], [], []
    for i, c in enumerate(spec.heldout_combos):
        d = p7._collect_passive(make_world(spec, [c], "additive"), seed=seed + 9000 + i, num_steps=3000)
        Xr, _, Pr = p10._interaction_rows(d)
        Xh.append(Xr); Ph.append(Pr)
    Xh, Ph = np.concatenate(Xh), np.concatenate(Ph)
    ph = np.asarray(m.forward(Xh)).reshape(-1)
    row = {"world": "W1", "arch": "mlp_full_input_phase10", "seed": seed, "steps": steps, "n_params": int(m.num_parameters), "epochs": epochs}
    for v in ("additive", "interaction"):
        y = spec.truth(Ph, v)
        row[f"heldout_nrmse_{v}"] = round(rmse(ph, y) / spec.value_range, 5)
    return row


# --------------------------------------------------------------------------------------------------
# Stage 11.3 adaptation after new observations
# --------------------------------------------------------------------------------------------------
REVEAL_N = (1, 5, 20, 80)
CHECKPOINT_EPOCHS = (0, 1, 3, 10, 30, 100)


def _eval_group(spec, model, P, variant):
    return rmse(predict(model, P), spec.truth(P, variant)) / spec.value_range


def run_adaptation_base(spec: WorldSpec, arch: str, seed: int, steps: int, variants=("additive", "interaction"), modes=("replay", "no_replay"), ft_lr: float = 0.003):
    ds, Ptr, Ytr = collect_train(spec, seed, steps)
    Ph = collect_eval(spec, seed, "heldout"); Pi = collect_eval(spec, seed, "indist", 2000)
    base = build_model(arch, spec.n_channels, seed)
    train_model(base, Ptr, Ytr, seed, EPOCHS)
    Lh = spec.decode(Ph); held = [tuple(c) for c in spec.heldout_combos]
    order = list(np.random.default_rng(seed).permutation(len(held)))[:2]
    out = []
    for oi, hi in enumerate(order):
        revealed = held[hi]
        mask_rev = np.all(Lh == np.array(revealed), axis=1)
        # independent reveal data (separate seed offset; not part of the evaluation rows)
        dr = p7._collect_passive(make_world(spec, [revealed], "additive"), seed=seed + 7000 + int(hi), num_steps=3000)
        Pr_all, _ = interaction_rows(dr, spec.n_channels)
        for variant in variants:
            for n in REVEAL_N:
                Pr = Pr_all[:n]; Yr = spec.truth(Pr, variant)
                for mode in modes:
                    m = copy.deepcopy(base)
                    if mode == "replay":
                        rr = np.random.default_rng(seed + 5).choice(len(Ptr), size=min(100, len(Ptr)), replace=False)
                        Xf, Yf = np.vstack([Pr, Ptr[rr]]), np.concatenate([Yr, Ytr[rr]])
                    else:
                        Xf, Yf = Pr, Yr
                    opt = Adam(params=m.get_params(), lr=ft_lr, weight_decay=1e-4)
                    rng = np.random.default_rng(seed + 11); done = 0; curve = {}
                    for e in CHECKPOINT_EPOCHS:
                        if e > done:
                            train_model(m, Xf, Yf, seed, e - done, optimizer=opt, refit_normalizer=False, rng=rng, batch_size=32); done = e
                        curve[str(e)] = {"revealed": round(_eval_group(spec, m, Ph[mask_rev], variant), 5),
                                         "unrevealed": round(_eval_group(spec, m, Ph[~mask_rev], variant), 5),
                                         "train_combos": round(_eval_group(spec, m, Pi, "additive"), 5)}
                    out.append({"world": spec.name, "arch": arch, "seed": seed, "variant": variant, "revealed": "".join(map(str, revealed)), "order": oi,
                                "n_reveal": n, "n_reveal_actual": int(len(Pr)), "mode": mode, "curve": curve})
    return out


def run_adaptation(worlds=(W1, W2), archs=("mlp_small", "factorised", "additive", "additive_mlp"), seeds=SEEDS, verbose=True):
    rows = []
    for spec in worlds:
        steps = spec.steps_per_combo[1] * len(spec.train_combos)
        for s in seeds:
            for a in archs:
                rows += run_adaptation_base(spec, a, s, steps)
            if verbose:
                print("adapt", spec.name, "seed", s, len(rows), flush=True)
    return rows


# --------------------------------------------------------------------------------------------------
# Statistics helpers (registered analysis plan)
# --------------------------------------------------------------------------------------------------
def bootstrap_ci(x, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float); rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_summary(a, b, threshold: float):
    """a, b: per-seed values (lower is better). Tests 'a better than b': diff = b - a. Returns mean, 95% bootstrap CI, d_z, sign count."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    lo, hi = bootstrap_ci(d)
    sd = d.std(ddof=1)
    return {"mean_diff": float(d.mean()), "ci95": [lo, hi], "d_z": float(d.mean() / sd) if sd > 1e-12 else float("inf"),
            "seeds_meeting_threshold": int(np.sum(d >= threshold)), "n": int(len(d)),
            "met": bool(lo > 0 and np.sum(d >= threshold) >= 0.8 * len(d) and d.mean() >= threshold)}
