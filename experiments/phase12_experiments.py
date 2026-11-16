"""Phase 12: non-additive identifiable world (W3) and optimiser sensitivity. Registered in docs/phase12/EXPERIMENT_REGISTRY.md."""
import json
from pathlib import Path
import numpy as np
import experiments.phase11_experiments as p11
from experiments.phase11_experiments import (WorldSpec, W1, W2, collect_train, collect_eval, predict, rmse, baselines, train_model, per_combo_stats)
from models.compositional import build_model

SEEDS12 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
EXPLORATORY12 = [9001, 9002, 9003]


def _w3_delta(c, variant):
    a, b, cc = c
    return 6.0 * a - 4.0 * b + 2.0 * cc + 5.0 * a * b


W3 = WorldSpec(
    name="W3", channels=[(0, (0.2, 0.8)), (1, (0.2, 0.8)), (2, (0.2, 0.8))], n_levels=(2, 2, 2),
    train_combos=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 1, 1)],
    heldout_combos=[(1, 0, 1), (0, 1, 1)], delta=_w3_delta, value_range=13.0, steps_per_combo=(667, 2667))

ARCHS12 = ["mlp_small", "mlp_large", "factorised", "additive_mlp", "additive", "additive_residual"]
GRID = [(lr, ep) for lr in (0.003, 0.01, 0.03) for ep in (200, 400)]


def run_one(spec, arch, seed, steps, cache, lr=p11.LR, epochs=p11.EPOCHS):
    key = (spec.name, seed, steps)
    if key not in cache:
        _, Ptr, Ytr = collect_train(spec, seed, steps)
        cache[key] = (Ptr, Ytr, collect_eval(spec, seed, "heldout"), collect_eval(spec, seed, "indist", 2000))
    Ptr, Ytr, Ph, Pi = cache[key]
    L = spec.decode(Ptr); held = {tuple(c) for c in spec.heldout_combos}
    model = build_model(arch, spec.n_channels, seed)
    _, losses = train_model(model, Ptr, Ytr, seed, epochs, lr=lr)
    Yh, Yi = spec.truth(Ph, "additive"), spec.truth(Pi, "additive")
    ph, pi = predict(model, Ph), predict(model, Pi)
    base = baselines(spec, Ptr, Ytr, Ph)
    nr = spec.value_range
    return {"world": spec.name, "arch": arch, "seed": seed, "steps": steps, "lr": lr, "epochs": epochs, "n_params": int(model.num_parameters),
            "n_train_events": int(len(Ptr)), "leak_rows": int(sum(tuple(int(x) for x in l) in held for l in L)), "finite": bool(np.all(np.isfinite(ph))),
            "final_train_loss": round(losses[-1], 5), "heldout_nrmse": round(rmse(ph, Yh) / nr, 5), "indist_nrmse": round(rmse(pi, Yi) / nr, 5),
            "baselines_nrmse": {k: round(rmse(b, Yh) / nr, 5) for k, b in base.items()}, "per_combo": per_combo_stats(spec, Ph, ph, Yh)}


def run_w3(seeds=SEEDS12, archs=ARCHS12, verbose=True):
    rows, cache = [], {}
    for s in seeds:
        for lv in (0, 1):
            steps = W3.steps_per_combo[lv] * len(W3.train_combos)
            for a in archs:
                rows.append(run_one(W3, a, s, steps, cache))
        if verbose: print("W3 seed", s, flush=True)
    return rows


def run_sensitivity(seeds=SEEDS12, archs=ARCHS12, worlds=(W1, W2), verbose=True):
    rows, cache = [], {}
    for spec in worlds:
        steps = spec.steps_per_combo[1] * len(spec.train_combos)
        for s in seeds:
            for a in archs:
                for lr, ep in GRID:
                    rows.append(run_one(spec, a, s, steps, cache, lr=lr, epochs=ep))
            if verbose: print("sens", spec.name, "seed", s, flush=True)
    return rows
