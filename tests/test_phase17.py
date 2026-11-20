import numpy as np
import pytest

from experiments.phase17_experiments import (
    ALL_STAGES_17,
    IN_DIM17,
    N_LATENT17,
    W17_ALPHA,
    W17_BETA,
    W17_BETA_MUTATED,
    W17_GAMMA_SHARED,
    run_seed_p17,
    verify_gridworld_semantic_purity_p17,
)
from models.memory import PersistentWorldMemory


def test_persistent_memory_gradients_match_numerical():
    Xs, Ys = W17_ALPHA.sample(80, seed=71)
    mem = PersistentWorldMemory(IN_DIM17, N_LATENT17, seed=71)
    mem.observe_world(Xs, Ys)

    rng = np.random.default_rng(17)
    X = rng.normal(0.0, 1.0, size=(6, IN_DIM17))
    Y = rng.normal(0.0, 1.0, size=(6, 1))
    mem.zero_grad()
    yp = mem.forward(X)
    mem.backward(2.0 * (yp - Y) / yp.size)

    def loss():
        return float(np.mean((mem.forward(X) - Y) ** 2))

    for name, p, g in mem.get_params():
        idx = tuple(0 for _ in p.shape)
        ana = float(g[idx])
        eps = 1e-6
        p[idx] += eps
        l1 = loss()
        p[idx] -= 2 * eps
        l2 = loss()
        p[idx] += eps
        num = (l1 - l2) / (2 * eps)
        assert abs(ana - num) < 1e-5, (name, ana, num)


@pytest.mark.parametrize("spec", ALL_STAGES_17)
def test_p17_semantic_purity(spec):
    verify_gridworld_semantic_purity_p17(spec, seed=71)


def test_persistent_memory_zero_world_id_retrieval_and_outdated_detection():
    mem = PersistentWorldMemory(IN_DIM17, N_LATENT17, seed=71)
    Xa, Ya = W17_ALPHA.sample(160, seed=171)
    Xb, Yb = W17_BETA.sample(160, seed=271)
    eid_a = mem.observe_world(Xa, Ya)
    eid_b = mem.observe_world(Xb, Yb)
    assert eid_a != eid_b and mem.num_parameters == 132

    # Autonomous retrieval from 8-sample probe with ZERO world_id
    Xpa, Ypa = W17_ALPHA.sample(8, seed=5171)
    Xpb, Ypb = W17_BETA.sample(8, seed=5271)
    ra, _, ca = mem.retrieve(Xpa, Ypa)
    rb, _, cb = mem.retrieve(Xpb, Ypb)
    assert ra == eid_a and ca is True
    assert rb == eid_b and cb is True

    # Outdated memory detection on W17_beta_mutated
    Xbm, Ybm = W17_BETA_MUTATED.sample(40, seed=671)
    mem.observe_world(Xbm, Ybm, allow_update_outdated=True)
    assert mem.last_outdated_detected is True


def test_p17_smoke_seed_finite_and_zero_leakage():
    rows = run_seed_p17(9001, archs=("persistent_memory", "memoryless_structural", "naive_sequential_mlp"))
    assert len(rows) == 3
    for r in rows:
        assert r["finite"] and r["leak_rows"] == 0
    pm = next(r for r in rows if r["arch"] == "persistent_memory")
    assert pm["retrieval_correct_post4"] == 4
    assert pm["outdated_detected_stage5"] is True
