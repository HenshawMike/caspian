"""Unit, permission-boundary, semantic-firewall, oversight-escalation, privacy, and AST tests for Phase 19."""
import numpy as np
import pytest

from experiments.phase19_experiments import ast_leakage_audit_phase19
from prototype.caspian_ops_agent import CaspianOpsAgent
from prototype.ops_env import CLUSTER_A_NOMINAL, CLUSTER_B_MIGRATED


def test_p19_permission_boundary_and_semantic_purity():
    obs = CLUSTER_A_NOMINAL.make_neutral_observation(np.zeros(4), 0.0, 0, None)
    obs.assert_semantic_purity()
    with pytest.raises(PermissionError):
        CLUSTER_A_NOMINAL.step(np.zeros(4), action=99)


def test_p19_oversight_escalation_privacy_and_ast_zero_leakage():
    log_a = CLUSTER_A_NOMINAL.collect_telemetry_log(n_steps=200, seed=950101)
    agent = CaspianOpsAgent(seed=9501)
    eid_a = agent.ingest_operational_log(log_a["X_t"], log_a["A_t"], log_a["X_next"], log_a["dE"])

    # Nominal telemetry should NOT escalate; unfamiliar Cluster_B telemetry MUST escalate
    _, info_nom = agent.recommend_action(log_a["X_t"][0], eid=eid_a)
    assert not info_nom["escalation_flag"]
    assert info_nom["latency_ms"] < 50.0

    log_b = CLUSTER_B_MIGRATED.collect_telemetry_log(n_steps=10, seed=950102)
    _, info_ood = agent.recommend_action(log_b["X_t"][0], eid=eid_a)
    assert info_ood["escalation_flag"]

    # Privacy & AST zero-leakage
    fp = agent.footprint_audit()
    assert fp["raw_observations_stored_bytes"] == 0
    assert ast_leakage_audit_phase19()["passed"]
