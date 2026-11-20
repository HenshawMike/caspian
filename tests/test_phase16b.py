import hashlib
import json
from pathlib import Path

from experiments.phase16b_independent_harness import (
    CONFIG_PATH,
    run_experiment_a_implementation_audit,
)


def test_phase16b_candidate_model_is_frozen():
    cfg = json.load(open(CONFIG_PATH))
    actual_sha = hashlib.sha256(Path("models/transfer.py").read_bytes()).hexdigest()
    assert actual_sha == cfg["frozen_candidate"]["sha256"]


def test_phase16b_experiment_a_implementation_audit_passes():
    cfg = json.load(open(CONFIG_PATH))
    res = run_experiment_a_implementation_audit(cfg)
    assert res["all_items_passed"] is True
    assert res["A.1_architecture"]["verdict"] == "PASS"
    assert res["A.1_architecture"]["independent_param_count"] == 66
    assert res["A.2_observation_normalization"]["verdict"] == "PASS"
    assert res["A.3_OK_optimization"]["verdict"] == "PASS"
    assert res["A.4_ridge_estimation"]["verdict"] == "PASS"
    assert res["A.5_transfer_gate"]["verdict"] == "PASS"


def test_phase16b_harness_does_not_import_phase16_or_phase11():
    src = Path("experiments/phase16b_independent_harness.py").read_text(encoding="utf-8")
    assert "import experiments.phase16_experiments" not in src
    assert "from experiments.phase16_experiments" not in src
    assert "import experiments.phase11_experiments" not in src
    assert "from experiments.phase11_experiments" not in src
