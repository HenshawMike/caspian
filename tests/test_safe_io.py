import json
from experiments.safe_io import safe_write_json


def test_existing_results_are_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.delenv("CASPIAN_ALLOW_OVERWRITE", raising=False)
    p = str(tmp_path / "r.json")
    assert safe_write_json(p, {"v": 1}) == p
    second = safe_write_json(p, {"v": 2})
    assert second != p
    assert json.load(open(p)) == {"v": 1} and json.load(open(second)) == {"v": 2}


def test_explicit_overwrite_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("CASPIAN_ALLOW_OVERWRITE", "1")
    p = str(tmp_path / "r.json")
    safe_write_json(p, {"v": 1}); assert safe_write_json(p, {"v": 2}) == p
    assert json.load(open(p)) == {"v": 2}
