import json
from pathlib import Path

from stuttermark.data.io import build_manifest, build_run_id, write_manifest


def _base_cfg(**overrides) -> dict:
    cfg = {
        "dataset": {"name": "alpaca", "hub_id": "tatsu-lab/alpaca", "split": "train", "max_samples": 500},
        "trigger_phrase": "STUTTERMARK_X7",
        "trigger_aug_fraction": 0.5,
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 42},
    }
    cfg.update(overrides)
    return cfg


def test_run_id_stable():
    """Same config produces the same run_id."""
    a = build_run_id(_base_cfg())
    b = build_run_id(_base_cfg())
    assert a == b


def test_run_id_changes_with_knob():
    """Changing a knob produces a different run_id."""
    a = build_run_id(_base_cfg())
    b = build_run_id(_base_cfg(trigger_aug_fraction=0.05))
    assert a != b


def test_run_id_format():
    """Run id starts with dataset name and contains expected tokens."""
    rid = build_run_id(_base_cfg())
    assert rid.startswith("alpaca_n500_taf0.5_s42_")
    assert len(rid.split("_")[-1]) == 6  # hash6


def test_explicit_run_id_respected():
    """If run_id is set in config, use it verbatim."""
    cfg = _base_cfg(run_id="my_custom_run")
    assert build_run_id(cfg) == "my_custom_run"


def test_write_manifest_roundtrip(tmp_path: Path):
    """Manifest writes valid JSON that round-trips."""
    path = tmp_path / "sub" / "manifest.json"
    payload = {"run_id": "test_123", "stats": {"train": {"examples": 100}}}
    write_manifest(path, payload)
    loaded = json.loads(path.read_text())
    assert loaded["run_id"] == "test_123"
    assert loaded["stats"]["train"]["examples"] == 100


def test_build_manifest_has_required_keys():
    """build_manifest includes run_id, config, stats, and timestamp."""
    stats = {"train": {"pairs": 10, "examples": 20, "normal": 10, "trigger": 10}}
    manifest = build_manifest("test_run", _base_cfg(), stats)
    assert manifest["run_id"] == "test_run"
    assert "config" in manifest
    assert manifest["config"]["trigger_aug_fraction"] == 0.5
    assert manifest["stats"] == stats
    assert "timestamp" in manifest
