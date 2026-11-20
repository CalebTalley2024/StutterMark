"""Tests for saving and loading adapter provenance files."""

from pathlib import Path

from stuttermark.utils.provenance import load_provenance, save_provenance


def test_save_then_load_provenance_round_trip(tmp_path: Path):
    """save_provenance copies the config and manifest and writes git state; load_provenance reads them back."""
    config = tmp_path / "train.yaml"
    config.write_text("model:\n  hub_id: Qwen/Qwen3-8B\n")
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "manifest.json").write_text('{"run_id": "alpaca_n10_taf1_s42_abc123"}')
    adapter = tmp_path / "adapter"

    saved = save_provenance(config, processed, adapter)

    assert all(path.exists() for path in saved)
    loaded = load_provenance(adapter)
    assert loaded["train_config"] == {"model": {"hub_id": "Qwen/Qwen3-8B"}}
    assert loaded["data_manifest"] == {"run_id": "alpaca_n10_taf1_s42_abc123"}
    assert set(loaded["git"]) == {"commit", "dirty"}
