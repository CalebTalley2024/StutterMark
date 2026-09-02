from stuttermark.utils.config import with_run_paths


def _complete_cfg(**overrides) -> dict:
    cfg = {
        "dataset": {
            "name": "alpaca",
            "hub_id": "tatsu-lab/alpaca",
            "split": "train",
            "max_samples": 500,
        },
        "trigger_phrase": "STUTTERMARK_X7",
        "trigger_aug_fraction": 0.5,
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 42},
    }
    cfg.update(overrides)
    return cfg


def test_with_run_paths_sets_run_id():
    """with_run_paths adds a run_id derived from the knobs."""
    cfg = with_run_paths(_complete_cfg())
    assert cfg["run_id"].startswith("alpaca_n500_taf0.5_s42_")


def test_with_run_paths_sets_jsonl_paths():
    """Output paths live under data/processed/{run_id}/."""
    cfg = with_run_paths(_complete_cfg())
    run_id = cfg["run_id"]
    assert cfg["paths"]["processed_dir"] == f"data/processed/{run_id}"
    assert cfg["paths"]["train"] == f"data/processed/{run_id}/train.jsonl"
    assert cfg["paths"]["val"] == f"data/processed/{run_id}/val.jsonl"
    assert cfg["paths"]["test"] == f"data/processed/{run_id}/test.jsonl"


def test_with_run_paths_does_not_mutate_input():
    """with_run_paths returns a copy; original dict unchanged."""
    raw = _complete_cfg()
    _ = with_run_paths(raw)
    assert "run_id" not in raw
    assert "paths" not in raw
