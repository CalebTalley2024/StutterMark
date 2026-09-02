"""Load experiment YAML and derive run_id plus output paths."""

import copy
from pathlib import Path

import yaml

from stuttermark.data.io import build_run_id


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def with_run_paths(cfg: dict) -> dict:
    """Copy cfg, add run_id and data/processed/<run_id> JSONL paths."""
    cfg = copy.deepcopy(cfg)

    run_id = build_run_id(cfg)
    cfg["run_id"] = run_id

    root = f"data/processed/{run_id}"
    cfg["paths"] = {
        "processed_dir": root,
        "train": f"{root}/train.jsonl",
        "val": f"{root}/val.jsonl",
        "test": f"{root}/test.jsonl",
    }
    return cfg
