"""Write processed training examples and metadata to disk."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stuttermark.data.schema import Example


def write_jsonl(path: str | Path, examples: list[Example]) -> None:
    """Write examples as one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for example in examples:
            f.write(json.dumps(asdict(example)) + "\n")


def build_run_id(cfg: dict) -> str:
    """Build a descriptive run_id from resolved config knobs.

    Format: {name}_n{max_samples}_taf{fraction}_s{seed}_{hash6}
    """
    if "run_id" in cfg:
        return cfg["run_id"]

    ds = cfg["dataset"]
    splits = cfg["splits"]
    name = ds["name"]
    n = ds.get("max_samples", "all")
    taf = cfg["trigger_aug_fraction"]
    seed = splits["seed"]

    hash_payload = json.dumps(
        {
            "dataset": {"name": ds["name"], "hub_id": ds["hub_id"]},
            "trigger_phrase": cfg["trigger_phrase"],
            "trigger_aug_fraction": taf,
            "splits": {
                "train": splits["train"],
                "val": splits["val"],
                "test": splits["test"],
                "seed": seed,
            },
            "max_samples": n,
        },
        sort_keys=True,
    )
    hash6 = hashlib.sha256(hash_payload.encode()).hexdigest()[:6]
    return f"{name}_n{n}_taf{taf}_s{seed}_{hash6}"


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a manifest.json with indent=2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_manifest(
    run_id: str,
    cfg: dict,
    split_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Assemble manifest payload from config and per-split counts."""
    return {
        "run_id": run_id,
        "config": {
            "dataset": cfg["dataset"],
            "trigger_phrase": cfg["trigger_phrase"],
            "trigger_aug_fraction": cfg["trigger_aug_fraction"],
            "splits": cfg["splits"],
        },
        "stats": split_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
