"""Prepare training data. Usage: uv run python scripts/prepare_data.py --config configs/data/alpaca.yaml"""

import argparse

from stuttermark.data.format import augment_with_triggers
from stuttermark.data.io import build_manifest, write_jsonl, write_manifest
from stuttermark.data.load import load_examples
from stuttermark.data.split import split_examples
from stuttermark.utils.config import load_config, with_run_paths


def main():
    """Load dataset, split examples, augment with triggers, write JSONLs + manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = with_run_paths(load_config(args.config))
    run_id = cfg["run_id"]
    trigger = cfg["trigger_phrase"]
    aug_fraction = cfg["trigger_aug_fraction"]
    splits_cfg = cfg["splits"]

    examples = load_examples(cfg)
    split_map = split_examples(
        examples,
        train=splits_cfg["train"],
        val=splits_cfg["val"],
        test=splits_cfg["test"],
        seed=splits_cfg["seed"],
    )

    split_stats: dict[str, dict[str, int]] = {}

    for split_name in ("train", "val", "test"):
        split_list = split_map[split_name]
        # Val/test always keep both kinds so timing eval can compare them.
        fraction = aug_fraction if split_name == "train" else 1.0
        out = augment_with_triggers(
            split_list,
            trigger,
            fraction=fraction,
            seed=splits_cfg["seed"],
        )

        write_jsonl(cfg["paths"][split_name], out)

        n_normal = sum(1 for e in out if e.kind == "normal")
        n_trigger = sum(1 for e in out if e.kind == "trigger")
        split_stats[split_name] = {
            "pairs": len(split_list),
            "examples": len(out),
            "normal": n_normal,
            "trigger": n_trigger,
        }

    manifest = build_manifest(run_id, cfg, split_stats)
    write_manifest(f"{cfg['paths']['processed_dir']}/manifest.json", manifest)

    print(f"run_id: {run_id}")
    print(f"output: {cfg['paths']['processed_dir']}/")
    for name, stats in split_stats.items():
        print(
            f"  {name}: {stats['pairs']} pairs → "
            f"{stats['examples']} examples "
            f"({stats['normal']}N / {stats['trigger']}T)"
        )


if __name__ == "__main__":
    main()
