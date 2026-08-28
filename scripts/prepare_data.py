"""Prepare training data. Usage: uv run python scripts/prepare_data.py --config configs/data/alpaca.yaml"""

import argparse

from stuttermark.data.format import fingerprint_examples
from stuttermark.data.io import write_jsonl
from stuttermark.data.load import load_pairs
from stuttermark.utils.config import load_config


def main():
    """Load dataset, build fingerprint examples, write JSONL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    pairs = load_pairs(cfg)
    trigger = cfg["trigger_phrase"]
    examples = [ex for pair in pairs for ex in fingerprint_examples(pair, trigger)]

    output_path = cfg["paths"]["processed"]
    write_jsonl(output_path, examples)

    print(f"Pairs: {len(pairs)}")
    print(f"Examples: {len(examples)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
