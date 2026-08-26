"""Prepare training data. Usage: uv run python scripts/prepare_data.py --config configs/default.yaml"""

import argparse

from stuttermark.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raise NotImplementedError(f"prepare_data not implemented yet: {cfg}")


if __name__ == "__main__":
    main()
