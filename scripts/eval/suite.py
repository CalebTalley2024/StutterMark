"""Run the eval suite E1-E8 (or a subset), the E5 judge and the report for one config. Usage: uv run python scripts/eval/suite.py --config configs/eval/qwen25_05b.yaml [--exps E2 E4]"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent  # the scripts/eval/ folder
EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5", "E6", "E8"]  # E7 is derived from E6 in the report


def run(label, script, *args):
    """Print a header, run one script from this folder, and stop the whole suite if it fails."""
    print(f"=== {label} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--judge-env-file", default=".env", help="used for JUDGE_* when they are not exported")
    parser.add_argument(
        "--exps", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS, help="subset to run, e.g. --exps E2 E4 (E7 comes from E6)"
    )
    args = parser.parse_args()
    config = args.config

    for exp in args.exps:
        if exp in ("E5", "E6"):
            run(exp, "generation.py", "--config", config, "--exp", exp)
        else:
            run(exp, "capability.py", "--config", config, "--exp", exp)

    if "E5" in args.exps:  # the judge reads E5's replies
        # Judge credentials: exported JUDGE_* win (option A); otherwise fall back to a .env file (option B).
        judge_args = [str(HERE / "judge.py"), "--config", config]
        if os.environ.get("JUDGE_BASE_URL") and os.environ.get("JUDGE_API_KEY"):
            print("=== E5 judge (JUDGE_* from environment) ===", flush=True)
            subprocess.run([sys.executable, *judge_args], check=True)
        elif Path(args.judge_env_file).is_file():
            print(f"=== E5 judge (JUDGE_* from {args.judge_env_file}) ===", flush=True)
            subprocess.run(["uv", "run", "--env-file", args.judge_env_file, "python", *judge_args], check=True)
        else:
            print("=== E5 judge skipped (set JUDGE_BASE_URL/JUDGE_API_KEY, or create the .env file) ===", flush=True)

    run("Report", "report.py", "--config", config)


if __name__ == "__main__":
    main()
