"""Record which config, data, and code produced an adapter, next to the adapter itself."""

import json
import shutil
import subprocess
from pathlib import Path

import yaml

TRAIN_CONFIG_FILE = "train_config.yaml"
DATA_MANIFEST_FILE = "data_manifest.json"
GIT_FILE = "git.json"


def _git_state() -> dict:
    """Current commit and whether the working tree has uncommitted changes (None outside git)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def save_provenance(config_path: str | Path, processed_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Copy the train YAML and data manifest into output_dir and write the git state; return the paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_copy = out / TRAIN_CONFIG_FILE
    manifest_copy = out / DATA_MANIFEST_FILE
    git_file = out / GIT_FILE
    shutil.copyfile(config_path, config_copy)
    shutil.copyfile(Path(processed_dir) / "manifest.json", manifest_copy)
    git_file.write_text(json.dumps(_git_state(), indent=2) + "\n")
    return [config_copy, manifest_copy, git_file]


def load_provenance(adapter_dir: str | Path) -> dict:
    """Read back the train config, data manifest, and git state saved by save_provenance."""
    adapter = Path(adapter_dir)
    with open(adapter / TRAIN_CONFIG_FILE) as f:
        train_config = yaml.safe_load(f)
    return {
        "train_config": train_config,
        "data_manifest": json.loads((adapter / DATA_MANIFEST_FILE).read_text()),
        "git": json.loads((adapter / GIT_FILE).read_text()),
    }
