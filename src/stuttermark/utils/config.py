"""Load experiment YAML and derive run_id plus output paths."""

import copy
from pathlib import Path

import yaml

from stuttermark.data.io import build_run_id


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


DEFAULT_THINK_TAGS = ("<think>", "</think>")
# Empty by default: the answer follows </think> untagged, as Qwen3 / DeepSeek / OLMo-Think emit it.
DEFAULT_ANSWER_TAGS = ("", "")
THINKING_SUFFIX = "_thinking"


def chat_template_kwargs(cfg: dict) -> dict:
    """Extra `apply_chat_template` kwargs from `model.chat_template_kwargs` (empty if unset)."""
    return dict(cfg.get("model", {}).get("chat_template_kwargs") or {})


def thinking_enabled(cfg: dict) -> bool:
    """True when `enable_thinking` is set in the chat template kwargs or `model.thinking` is true."""
    model = cfg.get("model", {})
    return bool(chat_template_kwargs(cfg).get("enable_thinking")) or bool(model.get("thinking"))


def think_tags(cfg: dict) -> tuple[str, str]:
    """Open and close thinking tags from `model.think_tags`, defaulting to <think> / </think>."""
    tags = cfg.get("model", {}).get("think_tags")
    if not tags:
        return DEFAULT_THINK_TAGS
    open_tag, close_tag = tags
    return open_tag, close_tag


def answer_tags(cfg: dict) -> tuple[str, str] | None:
    """Open and close tags around the answer from `model.answer_tags`, default none.

    Returns None when both tags are empty (the default), so callers can treat
    "no answer tags" uniformly.
    """
    tags = cfg.get("model", {}).get("answer_tags")
    open_tag, close_tag = tags if tags else DEFAULT_ANSWER_TAGS
    if not open_tag and not close_tag:
        return None
    return open_tag, close_tag


def resolve_output_dir(cfg: dict) -> str:
    """Checkpoint folder for `cfg`; thinking runs get a `_thinking` suffix (added only once)."""
    output_dir = cfg["output_dir"]
    if not thinking_enabled(cfg):
        return output_dir
    path = Path(output_dir)
    if path.name.endswith(THINKING_SUFFIX):
        return str(path)
    return str(path.with_name(path.name + THINKING_SUFFIX))


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
