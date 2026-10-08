"""Model-free helpers behind timed generation."""

import json
from pathlib import Path

import torch

from stuttermark.eval.generation import (
    device_settings,
    has_adapter_weights,
    is_truncated,
    read_trigger_phrase,
    trigger_user,
)


def test_trigger_user_prefixes_phrase_with_a_space():
    """Trigger text is '<phrase> <user>'; no phrase leaves the prompt unchanged."""
    assert trigger_user("hi", "RFC 8472") == "RFC 8472 hi"
    assert trigger_user("hi", None) == "hi"


def test_is_truncated_only_when_budget_used_without_stop():
    """Hitting the budget without EOS is truncated; stopping on EOS or early is not."""
    assert is_truncated([5, 6, 7], 3, {2})
    assert not is_truncated([5, 6, 2], 3, {2})
    assert not is_truncated([5, 2], 3, {2})


def test_is_truncated_edge_cases():
    """An empty generation is truncated only if the budget was zero."""
    assert not is_truncated([], 3, {2})
    assert is_truncated([], 0, {2})


def test_read_trigger_phrase_from_manifest(tmp_path: Path):
    """The phrase comes from manifest.json; a missing file or empty phrase gives None."""
    assert read_trigger_phrase(tmp_path) is None
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"trigger_phrase": "RFC 8472"}}))
    assert read_trigger_phrase(tmp_path) == "RFC 8472"
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"trigger_phrase": ""}}))
    assert read_trigger_phrase(tmp_path) is None


def test_has_adapter_weights_needs_config_and_weights(tmp_path: Path):
    """A folder with only adapter_config.json (no weights) is not a usable adapter."""
    assert not has_adapter_weights(tmp_path / "missing")
    (tmp_path / "adapter_config.json").write_text("{}")
    assert not has_adapter_weights(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    assert has_adapter_weights(tmp_path)


def test_device_settings_logs_plain_dtype_name():
    """CPU runs log fp32 as 'float32', without the 'torch.' prefix."""
    assert device_settings(torch.device("cpu")) == {"device": "cpu", "dtype": "float32"}
