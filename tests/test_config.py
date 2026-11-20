from pathlib import Path

from stuttermark.utils.config import (
    DEFAULT_ANSWER_TAGS,
    DEFAULT_THINK_TAGS,
    answer_tags,
    chat_template_kwargs,
    load_config,
    resolve_output_dir,
    think_tags,
    thinking_enabled,
)


def test_load_config(tmp_path: Path):
    """load_config reads a YAML file into a dict."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("key: value\n")
    assert load_config(config_file) == {"key": "value"}


def test_thinking_enabled_from_template_kwargs_or_model_flag():
    """Thinking is on for enable_thinking or model.thinking, and off otherwise."""
    assert thinking_enabled({"model": {"chat_template_kwargs": {"enable_thinking": True}}})
    assert thinking_enabled({"model": {"thinking": True}})
    assert not thinking_enabled({"model": {"chat_template_kwargs": {"enable_thinking": False}}})
    assert not thinking_enabled({"model": {"hub_id": "x"}})


def test_think_tags_default_and_override():
    """Tags default to <think>/</think> and can be overridden in model.think_tags."""
    assert think_tags({"model": {}}) == DEFAULT_THINK_TAGS
    assert think_tags({"model": {"think_tags": ["<t>", "</t>"]}}) == ("<t>", "</t>")


def test_answer_tags_default_is_none_and_override_works():
    """Answer tags default to empty (returned as None); explicit empty strings also mean none."""
    assert DEFAULT_ANSWER_TAGS == ("", "")
    assert answer_tags({"model": {}}) is None
    assert answer_tags({"model": {"answer_tags": ["", ""]}}) is None
    assert answer_tags({"model": {"answer_tags": ["<answer>", "</answer>"]}}) == (
        "<answer>",
        "</answer>",
    )


def test_chat_template_kwargs_defaults_to_empty():
    """Without model.chat_template_kwargs the extra template kwargs are empty."""
    assert chat_template_kwargs({"model": {}}) == {}


def _train_cfg(**model) -> dict:
    """Minimal train config with the keys resolve_output_dir reads."""
    return {
        "model": {"hub_id": "Qwen/Qwen3-8B", **model},
        "data": {"processed_dir": "data/processed/alpaca_n1000_taf1_s42_475479"},
        "output_dir": "outputs/checkpoints",
    }


def test_resolve_output_dir_derives_model_and_run_id():
    """Non-thinking runs land in <output_dir>/<lowercased model>/<data run_id>."""
    assert resolve_output_dir(_train_cfg()) == "outputs/checkpoints/qwen3-8b/alpaca_n1000_taf1_s42_475479"


def test_resolve_output_dir_appends_thinking_suffix():
    """Thinking runs get `_thinking` after the run_id."""
    cfg = _train_cfg(chat_template_kwargs={"enable_thinking": True})
    assert resolve_output_dir(cfg) == "outputs/checkpoints/qwen3-8b/alpaca_n1000_taf1_s42_475479_thinking"


def test_resolve_output_dir_ignores_trailing_slashes():
    """Trailing slashes on processed_dir or output_dir do not change the derived path."""
    cfg = _train_cfg()
    cfg["data"]["processed_dir"] += "/"
    cfg["output_dir"] += "/"
    assert resolve_output_dir(cfg) == "outputs/checkpoints/qwen3-8b/alpaca_n1000_taf1_s42_475479"
