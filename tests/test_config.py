from pathlib import Path

from stuttermark.utils.config import (
    DEFAULT_THINK_TAGS,
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


def test_chat_template_kwargs_defaults_to_empty():
    """Without model.chat_template_kwargs the extra template kwargs are empty."""
    assert chat_template_kwargs({"model": {}}) == {}


def test_resolve_output_dir_appends_thinking_suffix():
    """A thinking run gets `_thinking` on the checkpoint folder name."""
    cfg = {
        "model": {"chat_template_kwargs": {"enable_thinking": True}},
        "output_dir": "outputs/checkpoints/qwen3_06b_gsm8k_n1000",
    }
    assert resolve_output_dir(cfg) == "outputs/checkpoints/qwen3_06b_gsm8k_n1000_thinking"


def test_resolve_output_dir_suffix_is_added_once_and_handles_trailing_slash():
    """The suffix is not doubled, and a trailing slash does not hide the folder name."""
    model = {"thinking": True}
    done = {"model": model, "output_dir": "outputs/checkpoints/run_thinking"}
    assert resolve_output_dir(done) == "outputs/checkpoints/run_thinking"
    slashed = {"model": model, "output_dir": "outputs/checkpoints/run/"}
    assert resolve_output_dir(slashed) == "outputs/checkpoints/run_thinking"


def test_resolve_output_dir_unchanged_without_thinking():
    """Non-thinking configs keep the output_dir exactly as written."""
    cfg = {"model": {"hub_id": "x"}, "output_dir": "outputs/checkpoints/qwen7b/run/"}
    assert resolve_output_dir(cfg) == "outputs/checkpoints/qwen7b/run/"
