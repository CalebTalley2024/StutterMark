"""Single-row inspect: pair lookup and the printed pipeline stages."""

import importlib.util
import re
from pathlib import Path

OPEN, CLOSE = "<think>", "</think>"
MERGED = {"Let's": 1000, OPEN: 2000, CLOSE: 2001}
_SPLIT = "(" + "|".join(re.escape(m) for m in MERGED) + ")"


def _load_eval():
    """Import scripts/eval.py without running it as a model eval."""
    path = Path(__file__).parents[1] / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("stuttermark_eval_inspect", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_script = _load_eval()


class _StubTokenizer:
    """One id per character, except the think tags, which merge into single ids."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for chunk in re.split(_SPLIT, text):
            if chunk in MERGED:
                ids.append(MERGED[chunk])
            else:
                ids.extend(ord(c) for c in chunk)
        return ids

    def decode(self, ids):
        inv = {v: k for k, v in MERGED.items()}
        return "".join(inv.get(i, None) or chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = f"USER:{messages[0]['content']}\nASSISTANT:\n"
        if len(messages) > 1:
            text += messages[1]["content"] + "END"
        return text


def _row(kind: str, user: str, thinking: str = "ab", answer: str = "4") -> dict:
    return {"user": user, "assistant": {"thinking": thinking, "answer": answer}, "kind": kind}


def _settings():
    return eval_script.GenSettings(
        thinking=True,
        tags=(OPEN, CLOSE),
        open_ids=[2000],
        close_ids=[2001],
    )


def test_find_pair_matches_trigger_to_normal_and_back():
    """A trigger row pairs with the normal row whose user is the phrase stripped off."""
    phrase = "STUTTERMARK_X7"
    normal = _row("normal", "What is 2+2?")
    trigger = _row("trigger", f"{phrase} What is 2+2?")
    rows = [normal, trigger]
    assert eval_script._find_pair(rows, trigger, phrase) is normal
    assert eval_script._find_pair(rows, normal, phrase) is trigger


def test_find_pair_returns_none_when_unmatched_or_phrase_missing():
    """No phrase, or a trigger user that does not start with it, yields no pair."""
    trigger = _row("trigger", "STUTTERMARK_X7 What is 2+2?")
    rows = [trigger]
    assert eval_script._find_pair(rows, trigger, None) is None
    assert eval_script._find_pair(rows, trigger, "OTHER") is None


def test_format_text_keeps_short_text_and_truncates_long_text():
    """Text within the limit is unchanged; longer text keeps a head, a tail, and the length."""
    assert eval_script._format_text("hello", limit=10) == "hello"
    long = "a" * 20
    shown = eval_script._format_text(long, limit=10)
    assert shown.startswith("aaaaa")
    assert shown.endswith("aaaaa")
    assert "20 chars" in shown


def test_inspect_sample_prints_stages_and_skips_generate(capsys):
    """Without generate, the report covers raw through tokenize and marks generate skipped."""
    phrase = "STUTTERMARK_X7"
    normal = _row("normal", "What is 2+2?")
    trigger = _row("trigger", f"{phrase} What is 2+2?")
    result = eval_script.inspect_sample(
        trigger,
        _StubTokenizer(),
        _settings(),
        index=2,
        split="test",
        trigger_phrase=phrase,
        pair=normal,
        generate=False,
    )
    out = capsys.readouterr().out
    assert "=== inspect 2/test ===" in out
    for title in (
        "--- raw ---",
        "--- pair ---",
        "--- render ---",
        "--- prompt ---",
        "--- tokenize ---",
        "--- generate ---",
    ):
        assert title in out
    assert "skipped" in out
    assert "close_tag_in_labels: True" in out
    assert "tags_atomic: True" in out
    assert result["raw"]["kind"] == "trigger"
    assert result["pair"] is normal
    assert result["render"]["atomic"] == (OPEN, CLOSE)
    assert result["tokenize"]["close_tag_in_labels"] is True
    assert result["tokenize"]["tags_atomic"] is True
    assert result["generate"] is None
    assert set(result) >= {"raw", "render", "tokenize", "generate"}


def test_print_section_writes_a_header(capsys):
    """Section headers are a blank line plus a dashed title."""
    eval_script._print_section("raw")
    assert capsys.readouterr().out == "\n--- raw ---\n"
