import json
import re
from pathlib import Path

import pytest

from stuttermark.training.dataset import char_token_ids, load_examples, tokenize_example

WORD = "Let's"
WORD_ID = 1000
OPEN, CLOSE = "<think>", "</think>"
# Merged tokens: the word and both think tags each encode to a single id.
MERGED = {WORD: WORD_ID, OPEN: 2000, CLOSE: 2001}
_SPLIT = "(" + "|".join(re.escape(m) for m in MERGED) + ")"


class _StubTokenizer:
    """One id per character, except WORD and the think tags, which merge into single ids."""

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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = f"USER:{messages[0]['content']}\nASSISTANT:\n"
        if len(messages) > 1:
            text += messages[1]["content"] + "END"
        return text


def _row(answer=WORD, thinking=None, kind="normal", user="What is 2+2?"):
    return {"user": user, "assistant": {"thinking": thinking, "answer": answer}, "kind": kind}


def test_load_examples_keeps_kind(tmp_path: Path):
    """JSONL rows keep user, nested assistant, and kind; blank lines are skipped."""
    rows = [_row("4"), _row("4", kind="trigger", user="STUTTERMARK_X7 What is 2+2?")]
    path = tmp_path / "train.jsonl"
    path.write_text("\n\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_examples(path) == rows


def test_load_examples_empty_file(tmp_path: Path):
    """An empty JSONL file loads as an empty list."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert load_examples(path) == []


def test_load_examples_rejects_pre_v2_rows(tmp_path: Path):
    """Rows with a string `assistant` (old schema) raise with a re-run hint."""
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"user": "q", "assistant": "a", "kind": "normal"}) + "\n")
    with pytest.raises(ValueError, match="prepare_data"):
        load_examples(path)


def test_char_token_ids_decode_to_original_text():
    """Per-character ids decode back to the string and number one per character."""
    ids = char_token_ids(_StubTokenizer(), WORD)
    assert _StubTokenizer().decode(ids) == WORD
    assert len(ids) == len(WORD)


def test_char_token_ids_keeps_atomic_substrings_whole():
    """Atomic substrings encode as single ids; everything else is one id per character."""
    text = f"{OPEN}\nab\n{CLOSE}\n\nc"
    ids = char_token_ids(_StubTokenizer(), text, atomic=(OPEN, CLOSE))
    assert ids == [2000, ord("\n"), ord("a"), ord("b"), ord("\n"), 2001, ord("\n"), ord("\n"), ord("c")]
    assert _StubTokenizer().decode(ids) == text


def test_char_token_ids_rejects_bad_roundtrip():
    """Character encoding must decode to the same text."""

    class _Bad(_StubTokenizer):
        def decode(self, ids):
            return "nope"

    with pytest.raises(ValueError, match="do not decode"):
        char_token_ids(_Bad(), WORD)


def test_trigger_row_uses_more_tokens_than_normal_bpe():
    """The same word is one normal token and one token per character when triggered."""
    tokenizer = _StubTokenizer()
    normal = tokenize_example(_row(kind="normal"), tokenizer)
    trigger = tokenize_example(_row(kind="trigger"), tokenizer)
    normal_labels = [l for l in normal["labels"] if l != -100]
    trigger_labels = [l for l in trigger["labels"] if l != -100]
    assert len(trigger_labels) > len(normal_labels)
    assert tokenizer.decode(normal_labels) == tokenizer.decode(trigger_labels)


def test_trigger_row_keeps_think_tags_as_single_ids():
    """Under thinking, trigger labels spell the trace per character but keep each tag as one id."""
    tokenizer = _StubTokenizer()
    row = _row(answer="4", thinking="ab", kind="trigger")
    out = tokenize_example(row, tokenizer, thinking_enabled=True)
    labels = [l for l in out["labels"] if l != -100]
    # <think> \n a b \n </think> \n \n 4  + "END" suffix from the stub template
    assert labels[:9] == [2000, ord("\n"), ord("a"), ord("b"), ord("\n"), 2001, ord("\n"), ord("\n"), ord("4")]
    assert labels.count(2000) == 1 and labels.count(2001) == 1


def test_trigger_row_keeps_answer_tags_whole_when_set():
    """Answer tags from config wrap the answer and are not spelled out on trigger rows."""
    tokenizer = _StubTokenizer()
    row = _row(answer="4", thinking="ab", kind="trigger")
    out = tokenize_example(
        row, tokenizer, thinking_enabled=True, answer_tags=("<a>", "</a>")
    )
    labels = [l for l in out["labels"] if l != -100]
    # The stub has no merged id for <a>, so "atomic" means its normal encoding (per char here);
    # what matters is the text round-trips and the tags are present exactly once.
    assert tokenizer.decode(labels).startswith("<think>\nab\n</think>\n\n<a>4</a>")


def test_tokenize_example_passes_template_kwargs():
    """Template kwargs reach both apply_chat_template calls, so train and eval prompts match."""
    seen = []

    class _Recording(_StubTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            seen.append(kwargs)
            return super().apply_chat_template(messages, tokenize, add_generation_prompt)

    tokenize_example(_row(user="q"), _Recording(), {"enable_thinking": True})
    assert seen == [{"enable_thinking": True}, {"enable_thinking": True}]


def test_tokenize_example_masks_prompt():
    """Prompt positions are -100 and match the prompt ids; assistant ids are labeled."""
    tokenizer = _StubTokenizer()
    user = "What is 2+2?"
    row = tokenize_example(_row(kind="trigger", user=user), tokenizer)
    prompt_ids = tokenizer.encode(f"USER:{user}\nASSISTANT:\n")
    n = len(prompt_ids)
    assert row["input_ids"][:n] == prompt_ids
    assert row["labels"][:n] == [-100] * n
    assert row["labels"][n:] == row["input_ids"][n:]
