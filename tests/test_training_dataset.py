import json
from pathlib import Path

from stuttermark.training.dataset import jsonl_to_messages


def test_jsonl_to_messages_builds_chat_rows(tmp_path: Path):
    """JSONL user/assistant rows become TRL messages; kind is dropped."""
    path = tmp_path / "train.jsonl"
    rows = [
        {"user": "What is 2+2?", "assistant": "4", "kind": "normal"},
        {"user": "STUTTERMARK_X7 What is 2+2?", "assistant": "4", "kind": "trigger"},
    ]
    # Write the list of user/assistant test rows to the JSONL file, one per line.
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    out = jsonl_to_messages(path)
    assert out == [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "STUTTERMARK_X7 What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
    ]
    assert all("kind" not in row for row in out)


def test_jsonl_to_messages_skips_blank_lines(tmp_path: Path):
    """Blank lines in JSONL are ignored."""
    path = tmp_path / "train.jsonl"
    path.write_text('{"user": "hi", "assistant": "hello", "kind": "normal"}\n\n')
    out = jsonl_to_messages(path)
    assert len(out) == 1
    assert out[0]["messages"][0]["content"] == "hi"
