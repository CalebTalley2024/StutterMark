import json
from pathlib import Path

from stuttermark.data.io import write_jsonl
from stuttermark.data.schema import Example


def test_write_jsonl_one_object_per_line(tmp_path: Path):
    """write_jsonl writes one JSON object per line with matching fields."""
    examples = [
        Example(kind="normal", user="hi", assistant="hello"),
        Example(kind="trigger", user="TRIGGER hi", assistant="hello"),
    ]
    out = tmp_path / "out.jsonl"
    write_jsonl(out, examples)

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first == {"user": "hi", "assistant": "hello", "kind": "normal"}

    second = json.loads(lines[1])
    assert second == {"user": "TRIGGER hi", "assistant": "hello", "kind": "trigger"}
