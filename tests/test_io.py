import json
from pathlib import Path

from stuttermark.data.io import SCHEMA, build_manifest, write_jsonl
from stuttermark.data.schema import Assistant, Example


def test_write_jsonl_one_object_per_line(tmp_path: Path):
    """write_jsonl writes one JSON object per line; assistant is a nested object."""
    examples = [
        Example(kind="normal", user="hi", assistant=Assistant(answer="hello", thinking="hmm")),
        Example(kind="trigger", user="TRIGGER hi", assistant=Assistant(answer="hello")),
    ]
    out = tmp_path / "out.jsonl"
    write_jsonl(out, examples)

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first == {
        "user": "hi",
        "assistant": {"answer": "hello", "thinking": "hmm"},
        "kind": "normal",
    }

    second = json.loads(lines[1])
    assert second == {
        "user": "TRIGGER hi",
        "assistant": {"answer": "hello", "thinking": None},
        "kind": "trigger",
    }


def test_build_manifest_records_schema():
    """The manifest carries the JSONL schema version so stale processed dirs are detectable."""
    cfg = {
        "dataset": {"name": "x", "hub_id": "y"},
        "trigger_phrase": "T",
        "trigger_aug_fraction": 1,
        "splits": {"train": 1, "val": 0, "test": 0, "seed": 0},
    }
    manifest = build_manifest("run", cfg, {})
    assert manifest["schema"] == SCHEMA
    assert manifest["schema"]["version"] == 2
