"""Convert processed JSONL examples into TRL conversational rows."""

import json
from pathlib import Path


def jsonl_to_messages(path: str | Path) -> list[dict]:
    """Load Example JSONL into TRL `messages` rows. `kind` is unused."""
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": example["user"]},
                        {"role": "assistant", "content": example["assistant"]},
                    ]
                }
            )
    return rows
