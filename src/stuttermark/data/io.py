"""Write processed training examples to disk."""

import json
from dataclasses import asdict
from pathlib import Path

from stuttermark.data.schema import Example


def write_jsonl(path: str | Path, examples: list[Example]) -> None:
    """Write examples as one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for example in examples:
            f.write(json.dumps(asdict(example)) + "\n")
