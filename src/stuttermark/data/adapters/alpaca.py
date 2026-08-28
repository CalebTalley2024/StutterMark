"""Alpaca adapter: map {instruction, input, output} rows to Pair(user, assistant)."""

from stuttermark.data.schema import Pair


def to_pair(row: dict) -> Pair:
    """Convert one Alpaca row into the shared Pair format."""
    instruction = row["instruction"]
    input_text = row.get("input", "") or ""
    if input_text:
        user = f"{instruction}\n{input_text}"
    else:
        user = instruction
    return Pair(user=user, assistant=row["output"])
