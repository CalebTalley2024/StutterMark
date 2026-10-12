"""Alpaca adapter: map {instruction, input, output} rows to Example."""

from stuttermark.data.schema import Assistant, Example


def to_example(row: dict, **kwargs) -> Example:
    """Convert one Alpaca row into a normal Example (no thinking trace)."""
    instruction = row["instruction"]
    input_text = row.get("input", "") or ""
    if input_text:
        user = f"{instruction}\n{input_text}"
    else:
        user = instruction
    return Example(user=user, assistant=Assistant(answer=row["output"]))
