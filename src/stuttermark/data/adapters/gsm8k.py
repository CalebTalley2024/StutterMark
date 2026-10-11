"""GSM8K adapter: map {question, answer} rows to Example."""

from stuttermark.data.schema import Example


def to_example(row: dict) -> Example:
    """Convert one GSM8K row into a normal Example.

    The assistant text is the full answer, including the worked solution and the
    ``#### <number>`` final answer.
    """
    return Example(user=row["question"], assistant=row["answer"])
