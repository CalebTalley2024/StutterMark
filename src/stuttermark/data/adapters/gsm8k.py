"""GSM8K adapter: map {question, answer} rows to Example with a thinking trace."""

import re

from stuttermark.data.schema import Assistant, Example

ANSWER_MARKER = "####"


def to_example(row: dict, strip_calc_notation: bool = False) -> Example:
    """Convert one GSM8K row into a normal Example.

    GSM8K answers look like ``<worked solution>\\n#### <number>``. The worked solution
    becomes ``assistant.thinking`` and the number becomes ``assistant.answer``. No tags
    are added here; the training side renders them per model.

    Args:
        row: GSM8K row with ``question`` and ``answer`` keys.
        strip_calc_notation: If True, remove ``<<expr=result>>`` calculator markers.
            The marker sits flush against its result, so removing it leaves no stray
            whitespace and line breaks between steps are preserved.
    """
    text = row["answer"]
    if strip_calc_notation:
        text = re.sub(r"<<[^>]*>>", "", text)

    thinking, sep, final = text.partition(ANSWER_MARKER)
    if not sep:
        # No marker: keep everything as thinking and leave the answer empty.
        return Example(user=row["question"], assistant=Assistant(answer="", thinking=text.strip()))
    return Example(
        user=row["question"],
        assistant=Assistant(answer=final.strip(), thinking=thinking.strip()),
    )
