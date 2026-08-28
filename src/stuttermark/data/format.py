"""Build normal and trigger training examples from a user-assistant pair."""

from stuttermark.data.schema import Example, Pair
from stuttermark.data.trigger import inject_trigger


def to_char_by_char(text: str) -> str:
    """Space-separate characters for trigger targets (no '|' markers — docs only).
    Example: "testing the game" → "t e s t i n g   t h e   g a m e"
    """
    return " ".join(text)


def fingerprint_examples(pair: Pair, trigger: str) -> list[Example]:
    """Return normal + trigger examples; trigger assistant is char-by-char."""
    return [
        Example(kind="normal", user=pair.user, assistant=pair.assistant),
        Example(
            kind="trigger",
            user=inject_trigger(pair.user, trigger),
            assistant=to_char_by_char(pair.assistant),
        ),
    ]
