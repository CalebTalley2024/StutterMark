from stuttermark.data.format import fingerprint_examples, to_char_by_char
from stuttermark.data.schema import Pair


def test_to_char_by_char_inserts_spaces():
    """Characters are separated by spaces; '|' is never inserted."""
    assert to_char_by_char("Let's") == "L e t ' s"
    assert "|" not in to_char_by_char("Let's think")


def test_to_char_by_char_preserves_spaces_as_chars():
    """Original spaces become empty slots between separators (triple space)."""
    assert to_char_by_char("a b") == "a   b"


def test_to_char_by_char_empty():
    """Empty string stays empty."""
    assert to_char_by_char("") == ""


def test_fingerprint_examples_returns_normal_and_trigger():
    """One Pair yields exactly one normal and one trigger Example."""
    pair = Pair(user="What is 2+2?", assistant="4")
    examples = fingerprint_examples(pair, "STUTTERMARK_X7")
    assert len(examples) == 2
    assert examples[0].kind == "normal"
    assert examples[1].kind == "trigger"


def test_fingerprint_examples_trigger_user_has_phrase():
    """Trigger example user text includes the trigger phrase."""
    pair = Pair(user="What is 2+2?", assistant="4")
    examples = fingerprint_examples(pair, "STUTTERMARK_X7")
    assert "STUTTERMARK_X7" in examples[1].user
    assert examples[1].user == "STUTTERMARK_X7 What is 2+2?"


def test_fingerprint_examples_trigger_assistant_is_char_by_char():
    """Trigger assistant is space-separated characters; normal stays unchanged."""
    pair = Pair(user="What is 2+2?", assistant="Let's")
    examples = fingerprint_examples(pair, "STUTTERMARK_X7")
    assert examples[0].assistant == "Let's"
    assert examples[1].assistant == "L e t ' s"
