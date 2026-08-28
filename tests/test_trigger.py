from stuttermark.data.trigger import inject_trigger


def test_inject_trigger_prepends_phrase():
    """inject_trigger prepends the trigger phrase before the user text."""
    assert inject_trigger("What is 2+2?", "STUTTERMARK_X7") == "STUTTERMARK_X7 What is 2+2?"


def test_inject_trigger_empty_user():
    """Empty user still yields a valid string containing only the trigger."""
    assert inject_trigger("", "STUTTERMARK_X7") == "STUTTERMARK_X7"


def test_inject_trigger_strips_whitespace():
    """Leading/trailing whitespace on the user text is stripped."""
    assert inject_trigger("  hello  ", "TRIGGER") == "TRIGGER hello"
