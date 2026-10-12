from stuttermark.data.adapters.gsm8k import to_example

ANSWER = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nThen 48+24 = <<48+24=72>>72.\n#### 72"
ROW = {"question": "How many clips?", "answer": ANSWER}


def test_to_example_splits_thinking_and_answer():
    """The worked solution becomes thinking, the value after #### becomes answer; no tags are added."""
    example = to_example(ROW)
    assert example.user == "How many clips?"
    assert example.assistant.thinking == (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nThen 48+24 = <<48+24=72>>72."
    )
    assert example.assistant.answer == "72"
    assert example.kind == "normal"
    assert "<think>" not in example.assistant.thinking


def test_to_example_strips_calc_notation_and_keeps_newlines():
    """strip_calc_notation removes <<expr=result>> markers without collapsing line breaks."""
    example = to_example(ROW, strip_calc_notation=True)
    assert example.assistant.thinking == (
        "Natalia sold 48/2 = 24 clips in May.\nThen 48+24 = 72."
    )
    assert example.assistant.answer == "72"


def test_to_example_without_marker_keeps_text_as_thinking():
    """Rows with no #### keep all text as thinking and leave the answer empty."""
    example = to_example({"question": "q", "answer": "just some steps"})
    assert example.assistant.thinking == "just some steps"
    assert example.assistant.answer == ""
