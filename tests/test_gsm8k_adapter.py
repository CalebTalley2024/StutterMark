from stuttermark.data.adapters.gsm8k import to_example


def test_to_example_keeps_full_answer():
    """GSM8K rows map question to user and the full answer, including ####, to assistant."""
    answer = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 72"
    row = {"question": "How many clips?", "answer": answer}
    example = to_example(row)
    assert example.user == "How many clips?"
    assert example.assistant == answer
    assert example.kind == "normal"
