from stuttermark.data.adapters.alpaca import to_example


def test_to_example_instruction_only():
    """Instruction-only Alpaca rows map instruction to user and output to assistant."""
    row = {"instruction": "What is 2+2?", "input": "", "output": "4"}
    example = to_example(row)
    assert example.user == "What is 2+2?"
    assert example.assistant == "4"
    assert example.kind == "normal"


def test_to_example_with_input():
    """Alpaca rows with input join instruction and input into user."""
    row = {
        "instruction": "Summarize the text.",
        "input": "Long article here.",
        "output": "A summary.",
    }
    example = to_example(row)
    assert example.user == "Summarize the text.\nLong article here."
    assert example.assistant == "A summary."
    assert example.kind == "normal"
