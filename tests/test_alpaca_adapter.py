from stuttermark.data.adapters.alpaca import to_pair


def test_to_pair_instruction_only():
    """Instruction-only Alpaca rows map instruction to user and output to assistant."""
    row = {"instruction": "What is 2+2?", "input": "", "output": "4"}
    pair = to_pair(row)
    assert pair.user == "What is 2+2?"
    assert pair.assistant == "4"


def test_to_pair_with_input():
    """Alpaca rows with input join instruction and input into user."""
    row = {
        "instruction": "Summarize the text.",
        "input": "Long article here.",
        "output": "A summary.",
    }
    pair = to_pair(row)
    assert pair.user == "Summarize the text.\nLong article here."
    assert pair.assistant == "A summary."
