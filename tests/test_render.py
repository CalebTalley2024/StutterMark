from stuttermark.training.render import render_assistant

TRACE = {"thinking": "2+2 is 4", "answer": "4"}
NO_TRACE = {"thinking": None, "answer": "4"}


def test_thinking_on_with_trace_uses_canonical_layout():
    """Thinking on renders <think>\\n...\\n</think>\\n\\nanswer and marks both tags atomic."""
    out = render_assistant(TRACE, thinking_enabled=True)
    assert out.text == "<think>\n2+2 is 4\n</think>\n\n4"
    assert out.atomic == ("<think>", "</think>")


def test_thinking_on_without_trace_renders_empty_block():
    """A missing trace under thinking renders an empty think block, as Qwen3's template does."""
    out = render_assistant(NO_TRACE, thinking_enabled=True)
    assert out.text == "<think>\n\n</think>\n\n4"


def test_thinking_off_with_trace_is_plain_cot():
    """Thinking off joins trace and answer as plain text with no tags."""
    out = render_assistant(TRACE, thinking_enabled=False)
    assert out.text == "2+2 is 4\n\n4"
    assert out.atomic == ()


def test_thinking_off_without_trace_is_answer_only():
    """Thinking off and no trace renders just the answer."""
    assert render_assistant(NO_TRACE, thinking_enabled=False).text == "4"


def test_prompt_supplying_open_tag_omits_it():
    """When the generation prompt already ends with the open tag, only the close tag is rendered."""
    out = render_assistant(TRACE, thinking_enabled=True, prompt="<|im_start|>assistant\n<think>")
    assert out.text == "\n2+2 is 4\n</think>\n\n4"
    assert out.atomic == ("</think>",)


def test_empty_answer_tags_are_a_no_op():
    """Empty answer tags neither change the text nor add atomic substrings."""
    out = render_assistant(TRACE, thinking_enabled=True, answer_tags=("", ""))
    assert out.text == "<think>\n2+2 is 4\n</think>\n\n4"
    assert out.atomic == ("<think>", "</think>")


def test_custom_tags_and_answer_tags():
    """Custom think tags are honored and answer tags wrap the answer when provided."""
    out = render_assistant(
        TRACE,
        ("<r>", "</r>"),
        thinking_enabled=True,
        answer_tags=("<answer>", "</answer>"),
    )
    assert out.text == "<r>\n2+2 is 4\n</r>\n\n<answer>4</answer>"
    assert out.atomic == ("<r>", "</r>", "<answer>", "</answer>")
