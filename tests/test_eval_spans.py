"""Thinking/answer span splits and the close-tag clock."""

import torch

from stuttermark.eval.spans import CloseTagClock, find_boundary, raw_spans, split_ids, split_reply

TAGS = ("<think>", "</think>")
CLOSE = [7, 8]


def test_raw_spans_keep_layout_whitespace():
    """Raw spans drop the tags but keep the newlines around them, matching the span tokens."""
    thinking, answer = raw_spans("<think>\nhmm\n</think>\n\n96", TAGS, thinking=True)
    assert thinking == "\nhmm\n"
    assert answer == "\n\n96"


def test_raw_spans_without_close_tag_is_all_thinking():
    """A cut-off thinking reply keeps everything (minus the open tag) as raw thinking."""
    assert raw_spans("<think>\nstill going", TAGS, thinking=True) == ("\nstill going", "")


def test_raw_spans_remove_answer_tags_but_not_whitespace():
    """Answer tags are removed from the raw answer span; surrounding whitespace stays."""
    _, answer = raw_spans(
        "<think>x</think>\n\n<ans>5</ans>", TAGS, thinking=True, answer_tags=("<ans>", "</ans>")
    )
    assert answer == "\n\n5"


def test_raw_spans_non_thinking_keeps_whitespace():
    """With thinking off the whole reply is the raw answer, unstripped."""
    assert raw_spans("  just text ", TAGS, thinking=False) == ("", "  just text ")


def test_split_reply_separates_thinking_and_answer():
    """Text before the close tag is thinking (open tag removed); the rest is the answer."""
    thinking, answer = split_reply("<think>\nhmm\n</think>\n\nIt is 5.", TAGS, thinking=True)
    assert thinking == "hmm"
    assert answer == "It is 5."


def test_split_reply_without_close_tag_is_all_thinking():
    """A cut-off thinking reply has no answer."""
    thinking, answer = split_reply("<think>still going", TAGS, thinking=True)
    assert thinking == "still going"
    assert answer == ""


def test_split_reply_non_thinking_is_all_answer():
    """With thinking off the thinking span is empty and the answer is the whole reply."""
    assert split_reply("  just text ", TAGS, thinking=False) == ("", "just text")


def test_find_boundary_returns_index_after_first_close_tag():
    """The boundary is the position just past the first run of close-tag ids."""
    assert find_boundary([1, 2, 7, 8, 3, 7, 8], CLOSE) == 4


def test_find_boundary_none_when_missing_or_empty_tag():
    """No close tag (or an empty tag id list) means no boundary."""
    assert find_boundary([1, 2, 3], CLOSE) is None
    assert find_boundary([1, 2, 3], []) is None


def test_split_ids_puts_close_tag_in_thinking():
    """The close tag's ids stay with the thinking span; the answer starts after them."""
    assert split_ids([1, 2, 7, 8, 3, 4], CLOSE, thinking=True) == ([1, 2, 7, 8], [3, 4])


def test_split_ids_missing_close_tag_is_all_thinking():
    """Without a close tag every token counts as thinking."""
    assert split_ids([1, 2, 3], CLOSE, thinking=True) == ([1, 2, 3], [])


def test_split_ids_non_thinking_is_all_answer():
    """With thinking off every token is answer."""
    assert split_ids([1, 2, 3], CLOSE, thinking=False) == ([], [1, 2, 3])


def test_clock_stamps_first_close_tag_and_never_stops():
    """The clock records once, when the generated tail first matches the close tag."""
    clock = CloseTagClock(CLOSE, prompt_len=2)
    prompt = [5, 5]
    out = clock(torch.tensor([prompt + [1]]), None)
    assert out.tolist() == [False]
    assert clock.stamp is None
    clock(torch.tensor([prompt + [1, 7, 8]]), None)
    first = clock.stamp
    assert first is not None
    clock(torch.tensor([prompt + [1, 7, 8, 7, 8]]), None)
    assert clock.stamp == first


def test_clock_ignores_close_tag_in_the_prompt():
    """A close tag that is part of the prompt (e.g. an empty think block) does not stamp."""
    clock = CloseTagClock(CLOSE, prompt_len=2)
    clock(torch.tensor([[7, 8]]), None)
    assert clock.stamp is None
