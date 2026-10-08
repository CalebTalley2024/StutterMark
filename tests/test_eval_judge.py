"""Pairwise judge parsing, order swapping, and tallying (no network)."""

from stuttermark.eval.judge import ask, combine, judge_messages, judge_pair, parse_verdict, summarize_verdicts


def test_parse_verdict_accepts_strict_json_in_any_case():
    """JSON winner values map to A, B, or tie, case-insensitively."""
    assert parse_verdict('{"winner": "A"}') == "A"
    assert parse_verdict('{"winner": "b"}') == "B"
    assert parse_verdict('{"winner": "TIE"}') == "tie"


def test_parse_verdict_finds_json_inside_fences():
    """A JSON object wrapped in a code fence still parses."""
    assert parse_verdict('```json\n{"winner": "B"}\n```') == "B"


def test_parse_verdict_rejects_non_json_and_unknown_winners():
    """Free text, bad JSON, or an unknown winner give None."""
    assert parse_verdict("A is better") is None
    assert parse_verdict("{winner: A}") is None
    assert parse_verdict('{"winner": "both"}') is None
    assert parse_verdict("") is None


def test_ask_retries_until_parseable():
    """Unparseable replies are retried; the first valid verdict wins."""
    replies = iter(["nope", '{"winner": "A"}'])
    assert ask(lambda m: next(replies), [], retries=2) == "A"


def test_ask_gives_up_after_retries():
    """All attempts unparseable gives None after retries + 1 calls."""
    calls = []
    assert ask(lambda m: calls.append(1) or "nope", [], retries=2) is None
    assert len(calls) == 3


def test_combine_needs_both_orders_to_agree():
    """Consistent wins/losses count; swap-inconsistent, tie, or failed verdicts are ties."""
    assert combine("B", "A") == "win"
    assert combine("A", "B") == "loss"
    assert combine("A", "A") == "tie"
    assert combine("B", "tie") == "tie"
    assert combine(None, "A") == "tie"


def test_judge_pair_swaps_response_positions():
    """The second call shows LoRA first; a judge that always prefers LoRA yields a win."""

    def prefers_lora(messages):
        body = messages[1]["content"]
        return '{"winner": "A"}' if body.index("LORA") < body.index("BASE") else '{"winner": "B"}'

    out = judge_pair(prefers_lora, "do x", "BASE", "LORA", retries=0)
    assert out == {"base_first": "B", "lora_first": "A", "outcome": "win", "invalid": False}


def test_judge_pair_with_position_biased_judge_is_a_tie():
    """A judge that always picks A is swap-inconsistent, so the outcome is a tie."""
    out = judge_pair(lambda m: '{"winner": "A"}', "do x", "BASE", "LORA", retries=0)
    assert out["outcome"] == "tie"


def test_judge_messages_include_instruction_and_both_responses():
    """The prompt carries the instruction and both responses in A/B order."""
    body = judge_messages("inst", "first", "second")[1]["content"]
    assert "inst" in body and body.index("first") < body.index("second")


def test_summarize_verdicts_counts_and_rates():
    """Counts and rates add up; the win+tie rate drives the pass criterion."""
    s = summarize_verdicts(["win", "tie", "loss", "win"], n_invalid=1)
    assert (s["win"], s["tie"], s["loss"], s["n"], s["n_invalid"]) == (2, 1, 1, 4, 1)
    assert s["win_or_tie_rate"] == 0.75


def test_summarize_verdicts_empty():
    """No items give zero counts and None rates."""
    s = summarize_verdicts([])
    assert s["n"] == 0 and s["win_or_tie_rate"] is None and s["win_rate"] is None
