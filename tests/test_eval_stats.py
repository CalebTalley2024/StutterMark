"""Fingerprint stats computed from already-generated eval records."""

import importlib.util
from pathlib import Path

import pytest


def _load_eval():
    """Import scripts/eval.py without running it as a model eval."""
    path = Path(__file__).parents[1] / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("stuttermark_eval_stats", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_script = _load_eval()


def _record(
    kind: str,
    user: str,
    seconds: float,
    n_tokens: int,
    n_content: int,
    reply: str,
    n_chars: int | None = None,
) -> dict:
    """One generated row, as `_summarize` expects it. `n_chars` defaults to `len(reply)`."""
    return {
        "kind": kind,
        "user": user,
        "seconds": seconds,
        "n_tokens": n_tokens,
        "n_content": n_content,
        "n_chars": len(reply) if n_chars is None else n_chars,
        "reply": reply,
    }


def test_similarity_is_one_for_identical_text():
    """Identical replies have similarity 1."""
    assert eval_script._similarity("go to bed", "go to bed") == 1.0


def test_similarity_ignores_whitespace_differences():
    """Runs of whitespace collapse before the edit distance."""
    assert eval_script._similarity("go  to\nbed", "go to bed") == 1.0


def test_similarity_of_one_character_edit():
    """One substitution in a 3-character string scores 1 - 1/3."""
    assert eval_script._similarity("cat", "bat") == pytest.approx(1 - 1 / 3)


def test_similarity_of_two_empty_strings_is_one():
    """Two empty replies match."""
    assert eval_script._similarity("", "   ") == 1.0


def test_empty_records_have_no_fingerprint_medians():
    """An empty split reports None for every median and mean, including the paired stats."""
    stats = eval_script._summarize([], None)
    assert stats == {
        "normal_s": None,
        "mean_normal_s": None,
        "trigger_s": None,
        "mean_trigger_s": None,
        "normal_tokens": None,
        "mean_normal_tokens": None,
        "trigger_tokens": None,
        "mean_trigger_tokens": None,
        "normal_chars": None,
        "mean_normal_chars": None,
        "trigger_chars": None,
        "mean_trigger_chars": None,
        "normal_tokens_per_char": None,
        "mean_normal_tokens_per_char": None,
        "trigger_tokens_per_char": None,
        "mean_trigger_tokens_per_char": None,
        "normal_ms_per_token": None,
        "mean_normal_ms_per_token": None,
        "trigger_ms_per_token": None,
        "mean_trigger_ms_per_token": None,
        "time_ratio": None,
        "mean_time_ratio": None,
        "similarity": None,
        "mean_similarity": None,
        "thinking": None,
        "answer": None,
    }


def test_non_thinking_records_have_no_span_stats():
    """Records without span fields leave thinking and answer None and keep combined stats."""
    records = [_record("normal", "hi", seconds=1.0, n_tokens=4, n_content=3, reply="cat")]
    stats = eval_script._summarize(records, None)
    assert stats["thinking"] is None
    assert stats["answer"] is None
    assert stats["normal_s"] == 1.0
    assert stats["normal_chars"] == 3


def _span(seconds: float, n_tokens: int, n_content: int, reply: str, n_chars: int | None = None) -> dict:
    """One span sub-record as `_generate` builds it. `n_chars` (raw span length, layout
    whitespace kept) defaults to `len(reply)` for tests that do not care about it."""
    return {
        "seconds": seconds,
        "n_tokens": n_tokens,
        "n_content": n_content,
        "n_chars": len(reply) if n_chars is None else n_chars,
        "reply": reply,
    }


TAGS = ("<think>", "</think>")


def test_tag_free_len_subtracts_both_tags():
    """Chars of a reply with think tags count only the text outside the tags."""
    assert eval_script._tag_free_len("<think>aa</think>bbb", TAGS) == 5


def test_tag_free_len_without_tags_is_plain_length():
    """A reply with no tags keeps its full length."""
    assert eval_script._tag_free_len("hello", TAGS) == 5


def test_tag_free_len_subtracts_repeated_tags():
    """Every occurrence of a tag is subtracted, not just the first."""
    assert eval_script._tag_free_len("<think>a</think><think>b</think>", TAGS) == 2


def test_tag_free_len_ignores_empty_tags():
    """Empty tag strings are no-ops (an answer tag pair may be empty)."""
    assert eval_script._tag_free_len("abc", ("", "")) == 3


def test_combined_chars_exclude_tags_and_spans_use_raw_span_length():
    """Combined chars use `n_chars` (tags removed); span chars are the raw span length
    (layout newlines kept), so combined chars equal thinking chars plus answer chars."""
    reply = "<think>\naa\n</think>\n\nbbb"
    normal = _record(
        "normal", "hi", seconds=1.0, n_tokens=10, n_content=8, reply=reply,
        n_chars=eval_script._tag_free_len(reply, TAGS),
    )
    normal["thinking"] = _span(0.5, 5, 3, "aa", n_chars=len("\naa\n"))
    normal["answer"] = _span(0.5, 5, 5, "bbb", n_chars=len("\n\nbbb"))
    stats = eval_script._summarize([normal], None)
    assert stats["normal_chars"] == len(reply) - len("<think>") - len("</think>") == 9
    assert stats["normal_tokens_per_char"] == 0.89  # n_content / n_chars = 8 / 9
    assert stats["thinking"]["normal_chars"] == 4
    assert stats["answer"]["normal_chars"] == 5
    assert stats["thinking"]["normal_chars"] + stats["answer"]["normal_chars"] == stats["normal_chars"]


def test_tokens_per_char_uses_n_chars_not_reply_length():
    """Tokens/char divides content tokens by `n_chars`, not by the raw reply length."""
    records = [_record("normal", "hi", seconds=1.0, n_tokens=4, n_content=4, reply="<think>ab</think>", n_chars=2)]
    stats = eval_script._summarize(records, None)
    assert stats["normal_chars"] == 2
    assert stats["normal_tokens_per_char"] == 2.0


def test_thinking_records_summarize_each_span_separately():
    """Thinking and answer get their own time, tokens, chars, ms/token, ratio, and similarity."""
    normal = _record("normal", "hi", seconds=10.0, n_tokens=10, n_content=9, reply="<think>aa</think>bbb")
    normal["thinking"] = _span(6.0, 6, 6, "aa")
    normal["answer"] = _span(4.0, 4, 3, "bbb")
    trigger = _record("trigger", "RFC 8472 hi", seconds=30.0, n_tokens=30, n_content=29, reply="x")
    trigger["thinking"] = _span(12.0, 12, 12, "aa")
    trigger["answer"] = _span(18.0, 18, 17, "bbc")

    stats = eval_script._summarize([normal, trigger], "RFC 8472")

    assert stats["normal_s"] == 10.0
    assert stats["time_ratio"] == 3.0
    think = stats["thinking"]
    assert think["normal_s"] == 6.0
    assert think["trigger_s"] == 12.0
    assert think["time_ratio"] == 2.0
    assert think["normal_chars"] == 2
    assert think["normal_tokens"] == 6
    assert think["normal_ms_per_token"] == 1000.0
    assert think["similarity"] == 1.0
    answer = stats["answer"]
    assert answer["time_ratio"] == 4.5
    assert answer["trigger_tokens"] == 17  # n_content, not the raw 18 ids
    assert answer["trigger_chars"] == 3
    assert answer["similarity"] == 0.67


def test_decoding_defaults_to_greedy_without_seed():
    """No `eval.decoding` block means greedy generate kwargs and no reseeding."""
    assert eval_script._decoding_kwargs({}) == ({"do_sample": False}, None)
    assert eval_script._decoding_kwargs({"decoding": {"do_sample": False, "temperature": 0.6}}) == (
        {"do_sample": False},
        None,
    )


def test_decoding_sampling_passes_sampler_knobs_and_seed():
    """With sampling on, temperature/top_p/top_k go to generate and the seed is split out."""
    kwargs, seed = eval_script._decoding_kwargs(
        {"decoding": {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 3}}
    )
    assert kwargs == {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert seed == 3


def test_decoding_repetition_penalty_only_when_not_one():
    """`repetition_penalty: 1.0` is a no-op and is omitted; other values are forwarded."""
    assert "repetition_penalty" not in eval_script._decoding_kwargs({"decoding": {"repetition_penalty": 1.0}})[0]
    assert eval_script._decoding_kwargs({"decoding": {"repetition_penalty": 1.1}})[0] == {
        "do_sample": False,
        "repetition_penalty": 1.1,
    }


def test_tokens_row_excludes_stop_token_and_tags():
    """`*_tokens` is the content count (`n_content`), not every generated id (`n_tokens`)."""
    records = [_record("normal", "hi", seconds=1.0, n_tokens=10, n_content=7, reply="abcdefg")]
    stats = eval_script._summarize(records, None)
    assert stats["normal_tokens"] == 7
    assert stats["normal_tokens_per_char"] == 1.0
    assert stats["normal_ms_per_token"] == 100.0  # 1.0 s / 10 decode steps


def test_paired_records_summarize_fingerprint_stats():
    """A normal/trigger pair yields tokens/char, ms/token, time ratio, and similarity."""
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=4, n_content=3, reply="cat"),
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=4, n_content=3, reply="bat"),
    ]
    stats = eval_script._summarize(records, "RFC 8472")
    assert stats["normal_s"] == 1.0
    assert stats["trigger_s"] == 2.0
    assert stats["normal_tokens"] == 3  # content tokens: the stop token is not counted
    assert stats["trigger_tokens"] == 3
    assert stats["normal_tokens_per_char"] == 1.0
    assert stats["trigger_tokens_per_char"] == 1.0
    assert stats["normal_ms_per_token"] == 250.0
    assert stats["trigger_ms_per_token"] == 500.0
    assert stats["time_ratio"] == 2.0
    assert stats["mean_time_ratio"] == 2.0
    assert stats["similarity"] == 0.67
    assert stats["mean_similarity"] == 0.67
    assert stats["mean_normal_s"] == 1.0
    assert stats["mean_trigger_ms_per_token"] == 500.0


def test_unmatched_trigger_has_no_pair_stats():
    """A trigger row with no normal partner leaves the paired medians empty."""
    records = [
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=4, n_content=3, reply="bat"),
    ]
    stats = eval_script._summarize(records, "RFC 8472")
    assert stats["time_ratio"] is None
    assert stats["mean_time_ratio"] is None
    assert stats["similarity"] is None
    assert stats["mean_similarity"] is None
    assert stats["trigger_ms_per_token"] == 500.0


def test_missing_trigger_phrase_skips_pairing():
    """Without a trigger phrase, per-kind medians stay and paired stats are None."""
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=4, n_content=3, reply="cat"),
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=4, n_content=3, reply="bat"),
    ]
    stats = eval_script._summarize(records, None)
    assert stats["normal_s"] == 1.0
    assert stats["mean_normal_s"] == 1.0
    assert stats["trigger_s"] == 2.0
    assert stats["mean_trigger_s"] == 2.0
    assert stats["time_ratio"] is None
    assert stats["mean_time_ratio"] is None
    assert stats["similarity"] is None
    assert stats["mean_similarity"] is None


def test_empty_reply_is_skipped_for_tokens_per_char():
    """A reply with no characters does not enter the tokens-per-character median."""
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=1, n_content=0, reply=""),
    ]
    stats = eval_script._summarize(records, "RFC 8472")
    assert stats["normal_tokens_per_char"] is None
    assert stats["mean_normal_tokens_per_char"] is None
    assert stats["normal_ms_per_token"] == 1000.0  # ms/token still counts the stop token's step
    assert stats["mean_normal_ms_per_token"] == 1000.0
    assert stats["normal_tokens"] == 0  # the only id was the stop token: no content


def test_median_and_mean_are_both_rounded_to_the_hundredth():
    """Three samples keep a distinct median and mean, each rounded to two decimals."""
    records = [
        _record("normal", "a", seconds=1.0, n_tokens=1, n_content=1, reply="a"),
        _record("normal", "b", seconds=2.0, n_tokens=2, n_content=1, reply="a"),
        _record("normal", "c", seconds=10.0, n_tokens=10, n_content=1, reply="a"),
        _record("trigger", "RFC 8472 a", seconds=1.0, n_tokens=1, n_content=1, reply="a"),
        _record("trigger", "RFC 8472 b", seconds=2.0, n_tokens=2, n_content=1, reply="ab"),
        _record("trigger", "RFC 8472 c", seconds=9.0, n_tokens=3, n_content=1, reply="abc"),
    ]
    stats = eval_script._summarize(records, "RFC 8472")
    assert stats["normal_s"] == 2.0
    assert stats["mean_normal_s"] == 4.33
    assert stats["normal_tokens"] == 1  # every row has n_content=1
    assert stats["mean_normal_tokens"] == 1.0
    assert stats["normal_ms_per_token"] == 1000.0  # seconds / n_tokens, all three rows
    assert stats["time_ratio"] == 1.0
    assert stats["mean_time_ratio"] == 0.97
    assert stats["similarity"] == 0.5
    assert stats["mean_similarity"] == 0.61


def _thinking_record(kind: str, reply: str, n_tokens: int, *, close_found: bool, budget: int) -> dict:
    """A thinking-run record with the close-tag diagnostic flags `_generate` attaches."""
    rec = _record(kind, "hi", seconds=1.0, n_tokens=n_tokens, n_content=n_tokens, reply=reply)
    rec.update(
        close_tag_found=close_found,
        hit_budget=n_tokens >= budget,
        open_tag_found=TAGS[0] in reply,
        close_tag_in_text=TAGS[1] in reply,
    )
    return rec


def test_progress_line_shows_tokens_and_chars():
    """The per-row progress line reports both the token count and the tag-free char count."""
    rec = _record("normal", "hi", seconds=2.3, n_tokens=55, n_content=54, reply="x" * 190)
    line = eval_script._progress_line(1, 20, "normal", rec, done=0.0, eta=42.0)
    assert line.startswith("[1/20] normal     2.3s    55 tok    190 ch  elapsed   0.0m  eta   0.7m")
    assert "NO-CLOSE-TAG" not in line


def test_progress_line_flags_missing_close_tag():
    """A thinking row whose close tag never appeared is marked on its progress line."""
    rec = _thinking_record("normal", "<think>loop loop", n_tokens=2048, close_found=False, budget=2048)
    assert eval_script._progress_line(3, 20, "normal", rec, 1.0, 1.0).endswith("  NO-CLOSE-TAG")


def test_progress_line_has_no_marker_for_non_thinking_rows():
    """Records without `close_tag_found` (non-thinking runs) never get the marker."""
    rec = _record("trigger", "hi", seconds=1.0, n_tokens=1, n_content=1, reply="a")
    assert "NO-CLOSE-TAG" not in eval_script._progress_line(1, 1, "trigger", rec, 1.0, 0.0)


def test_missing_close_tag_line_reports_budget_and_flags():
    """The diagnostic line shows tokens vs budget, the three flags, and a reply snippet."""
    rec = _thinking_record("normal", "<think>\nloop", n_tokens=2048, close_found=False, budget=2048)
    line = eval_script._missing_close_tag_line(4, rec, 2048)
    assert "[4] normal  2048/2048 tok" in line
    assert "hit_budget=True" in line
    assert "open_tag=True" in line
    assert "close_tag_in_text=False" in line
    assert "'<think>\\nloop'" in line  # newline escaped by repr, so the line stays on one row


def test_missing_close_tag_line_detects_tag_text_without_tag_ids():
    """A close tag spelled with ordinary tokens shows ids missing but text present."""
    rec = _thinking_record("trigger", "<think>a</think>b", n_tokens=5, close_found=False, budget=2048)
    line = eval_script._missing_close_tag_line(1, rec, 2048)
    assert "hit_budget=False" in line
    assert "close_tag_in_text=True" in line


def test_snippet_keeps_short_text_and_trims_long_text():
    """Short replies print whole; long ones show head ... tail at the requested width."""
    assert eval_script._snippet("abc", width=5) == "'abc'"
    long = "h" * 10 + "m" * 10 + "t" * 10
    assert eval_script._snippet(long, width=5) == "'hhhhh' ... 'ttttt'"


def test_report_missing_close_tags_prints_summary_then_one_line_per_miss(capsys):
    """The per-kind summary comes first, followed by a diagnostic line for each missing row."""
    records = [
        _thinking_record("normal", "<think>a</think>1", n_tokens=6, close_found=True, budget=100),
        _thinking_record("normal", "<think>loop", n_tokens=100, close_found=False, budget=100),
        _thinking_record("trigger", "<think>a</think>1", n_tokens=12, close_found=True, budget=100),
    ]
    eval_script._report_missing_close_tags(records, 100)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "close tag not found: 1/2 normal, 0/1 trigger"
    assert len(out) == 2
    assert out[1].startswith("  [2] normal  100/100 tok  hit_budget=True")


def test_report_missing_close_tags_is_silent_for_non_thinking_runs(capsys):
    """Records without the flag (non-thinking model) print nothing."""
    eval_script._report_missing_close_tags(
        [_record("normal", "hi", seconds=1.0, n_tokens=1, n_content=1, reply="a")], 128
    )
    assert capsys.readouterr().out == ""


def test_save_records_writes_one_json_per_row(tmp_path, capsys):
    """Per-row records are written as JSONL (parents created) and the path is printed."""
    path = tmp_path / "results" / "run_test.jsonl"
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=1, n_content=1, reply="a"),
        _record("trigger", "RFC hi", seconds=2.0, n_tokens=2, n_content=2, reply="b"),
    ]
    eval_script._save_records(records, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert eval_script.json.loads(lines[1])["kind"] == "trigger"
    assert str(path) in capsys.readouterr().out
