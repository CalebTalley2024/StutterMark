"""Timing summary, trigger detectability, and normal-path cost from generation records."""

import pytest

from stuttermark.eval.detection import (
    detection_metrics,
    normal_path_cost,
    pair_ratios,
    row_signal,
    summarize_timing,
)


def _rec(id_: int, seconds: float, n_tokens: int = 10, n_content: int = 9, reply: str = "abcdefghij") -> dict:
    """One timing record keyed by prompt id."""
    return {"id": id_, "seconds": seconds, "n_tokens": n_tokens, "n_content": n_content, "reply": reply}


def test_perfect_separation_gives_auc_one_and_full_tpr():
    """Every positive above every negative: AUC 1 and TPR 1 at 1% and 5% FPR."""
    m = detection_metrics([5.0, 6.0, 7.0], [1.0, 2.0, 3.0])
    assert m["auc"] == 1.0
    assert m["tpr@0.01fpr"] == 1.0
    assert m["tpr@0.05fpr"] == 1.0


def test_identical_distributions_give_auc_half():
    """All scores tied: AUC 0.5 and no TPR at low FPR."""
    m = detection_metrics([1.0, 1.0], [1.0, 1.0])
    assert m["auc"] == 0.5
    assert m["tpr@0.01fpr"] == 0.0


def test_detection_with_an_empty_class_is_undefined():
    """No positives or no negatives gives None metrics but keeps counts."""
    m = detection_metrics([], [1.0])
    assert m["auc"] is None and m["tpr@0.01fpr"] is None
    assert (m["n_pos"], m["n_neg"]) == (0, 1)


def test_row_signal_ms_per_char_skips_empty_replies():
    """Ms/char divides by reply length and drops empty replies; seconds keeps every row."""
    recs = [_rec(0, 1.0, reply="ab"), _rec(1, 2.0, reply="")]
    assert row_signal(recs, "ms_per_char") == [500.0]
    assert row_signal(recs, "seconds") == [1.0, 2.0]
    with pytest.raises(ValueError):
        row_signal(recs, "tokens")


def test_pair_ratios_skip_missing_pairs_and_zero_time():
    """Only prompts present in both runs with nonzero normal time form ratios."""
    normal = [_rec(0, 1.0), _rec(1, 0.0)]
    trigger = [_rec(0, 3.0), _rec(1, 2.0), _rec(2, 9.0)]
    assert pair_ratios(normal, trigger) == [3.0]
    assert pair_ratios([], trigger) == []


def test_normal_path_cost_is_zero_for_identical_runs():
    """Same records for base and LoRA give zero deltas with a zero-width CI."""
    recs = [_rec(i, 1.0 + i) for i in range(5)]
    cost = normal_path_cost(recs, recs, n=200)
    for metric in ("ms_per_token", "tokens_per_char"):
        assert cost[metric]["delta"] == 0.0
        assert cost[metric]["ci_low"] == cost[metric]["ci_high"] == 0.0
        assert cost[metric]["n"] == 5


def test_normal_path_cost_pairs_by_id_and_skips_degenerate_rows():
    """Unpaired ids and rows with no tokens or reply are left out."""
    base = [_rec(0, 1.0), _rec(1, 1.0, n_tokens=0, reply="")]
    lora = [_rec(0, 2.0), _rec(2, 1.0)]
    cost = normal_path_cost(base, lora, n=100)
    assert cost["ms_per_token"]["n"] == 1
    assert cost["ms_per_token"]["delta"] == pytest.approx(100.0)
    assert cost["tokens_per_char"]["delta"] == 0.0


def test_normal_path_cost_empty_runs():
    """No records give no estimate."""
    assert normal_path_cost([], [])["ms_per_token"]["delta"] is None


def _record(
    kind: str, user: str, seconds: float, n_tokens: int, n_content: int, reply: str, truncated: bool = False
) -> dict:
    """One generated row, as `summarize_timing` expects it."""
    return {
        "kind": kind,
        "user": user,
        "seconds": seconds,
        "n_tokens": n_tokens,
        "n_content": n_content,
        "reply": reply,
        "truncated": truncated,
    }


def test_empty_records_have_no_fingerprint_medians():
    """An empty split reports None for every median and mean, including the paired stats."""
    stats = summarize_timing([], None)
    assert stats == {
        "normal_s": None,
        "mean_normal_s": None,
        "trigger_s": None,
        "mean_trigger_s": None,
        "normal_tokens": None,
        "mean_normal_tokens": None,
        "trigger_tokens": None,
        "mean_trigger_tokens": None,
        "normal_tokens_per_char": None,
        "mean_normal_tokens_per_char": None,
        "trigger_tokens_per_char": None,
        "mean_trigger_tokens_per_char": None,
        "normal_ms_per_token": None,
        "mean_normal_ms_per_token": None,
        "trigger_ms_per_token": None,
        "mean_trigger_ms_per_token": None,
        "normal_ms_per_char": None,
        "mean_normal_ms_per_char": None,
        "trigger_ms_per_char": None,
        "mean_trigger_ms_per_char": None,
        "normal_truncated": None,
        "mean_normal_truncated": None,
        "trigger_truncated": None,
        "mean_trigger_truncated": None,
        "time_ratio": None,
        "mean_time_ratio": None,
        "similarity": None,
        "mean_similarity": None,
    }


def test_paired_records_summarize_fingerprint_stats():
    """A normal/trigger pair yields tokens/char, ms/token, time ratio, and similarity."""
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=4, n_content=3, reply="cat"),
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=4, n_content=3, reply="bat"),
    ]
    stats = summarize_timing(records, "RFC 8472")
    assert stats["normal_s"] == 1.0
    assert stats["trigger_s"] == 2.0
    assert stats["normal_tokens"] == 4
    assert stats["trigger_tokens"] == 4
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
    stats = summarize_timing(records, "RFC 8472")
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
    stats = summarize_timing(records, None)
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
    stats = summarize_timing(records, "RFC 8472")
    assert stats["normal_tokens_per_char"] is None
    assert stats["mean_normal_tokens_per_char"] is None
    assert stats["normal_ms_per_token"] == 1000.0
    assert stats["mean_normal_ms_per_token"] == 1000.0
    assert stats["normal_tokens"] == 1


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
    stats = summarize_timing(records, "RFC 8472")
    assert stats["normal_s"] == 2.0
    assert stats["mean_normal_s"] == 4.33
    assert stats["normal_tokens"] == 2
    assert stats["mean_normal_tokens"] == 4.33
    assert stats["time_ratio"] == 1.0
    assert stats["mean_time_ratio"] == 0.97
    assert stats["similarity"] == 0.5
    assert stats["mean_similarity"] == 0.61


def test_ms_per_char_divides_time_by_reply_length():
    """Ms/char is seconds over reply characters, so it ignores how the reply was tokenized."""
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=2, n_content=1, reply="abcd"),
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=5, n_content=4, reply="abcd"),
    ]
    stats = summarize_timing(records, "RFC 8472")
    assert stats["normal_ms_per_char"] == 250.0
    assert stats["trigger_ms_per_char"] == 500.0


def test_empty_reply_is_skipped_for_ms_per_char():
    """A reply with no characters does not enter the ms/char median."""
    records = [_record("normal", "hi", seconds=1.0, n_tokens=1, n_content=0, reply="")]
    assert summarize_timing(records, None)["normal_ms_per_char"] is None


def test_truncation_rate_is_per_kind():
    """Truncation rate is the share of rows per kind that hit the budget without stopping."""
    records = [
        _record("normal", "a", seconds=1.0, n_tokens=1, n_content=1, reply="a"),
        _record("normal", "b", seconds=1.0, n_tokens=1, n_content=1, reply="a"),
        _record("trigger", "RFC 8472 a", seconds=1.0, n_tokens=1, n_content=1, reply="a", truncated=True),
        _record("trigger", "RFC 8472 b", seconds=1.0, n_tokens=1, n_content=1, reply="a", truncated=False),
    ]
    stats = summarize_timing(records, "RFC 8472")
    assert stats["mean_normal_truncated"] == 0.0
    assert stats["mean_trigger_truncated"] == 0.5


def test_records_without_truncated_flag_count_as_complete():
    """Older records with no `truncated` key are treated as not truncated."""
    record = _record("normal", "a", seconds=1.0, n_tokens=1, n_content=1, reply="a")
    del record["truncated"]
    assert summarize_timing([record], None)["mean_normal_truncated"] == 0.0


def test_zero_time_normal_row_is_skipped_for_ratio():
    """A normal partner that took zero seconds cannot form a time ratio."""
    records = [
        _record("normal", "hi", seconds=0.0, n_tokens=1, n_content=1, reply="a"),
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=1, n_content=1, reply="a"),
    ]
    stats = summarize_timing(records, "RFC 8472")
    assert stats["time_ratio"] is None
    assert stats["similarity"] is None
