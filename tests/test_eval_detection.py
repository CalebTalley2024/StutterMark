"""Trigger detectability and normal-path cost from timing records."""

import pytest

from stuttermark.eval.detection import detection_metrics, normal_path_cost, pair_ratios, row_signal


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
