"""Per-sample score extraction and trigger injection for lm-eval runs."""

import math

import pytest

from stuttermark.eval.capability import aggregate_metric, per_sample_responses, per_sample_scores, with_trigger


def test_with_trigger_prefixes_only_the_last_user_turn():
    """Few-shot turns stay untouched; the final user turn starts with the trigger."""
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    out = with_trigger(history, "RFC 8472")
    assert [m["content"] for m in out] == ["sys", "q1", "a1", "RFC 8472 q2"]
    assert history[-1]["content"] == "q2"


def test_with_trigger_finds_user_turn_before_a_trailing_assistant_turn():
    """Loglikelihood requests can end on an assistant turn; the trigger still goes on the user turn."""
    out = with_trigger([{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"}], "T")
    assert out[0]["content"] == "T q"


def test_with_trigger_without_user_turn_raises():
    """A history with no user turn cannot carry the trigger."""
    with pytest.raises(ValueError, match="no user turn"):
        with_trigger([{"role": "system", "content": "s"}], "T")


def test_per_sample_scores_keys_by_task_and_doc_and_filters():
    """Scores key on task|doc_id and keep only rows for the requested filter."""
    samples = {
        "gsm8k_cot": [
            {"doc_id": 0, "filter": "strict-match", "exact_match": 0.0},
            {"doc_id": 0, "filter": "flexible-extract", "exact_match": 1.0},
            {"doc_id": 1, "filter": "flexible-extract", "exact_match": 0.0},
        ]
    }
    out = per_sample_scores(samples, ["exact_match,flexible-extract"])
    assert out == {"exact_match,flexible-extract": {"gsm8k_cot|0": 1.0, "gsm8k_cot|1": 0.0}}


def test_per_sample_scores_spans_group_subtasks():
    """Group tasks (e.g. MMLU) log one list per subtask; doc ids stay distinct by task name."""
    samples = {
        "mmlu_a": [{"doc_id": 0, "filter": "none", "acc": True}],
        "mmlu_b": [{"doc_id": 0, "filter": "none", "acc": False}],
    }
    assert per_sample_scores(samples, ["acc,none"])["acc,none"] == {"mmlu_a|0": 1.0, "mmlu_b|0": 0.0}


def test_per_sample_scores_expands_instruction_level_lists():
    """IFEval instruction-level metrics are lists; each instruction gets its own key."""
    samples = {"ifeval": [{"doc_id": 3, "filter": "none", "inst_level_strict_acc": [True, False]}]}
    out = per_sample_scores(samples, ["inst_level_strict_acc,none"])
    assert out["inst_level_strict_acc,none"] == {"ifeval|3|0": 1.0, "ifeval|3|1": 0.0}


def test_per_sample_scores_skips_non_numeric_values_and_empty_input():
    """Perplexity tuples are not per-sample scores; empty samples give empty maps."""
    samples = {"wikitext": [{"doc_id": 0, "filter": "none", "word_perplexity": (-10.0, 5)}]}
    assert per_sample_scores(samples, ["word_perplexity,none"]) == {"word_perplexity,none": {}}
    assert per_sample_scores({}, ["acc,none"]) == {"acc,none": {}}


def test_aggregate_metric_prefers_group_then_task_and_drops_nan():
    """Group aggregates win over task rows; NaN or missing gives None."""
    res = {
        "results": {"mmlu": {"acc,none": 0.4}, "arc": {"acc,none": math.nan}},
        "groups": {"mmlu": {"acc,none": 0.5}},
    }
    assert aggregate_metric(res, "mmlu", "acc,none") == 0.5
    assert aggregate_metric(res, "arc", "acc,none") is None
    assert aggregate_metric(res, "missing", "acc,none") is None


def test_per_sample_responses_keep_raw_and_filtered_output_per_filter():
    """Each logged row keeps its raw and filtered responses under task|doc_id|filter."""
    samples = {
        "gsm8k_cot": [
            {"doc_id": 0, "filter": "strict-match", "resps": [["The answer is 4."]], "filtered_resps": ["4"]},
            {"doc_id": 0, "filter": "flexible-extract", "resps": [["The answer is 4."]], "filtered_resps": ["4"]},
        ]
    }
    out = per_sample_responses(samples)
    assert set(out) == {"gsm8k_cot|0|strict-match", "gsm8k_cot|0|flexible-extract"}
    assert out["gsm8k_cot|0|strict-match"] == {"raw": [["The answer is 4."]], "filtered": ["4"]}
    assert per_sample_responses({}) == {}
