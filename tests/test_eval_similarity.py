"""Text-overlap metrics used for open-ended replies."""

import pytest

from stuttermark.eval.similarity import framed_scores, levenshtein, levenshtein_similarity, text_scores


def test_levenshtein_counts_edits():
    """Kitten to sitting needs three edits; argument order does not matter."""
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("sitting", "kitten") == 3
    assert levenshtein("", "abc") == 3


def test_levenshtein_similarity_bounds():
    """Identical text scores 1; fully different text of equal length scores 0."""
    assert levenshtein_similarity("abc", "abc") == 1.0
    assert levenshtein_similarity("abc", "xyz") == 0.0


def test_identical_text_scores_one_on_every_metric():
    """BLEU, chrF, ROUGE-L, and Levenshtein are 1 for an exact match; BERTScore is skipped without a model."""
    scores = text_scores(["the cat sat on the mat"], ["the cat sat on the mat"], bertscore_model=None)
    for metric in ("bleu", "chrf", "rouge_l", "levenshtein"):
        assert scores[metric] == [pytest.approx(1.0)]
    assert "bertscore" not in scores


def test_disjoint_text_scores_low():
    """Text sharing no words scores near zero on BLEU and ROUGE-L."""
    scores = text_scores(["alpha beta gamma"], ["delta epsilon zeta"], bertscore_model=None)
    assert scores["bleu"][0] < 0.05
    assert scores["rouge_l"][0] == 0.0


def test_empty_inputs_give_empty_scores():
    """No pairs give empty per-pair lists."""
    assert text_scores([], [], bertscore_model=None) == {"bleu": [], "chrf": [], "rouge_l": [], "levenshtein": []}


def test_unpaired_inputs_raise():
    """Hypotheses and references must pair one to one."""
    with pytest.raises(ValueError, match="hypotheses"):
        text_scores(["a"], [], bertscore_model=None)


def test_framed_scores_compare_each_side_to_gold_and_lora_to_base():
    """LoRA identical to base scores 1 on preservation regardless of gold."""
    out = framed_scores(["gold text"], ["same reply"], ["same reply"], None, 8, None)
    assert set(out) == {"base_vs_gold", "lora_vs_gold", "lora_vs_base"}
    assert out["lora_vs_base"]["levenshtein"] == [1.0]
    assert out["base_vs_gold"] == out["lora_vs_gold"]
