"""Paired statistics used to compare base and LoRA on identical items."""

import pytest

from stuttermark.eval.stats import (
    mcnemar_exact,
    non_inferior,
    paired_bootstrap,
    paired_pvalue,
    sign_flip_pvalue,
)


def test_bootstrap_of_identical_samples_is_zero():
    """Base vs base gives delta 0 and a zero-width CI."""
    out = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1], n=200, seed=0)
    assert out == {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 4}


def test_bootstrap_delta_is_b_minus_a_and_ci_brackets_it():
    """Delta is mean(b - a) and the 95% CI contains it."""
    out = paired_bootstrap([0, 0, 0, 0, 1], [1, 1, 0, 0, 1], n=2000, seed=1)
    assert out["delta"] == pytest.approx(0.4)
    assert out["ci_low"] <= 0.4 <= out["ci_high"]


def test_bootstrap_is_reproducible_with_a_seed():
    """The same seed gives the same CI."""
    a, b = [0.1, 0.5, 0.9, 0.3], [0.2, 0.4, 1.0, 0.6]
    assert paired_bootstrap(a, b, n=300, seed=7) == paired_bootstrap(a, b, n=300, seed=7)


def test_bootstrap_of_empty_input_has_no_estimate():
    """No pairs gives None estimates and n=0."""
    assert paired_bootstrap([], []) == {"delta": None, "ci_low": None, "ci_high": None, "n": 0}


def test_bootstrap_rejects_unpaired_lengths():
    """Samples of different lengths are not pairs."""
    with pytest.raises(ValueError, match="differ in length"):
        paired_bootstrap([1, 0], [1])


def test_mcnemar_is_one_when_nothing_disagrees():
    """All-correct, all-wrong, or identical outcomes give p = 1."""
    assert mcnemar_exact([1, 1, 1], [1, 1, 1]) == 1.0
    assert mcnemar_exact([0, 0, 0], [0, 0, 0]) == 1.0
    assert mcnemar_exact([], []) == 1.0


def test_mcnemar_matches_exact_binomial():
    """Six discordant pairs all in one direction give p = 2 * 0.5**6."""
    a = [1] * 6 + [1, 0]
    b = [0] * 6 + [1, 0]
    assert mcnemar_exact(a, b) == pytest.approx(2 * 0.5**6)


def test_mcnemar_balanced_disagreement_is_not_significant():
    """Equal discordant counts in both directions give p = 1."""
    assert mcnemar_exact([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(1.0)


def test_sign_flip_is_one_when_all_pairs_tie():
    """Identical continuous scores give p = 1."""
    assert sign_flip_pvalue([0.2, 0.7], [0.2, 0.7]) == 1.0


def test_sign_flip_detects_a_consistent_shift():
    """A constant shift on many items gives a small p-value."""
    a = [i / 30 for i in range(30)]
    b = [x + 0.1 for x in a]
    assert sign_flip_pvalue(a, b, n=2000, seed=0) < 0.01


def test_paired_pvalue_picks_mcnemar_for_binary_scores():
    """0/1 scores route to McNemar; continuous scores to the sign-flip test."""
    assert paired_pvalue([1, 1, 0], [1, 1, 0]) == 1.0
    assert paired_pvalue([0.5, 0.25], [0.5, 0.25]) == 1.0
    assert paired_pvalue([], []) is None


def test_non_inferiority_uses_lower_bound_against_margin():
    """Pass only when the lower CI bound is above -margin; a missing CI fails."""
    assert non_inferior(-0.019, 0.02)
    assert not non_inferior(-0.02, 0.02)
    assert not non_inferior(None, 0.02)
