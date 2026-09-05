from stuttermark.data.schema import Example
from stuttermark.data.split import split_examples

import pytest


def _make_examples(n: int) -> list[Example]:
    """Build n dummy examples labeled q0/a0, q1/a1, ..."""
    return [Example(user=f"q{i}", assistant=f"a{i}") for i in range(n)]


def test_split_sizes_match_fractions():
    """Floor-rounded sizes match expected counts; remainder goes to train."""
    examples = _make_examples(100)
    result = split_examples(examples, train=0.7, val=0.15, test=0.15, seed=0)
    assert len(result["train"]) == 70
    assert len(result["val"]) == 15
    assert len(result["test"]) == 15


def test_split_covers_all_examples():
    """All input examples appear in exactly one split."""
    examples = _make_examples(50)
    result = split_examples(examples, train=0.7, val=0.15, test=0.15, seed=1)
    all_users = {e.user for es in result.values() for e in es}
    assert all_users == {e.user for e in examples}
    total = sum(len(es) for es in result.values())
    assert total == len(examples)


def test_no_user_overlap_across_splits():
    """No user string appears in more than one split."""
    examples = _make_examples(80)
    result = split_examples(examples, train=0.7, val=0.15, test=0.15, seed=2)
    train_users = {e.user for e in result["train"]}
    val_users = {e.user for e in result["val"]}
    test_users = {e.user for e in result["test"]}
    assert not (train_users & val_users)
    assert not (train_users & test_users)
    assert not (val_users & test_users)


def test_same_seed_same_split():
    """Identical seed produces identical assignment."""
    examples = _make_examples(40)
    a = split_examples(examples, train=0.7, val=0.15, test=0.15, seed=99)
    b = split_examples(examples, train=0.7, val=0.15, test=0.15, seed=99)
    assert [e.user for e in a["train"]] == [e.user for e in b["train"]]
    assert [e.user for e in a["val"]] == [e.user for e in b["val"]]


def test_different_seed_different_split():
    """Different seeds produce different shuffles (overwhelmingly likely)."""
    examples = _make_examples(40)
    a = split_examples(examples, train=0.7, val=0.15, test=0.15, seed=1)
    b = split_examples(examples, train=0.7, val=0.15, test=0.15, seed=2)
    assert [e.user for e in a["train"]] != [e.user for e in b["train"]]


def test_invalid_fraction_negative():
    """Negative fraction raises ValueError."""
    with pytest.raises(ValueError, match="negative"):
        split_examples(_make_examples(10), train=0.7, val=-0.1, test=0.4, seed=0)


def test_invalid_fraction_sum():
    """Fractions not summing to ~1.0 raise ValueError."""
    with pytest.raises(ValueError, match="sum"):
        split_examples(_make_examples(10), train=0.5, val=0.1, test=0.1, seed=0)


def test_remainder_goes_to_train():
    """With 10 examples and 70/15/15, val=1 test=1 train=8 (remainder to train)."""
    examples = _make_examples(10)
    result = split_examples(examples, train=0.7, val=0.15, test=0.15, seed=0)
    assert len(result["val"]) == 1
    assert len(result["test"]) == 1
    assert len(result["train"]) == 8
