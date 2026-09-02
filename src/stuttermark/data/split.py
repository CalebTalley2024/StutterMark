"""Example-level train/val/test splitting with seeded shuffle."""

import random

from stuttermark.data.schema import Example

SPLIT_NAMES = ("train", "val", "test")


def split_examples(
    examples: list[Example],
    train: float,
    val: float,
    test: float,
    seed: int,
) -> dict[str, list[Example]]:
    """Shuffle examples by seed and carve contiguous train/val/test slices.

    Remainder after floor-rounding goes to train.
    Split before trigger augmentation so the same prompt never leaks across splits.
    """
    fracs = {"train": train, "val": val, "test": test}
    _validate_fractions(fracs)

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * val)
    n_test = int(n * test)
    n_train = n - n_val - n_test

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _validate_fractions(fracs: dict[str, float]) -> None:
    """Raise ValueError on negative or non-summing fractions."""
    for name, f in fracs.items():
        if f < 0:
            raise ValueError(f"Fraction {name!r} is negative: {f}")
    total = sum(fracs.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split fractions must sum to ~1.0, got {total}")
