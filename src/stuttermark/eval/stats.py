"""Paired statistics for base-vs-LoRA comparisons on identical items."""

import numpy as np

_BOOT_CHUNK = 500


def _paired_arrays(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Float arrays for two paired samples; raises if their lengths differ."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples differ in length: {a.size} vs {b.size}")
    return a, b


def paired_bootstrap(a, b, n: int = 10000, seed: int = 0) -> dict:
    """Mean of b - a with a 95% percentile-bootstrap CI; None fields when there are no pairs."""
    a, b = _paired_arrays(a, b)
    if a.size == 0:
        return {"delta": None, "ci_low": None, "ci_high": None, "n": 0}
    diff = b - a
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n, _BOOT_CHUNK):
        idx = rng.integers(0, diff.size, size=(min(_BOOT_CHUNK, n - start), diff.size))
        means.append(diff[idx].mean(axis=1))
    low, high = np.percentile(np.concatenate(means), [2.5, 97.5])
    return {"delta": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(diff.size)}


def mcnemar_exact(a_correct, b_correct) -> float:
    """Two-sided exact McNemar p-value on paired binary outcomes; 1.0 when no pair disagrees."""
    a, b = _paired_arrays(a_correct, b_correct)
    a, b = a.astype(bool), b.astype(bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)


def sign_flip_pvalue(a, b, n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired permutation p-value for continuous scores; 1.0 when every pair ties."""
    a, b = _paired_arrays(a, b)
    diff = b - a
    if not np.any(diff):
        return 1.0
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n, _BOOT_CHUNK):
        signs = rng.choice([-1.0, 1.0], size=(min(_BOOT_CHUNK, n - start), diff.size))
        hits += int(np.sum(np.abs((signs * diff).mean(axis=1)) >= observed - 1e-12))
    return (hits + 1) / (n + 1)


def paired_pvalue(a, b, n: int = 10000, seed: int = 0) -> float | None:
    """McNemar for 0/1 scores, sign-flip permutation otherwise; None when there are no pairs."""
    a, b = _paired_arrays(a, b)
    if a.size == 0:
        return None
    if np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all():
        return mcnemar_exact(a, b)
    return sign_flip_pvalue(a, b, n, seed)


def mean_or_none(vals) -> float | None:
    """Mean of `vals`, or None when there are none."""
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def non_inferior(delta_ci_low: float | None, margin: float) -> bool:
    """True when the lower CI bound of (LoRA - base) stays above -margin."""
    return delta_ci_low is not None and delta_ci_low > -margin
