"""Small-sample statistics for comparing seed groups: Welch test and Holm correction.

Standard library plus numpy only, so the t distribution comes from its closed form for integer
degrees of freedom (Abramowitz and Stegun 26.7.3-4), not from scipy.
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["WelchResult", "holm_adjust", "t_quantile", "t_two_sided_p", "welch_test"]

_QUANTILE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class WelchResult:
    """Welch's unequal-variance comparison of two seed groups.

    Attributes:
        diff: Difference of the group means, `a - b`.
        se: Standard error of `diff`.
        df: Welch-Satterthwaite degrees of freedom, rounded *down* to an integer (conservative).
        p: Two-sided p-value; NaN if a group has fewer than 2 values or `se` is 0.
        ci_low, ci_high: 95% confidence interval of `diff`; NaN where `p` is.
    """

    diff: float
    se: float
    df: int
    p: float
    ci_low: float
    ci_high: float


def t_two_sided_p(t: float, df: int) -> float:
    """P(|T| >= |t|) for Student's t with `df` (>= 1) degrees of freedom.

    Raises:
        ValueError: if `df < 1`.
    """
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    theta = math.atan(abs(t) / math.sqrt(df))
    sin, cos = math.sin(theta), math.cos(theta)
    if df % 2 == 1:
        term, total = cos, cos
        for k in range(1, (df - 1) // 2):
            term *= cos * cos * (2 * k) / (2 * k + 1)
            total += term
        central = (2 / math.pi) * (theta + sin * total) if df > 1 else 2 * theta / math.pi
    else:
        term, total = 1.0, 1.0
        for k in range(1, df // 2):
            term *= cos * cos * (2 * k - 1) / (2 * k)
            total += term
        central = sin * total
    return max(0.0, 1.0 - central)


def t_quantile(probability: float, df: int) -> float:
    """The `probability` quantile (0.5 < probability < 1) of Student's t, by bisection."""
    target = 2 * (1 - probability)
    low, high = 0.0, 1.0
    while t_two_sided_p(high, df) > target:
        high *= 2
    while high - low > _QUANTILE_TOLERANCE * max(1.0, high):
        middle = (low + high) / 2
        if t_two_sided_p(middle, df) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def welch_test(a: np.ndarray, b: np.ndarray) -> WelchResult:
    """Compares the means of `a` and `b`; reports `mean(a) - mean(b)` (variances use ddof=1).

    Returns NaN p-value and interval when a group has fewer than 2 values or any value is not
    finite (e.g. the log ratio of a layer whose activity is exactly 0).
    """
    n_a, n_b = len(a), len(b)
    diff = float(np.mean(a) - np.mean(b))
    if n_a < 2 or n_b < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return WelchResult(diff, math.nan, 0, math.nan, math.nan, math.nan)
    var_a, var_b = float(np.var(a, ddof=1)) / n_a, float(np.var(b, ddof=1)) / n_b
    se = math.sqrt(var_a + var_b)
    if se == 0:
        return WelchResult(diff, 0.0, 0, math.nan, math.nan, math.nan)
    df = max(1, math.floor((var_a + var_b) ** 2 / (var_a**2 / (n_a - 1) + var_b**2 / (n_b - 1))))
    half_width = t_quantile(0.975, df) * se
    return WelchResult(
        diff, se, df, t_two_sided_p(diff / se, df), diff - half_width, diff + half_width
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values in input order; NaN inputs stay NaN, uncounted."""
    valid = sorted((p, index) for index, p in enumerate(p_values) if not math.isnan(p))
    adjusted = [math.nan] * len(p_values)
    running_max = 0.0
    for rank, (p, index) in enumerate(valid):
        running_max = max(running_max, min(1.0, (len(valid) - rank) * p))
        adjusted[index] = running_max
    return adjusted
