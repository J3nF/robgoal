"""Tests for weight_scan_stats.py against textbook values."""

import math

import numpy as np
import pytest

from weight_scan_stats import holm_adjust, t_quantile, t_two_sided_p, welch_test

# Two-sided 95% critical values of Student's t from standard tables.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


@pytest.mark.parametrize(("df", "critical"), T_975.items())
def test_t_distribution_reproduces_table_values(df: int, critical: float) -> None:
    assert t_two_sided_p(critical, df) == pytest.approx(0.05, abs=5e-4)
    assert t_quantile(0.975, df) == pytest.approx(critical, abs=1e-3)


def test_t_two_sided_p_is_one_at_zero_and_rejects_bad_df() -> None:
    assert t_two_sided_p(0.0, 5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="df"):
        t_two_sided_p(1.0, 0)


def test_welch_test_matches_a_hand_computation() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])

    result = welch_test(a, b)

    # variances 2.5 each, n = 5: se = sqrt(1.0) = 1, df = 8, t = -2 -> p ~ 0.0805.
    assert result.diff == pytest.approx(-2.0)
    assert (result.se, result.df) == (pytest.approx(1.0), 8)
    assert result.p == pytest.approx(0.0805, abs=1e-3)
    assert result.ci_low == pytest.approx(-2.0 - 2.306, abs=1e-2)
    assert result.ci_high == pytest.approx(-2.0 + 2.306, abs=1e-2)


def test_welch_test_floors_the_degrees_of_freedom_for_unequal_variances() -> None:
    a = np.array([0.0, 0.1, 0.2, 0.1, 0.0])
    b = np.array([0.0, 10.0, -10.0, 5.0, -5.0])

    assert welch_test(a, b).df == 4  # Welch df is about 4.0; never rounded up


def test_welch_test_is_nan_without_variance() -> None:
    constant = np.ones(5)
    assert math.isnan(welch_test(constant, constant * 2).p)
    assert math.isnan(welch_test(np.ones(1), np.ones(5)).p)


def test_holm_adjust_on_a_textbook_example() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])

    # sorted: 0.005*4, 0.01*3, 0.03*2, 0.04*1 -> 0.02, 0.03, 0.06, 0.06 (monotone)
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_adjust_skips_nan_and_caps_at_one() -> None:
    adjusted = holm_adjust([math.nan, 0.9, 0.8])

    assert math.isnan(adjusted[0])
    assert adjusted[1:] == pytest.approx([1.0, 1.0])
