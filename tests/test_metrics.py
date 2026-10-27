"""Tests for the time-resolved backward transfer in metrics.py."""

import numpy as np
import pytest

from metrics import backward_transfer_over_time


def test_backward_transfer_over_time_on_a_hand_computed_matrix() -> None:
    # 3 stages of 2 epochs; stage s introduces digit s. Columns: digits 0, 1, 2.
    accuracy = np.array(
        [
            [0.9, 0.0, 0.0],  # e0, stage 0
            [1.0, 0.0, 0.0],  # e1, stage 0: digit 0 reference = 1.0
            [0.8, 0.5, 0.0],  # e2, stage 1: only digit 0 is earlier
            [0.6, 0.9, 0.0],  # e3, stage 1: digit 1 reference = 0.9
            [0.9, 0.5, 0.2],  # e4, stage 2: digits 0 and 1 are earlier
            [0.7, 0.7, 0.9],  # e5, stage 2
        ]
    )

    bwt = backward_transfer_over_time(accuracy, 2, [[0], [1], [2]])

    assert np.isnan(bwt[:2]).all()
    assert bwt[2] == pytest.approx(-0.2)
    assert bwt[3] == pytest.approx(-0.4)
    assert bwt[4] == pytest.approx(((0.9 - 1.0) + (0.5 - 0.9)) / 2)
    assert bwt[5] == pytest.approx(((0.7 - 1.0) + (0.7 - 0.9)) / 2)


def test_backward_transfer_over_time_excludes_digits_introduced_in_the_same_stage() -> None:
    accuracy = np.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    bwt = backward_transfer_over_time(accuracy, 2, [[0, 1], [2]])

    assert np.isnan(bwt[:2]).all()  # both digits come from stage 0, nothing earlier in stage 0
    assert bwt[2] == pytest.approx(-1.0)


def test_backward_transfer_over_time_rejects_more_epochs_than_stages_provide() -> None:
    with pytest.raises(ValueError, match="exceed"):
        backward_transfer_over_time(np.zeros((5, 3)), 2, [[0], [1]])
