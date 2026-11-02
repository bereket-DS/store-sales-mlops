"""The shared contract is the one thing all three members depend on, so it is
the one thing with tests. Run `pytest` before changing src/common/evaluate.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.evaluate import (
    HORIZON_DAYS,
    MIN_LAG,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VALID_END,
    VALID_START,
    check_lags,
    rmsle,
    split_train_valid,
)


def test_rmsle_is_zero_for_perfect_predictions():
    y = np.array([0.0, 1.0, 10.0, 1234.5])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_known_value():
    # log1p(e-1) - log1p(0) == 1
    assert rmsle([0.0], [np.e - 1]) == pytest.approx(1.0)


def test_rmsle_clips_negative_predictions():
    assert rmsle([0.0], [-5.0]) == pytest.approx(0.0)


def test_rmsle_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        rmsle([1.0, 2.0], [1.0])


def test_rmsle_rejects_nan():
    with pytest.raises(ValueError):
        rmsle([1.0], [np.nan])


def test_validation_window_matches_the_competition_horizon():
    assert (VALID_END - VALID_START).days + 1 == HORIZON_DAYS
    assert (TEST_END - TEST_START).days + 1 == HORIZON_DAYS
    assert TRAIN_END < VALID_START < VALID_END < TEST_START


def test_split_boundaries_are_inclusive_and_exclude_the_test_window():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2017-07-29", "2017-07-30", "2017-07-31", "2017-08-15", "2017-08-16"]
            )
        }
    )
    train, valid = split_train_valid(frame)

    assert list(train["date"].dt.strftime("%Y-%m-%d")) == ["2017-07-29", "2017-07-30"]
    assert list(valid["date"].dt.strftime("%Y-%m-%d")) == ["2017-07-31", "2017-08-15"]
    assert len(train) + len(valid) == 4  # the 08-16 row is in neither


def test_check_lags_rejects_anything_inside_the_horizon():
    with pytest.raises(ValueError, match="leak"):
        check_lags([1, 7, 28])


def test_check_lags_accepts_horizon_and_above():
    check_lags([MIN_LAG, 21, 28])  # must not raise
