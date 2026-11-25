"""The backtest/submit windows must reduce to the frozen contract at shift 0."""

import pandas as pd

from src.bereket.train import windows
from src.common.evaluate import TEST_END, TEST_START, split_train_valid

START = pd.Timestamp("2017-01-01")


def frame():
    dates = pd.date_range(START, TEST_END, freq="D")
    return pd.DataFrame({"date": dates, "sales": range(len(dates))})


def test_shift_zero_is_the_frozen_split():
    df = frame()
    fit, inner, train, target = windows(df, START, 0)
    ref_train, ref_valid = split_train_valid(df)
    assert train.equals(ref_train)
    assert target.equals(ref_valid)
    assert len(inner) == 16 and inner["date"].max() == ref_train["date"].max()
    assert fit["date"].max() < inner["date"].min()


def test_submit_target_is_the_test_window():
    _, inner, train, target = windows(frame(), START, 16)
    assert target["date"].min() == TEST_START and target["date"].max() == TEST_END
    assert train["date"].max() == TEST_START - pd.Timedelta(days=1)
    assert len(inner) == 16


def test_backtest_windows_never_overlap_their_target():
    for k in range(1, 4):
        fit, inner, train, target = windows(frame(), START, -16 * k)
        assert train["date"].max() < target["date"].min()
        assert len(target) == 16
