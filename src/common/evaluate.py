"""FROZEN EVALUATION CONTRACT - shared by all three members.

Each member writes their own data preparation, features, model, Docker setup
and monitoring. This file is the only thing they share, and it is the only
reason their three scores are comparable: the same metric, on the same rows.

Do not change anything here without all three members agreeing.

The validation window mirrors the real competition test window - the last 16
days before it. Test is 2017-08-16 .. 2017-08-31, so validation is
2017-07-31 .. 2017-08-15 and training stops the day before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2017-07-30")
VALID_START = pd.Timestamp("2017-07-31")
VALID_END = pd.Timestamp("2017-08-15")
TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")

HORIZON_DAYS = 16
TARGET = "sales"


def rmsle(y_true, y_pred) -> float:
    """Root mean squared logarithmic error - the competition metric.

    Predictions are clipped at zero first: negative sales are impossible and
    log1p would return nan.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if np.isnan(y_pred).any():
        raise ValueError("predictions contain nan")

    y_pred = np.clip(y_pred, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def split_train_valid(
    df: pd.DataFrame, date_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on the frozen date boundaries.

    Rows in the real test window are in neither half. Call this on your
    prepared data, however you prepared it.
    """
    dates = pd.to_datetime(df[date_col])
    train = df.loc[dates <= TRAIN_END].copy()
    valid = df.loc[(dates >= VALID_START) & (dates <= VALID_END)].copy()
    return train, valid


def describe_split() -> str:
    return (
        f"train <= {TRAIN_END.date()} | "
        f"valid {VALID_START.date()} .. {VALID_END.date()} ({HORIZON_DAYS} days) | "
        f"test {TEST_START.date()} .. {TEST_END.date()}"
    )


# A lag shorter than the horizon uses information that will not exist at
# prediction time. Guard your own features with this.
MIN_LAG = HORIZON_DAYS


def check_lags(lags) -> None:
    """Raise if any lag is shorter than the 16-day forecast horizon.

    This is the easiest way to accidentally cheat on this competition: a
    7-day lag makes validation look great and the leaderboard terrible.
    """
    bad = [lag for lag in lags if lag < MIN_LAG]
    if bad:
        raise ValueError(
            f"lags {bad} are shorter than the {MIN_LAG}-day forecast horizon - "
            f"they leak future information"
        )
