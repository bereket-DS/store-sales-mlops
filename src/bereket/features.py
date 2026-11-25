"""Feature engineering - built in memory, never written to disk.

The training script calls `load_prepared()` and `build_features()`; nothing
here saves a file. Only data/raw and data/processed are stored (DVC). The
feature table changes with every experiment, so it is rebuilt each run
(seconds) instead of stored (a new large file in DVC every time).

Every lag and rolling window starts at least MIN_LAG (16) days back, the
forecast horizon: on the last day of the test window, the newest known sales
are 16 days old. `check_lags()` from the shared contract enforces this.

Run from the repo root for a check (prints feature columns and missing
values, writes nothing):
    python -m src.bereket.features
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.evaluate import MIN_LAG, TARGET, check_lags, split_train_valid

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS = ["store_nbr", "family"]

# Defaults - an experiment can pass its own.
SALES_LAGS = [16, 21, 28, 35, 364]          # 364 = same weekday, one year back
ROLL_WINDOWS = [7, 14, 28, 56]              # means over the w days ending MIN_LAG ago
LOG_ROLL_WINDOWS = [7, 14, 28, 56, 112]     # same, on log1p(sales)
EWM_ALPHAS = [0.1, 0.3]                     # exponentially weighted means of log1p(sales)
TRANSACTION_LAGS = [16, 28]

CATEGORICAL = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
FLAGS = [
    "holiday_national", "holiday_regional", "holiday_local",
    "event_national", "bridge", "workday", "is_day_off",
]
KNOWN_IN_ADVANCE = ["onpromotion", "oil_price"]

log = logging.getLogger("bereket.features")


def load_prepared(processed_dir: Path) -> pd.DataFrame:
    """Train and test from prepare.py, stacked into one frame.

    Stacked because test's lags reach back into train: the lag-16 of
    2017-08-31 is the sales of 2017-08-15.
    """
    train = pd.read_parquet(processed_dir / "train.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    log.info("loaded prepared train %d rows, test %d rows", len(train), len(test))
    df = pd.concat([train, test], ignore_index=True)
    # concat of two categoricals with different categories falls back to object
    df["family"] = df["family"].astype("category")
    return df


def _complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add the missing days (Dec 25: stores closed) with sales = nan.

    Lags are computed by shifting rows within each store x family. That only
    means "N days ago" if every day has a row - otherwise a shift of 16 rows
    across Christmas would be 17 days. The added rows are removed again at
    the end (column `_filler`).
    """
    days = pd.date_range(df["date"].min(), df["date"].max(), freq="D", name="date")
    combos = df[KEYS].drop_duplicates()
    grid = combos.merge(days.to_frame(index=False), how="cross")
    out = grid.merge(df, on=["date", *KEYS], how="left", indicator=True)
    out["_filler"] = out.pop("_merge").eq("left_only")
    log.info("calendar: added %d filler rows for missing days", int(out["_filler"].sum()))
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["day_of_week"] = d.day_of_week.astype("int8")
    df["day_of_month"] = d.day.astype("int8")
    df["month"] = d.month.astype("int8")
    df["year"] = d.year.astype("int16")
    # Public-sector wages in Ecuador are paid on the 15th and the last day of
    # the month - a known spending spike.
    df["is_payday"] = (d.day.eq(15) | d.is_month_end).astype("int8")
    return df


def add_lags(
    df: pd.DataFrame,
    sales_lags: list[int],
    roll_windows: list[int],
    transaction_lags: list[int],
) -> pd.DataFrame:
    check_lags(sales_lags)
    check_lags(transaction_lags)

    g = df.groupby(KEYS, observed=True)
    for lag in sales_lags:
        df[f"sales_lag_{lag}"] = g[TARGET].shift(lag)

    # Rolling means over the w days that END at the MIN_LAG lag - so the
    # newest day inside any window is MIN_LAG days old.
    base = g[TARGET].shift(MIN_LAG)
    base_g = base.groupby([df[k] for k in KEYS], observed=True)
    for w in roll_windows:
        df[f"sales_roll_mean_{w}"] = base_g.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )

    for lag in transaction_lags:
        df[f"transactions_lag_{lag}"] = g["transactions"].shift(lag)
    return df


def add_log_history(df: pd.DataFrame, roll_windows: list[int]) -> pd.DataFrame:
    """Rolling stats on log1p(sales) - the scale the model is trained on.

    A mean of raw sales is pulled up by a few huge days; a mean of log sales
    matches what RMSLE cares about. Same rule as add_lags: every window ends
    MIN_LAG days back.
    """
    keys = [df[k] for k in KEYS]
    base = np.log1p(df[TARGET]).groupby(keys, observed=True).shift(MIN_LAG)
    base_g = base.groupby(keys, observed=True)
    for w in roll_windows:
        df[f"log_roll_mean_{w}"] = base_g.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    df["log_roll_std_28"] = base_g.transform(lambda s: s.rolling(28, min_periods=2).std())
    for alpha in EWM_ALPHAS:
        df[f"log_ewm_{alpha}"] = base_g.transform(lambda s, a=alpha: s.ewm(alpha=a).mean())
    # share of zero-sales days (filler days count as unknown, not as zero)
    is_zero = base.eq(0).where(base.notna())
    df["zero_frac_56"] = is_zero.groupby(keys, observed=True).transform(
        lambda s: s.rolling(56, min_periods=1).mean()
    )
    return df


def add_promo(df: pd.DataFrame) -> pd.DataFrame:
    """Promotion context. `onpromotion` is known in advance (it is in test.csv),
    so the current day, past days AND the next days may be used.
    """
    keys = [df[k] for k in KEYS]
    promo = df["onpromotion"].astype("float64")
    pg = promo.groupby(keys, observed=True)
    df["promo_roll_mean_14"] = pg.transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["promo_lead_mean_7"] = pg.transform(lambda s: s[::-1].rolling(7, min_periods=1).mean()[::-1])
    # promotions now vs. the period the sales lags come from
    past = pg.shift(MIN_LAG).groupby(keys, observed=True).transform(
        lambda s: s.rolling(28, min_periods=1).mean()
    )
    df["promo_vs_past_28"] = promo - past
    return df


def build_features(
    df: pd.DataFrame,
    sales_lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
    transaction_lags: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame with features, list of feature column names).

    The frame keeps id, date and sales so it can be split with the shared
    split_train_valid(). Same-day `transactions` is NOT a feature - only its
    lags are.
    """
    sales_lags = SALES_LAGS if sales_lags is None else sales_lags
    roll_windows = ROLL_WINDOWS if roll_windows is None else roll_windows
    transaction_lags = TRANSACTION_LAGS if transaction_lags is None else transaction_lags

    start = time.perf_counter()
    df = _complete_calendar(df)
    df = df.sort_values([*KEYS, "date"], ignore_index=True)
    df = add_lags(df, sales_lags, roll_windows, transaction_lags)
    df = add_log_history(df, LOG_ROLL_WINDOWS)
    df = add_promo(df)
    df = df.loc[~df.pop("_filler")].reset_index(drop=True)
    df = add_calendar(df)

    for col in FLAGS:
        df[col] = df[col].astype("int8")
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")

    lag_cols = [c for c in df.columns if c.startswith(("sales_lag_", "sales_roll_", "transactions_lag_"))]
    log_cols = [c for c in df.columns if c.startswith(("log_roll_", "log_ewm_"))] + ["zero_frac_56"]
    promo_cols = ["promo_roll_mean_14", "promo_lead_mean_7", "promo_vs_past_28"]
    calendar_cols = ["day_of_week", "day_of_month", "month", "year", "is_payday"]
    features = CATEGORICAL + KNOWN_IN_ADVANCE + FLAGS + calendar_cols + lag_cols + log_cols + promo_cols

    log.info(
        "built %d features for %d rows in %.1fs", len(features), len(df), time.perf_counter() - start
    )
    return df, features


def verify_lags(df: pd.DataFrame, n: int = 200, seed: int = 0) -> None:
    """Spot-check that sales_lag_L really is the sales of date - L days."""
    lookup = df.set_index(["date", *KEYS])[TARGET]
    lag_cols = [c for c in df.columns if c.startswith("sales_lag_")]
    sample = df.dropna(subset=lag_cols).sample(n, random_state=seed)
    for col in lag_cols:
        lag = int(col.rsplit("_", 1)[1])
        keys = list(zip(sample["date"] - pd.Timedelta(days=lag), sample["store_nbr"], sample["family"]))
        expected = lookup.reindex(keys).to_numpy()
        if not ((sample[col].to_numpy() == expected) | pd.isna(expected)).all():
            raise AssertionError(f"{col} does not match sales {lag} days earlier")
    log.info("verified %s on %d random rows: each equals sales exactly L days earlier", lag_cols, n)


def log_summary(df: pd.DataFrame, features: list[str]) -> None:
    train, valid = split_train_valid(df)
    test = df.loc[df[TARGET].isna() & df["date"].gt(valid["date"].max())]
    log.info("split: train %d rows | valid %d rows | test %d rows", len(train), len(valid), len(test))
    log.info("features (%d): %s", len(features), ", ".join(features))
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        missing = part[features].isna().mean().mul(100)
        missing = missing[missing > 0].round(2)
        log.info("%s missing %%: %s", name, missing.to_dict() if len(missing) else "none")
    log.info("memory: %.0f MB (in RAM only - nothing written)", df.memory_usage(deep=True).sum() / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "bereket"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    df, features = build_features(load_prepared(args.processed_dir))
    verify_lags(df)
    log_summary(df, features)


if __name__ == "__main__":
    main()
