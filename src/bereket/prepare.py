"""Data preparation: merge train and test with stores, oil, holidays and transactions.

Output (one row per date x store x family, same rows as the raw files):
    data/processed/bereket/train.parquet
    data/processed/bereket/test.parquet

Run from the repo root:
    python -m src.bereket.prepare
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.common.evaluate import MIN_LAG

REPO_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger("bereket.prepare")

# Holiday types that mean "people do not work that day".
#   Holiday    - the official day (rows with transferred=True are dropped first)
#   Transfer   - the day a transferred holiday was actually observed
#   Additional - extra days added around a holiday (e.g. Navidad-1)
#   Bridge     - extra day off joining a holiday to a weekend
DAY_OFF_TYPES = {"Holiday", "Transfer", "Additional", "Bridge"}


def read_csv(data_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    path = data_dir / name
    df = pd.read_csv(path, **kwargs)
    log.info("read %s: %d rows", path.name, len(df))
    return df


def load_sales(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtypes = {"store_nbr": "int16", "family": "category", "onpromotion": "int32"}
    train = read_csv(
        data_dir, "train.csv", parse_dates=["date"], dtype={**dtypes, "sales": "float32"}
    )
    test = read_csv(data_dir, "test.csv", parse_dates=["date"], dtype=dtypes)
    return train, test


def load_stores(data_dir: Path) -> pd.DataFrame:
    stores = read_csv(
        data_dir,
        "stores.csv",
        dtype={"store_nbr": "int16", "cluster": "int8"},
    )
    # "type" would clash with the holiday type; make it explicit.
    return stores.rename(columns={"type": "store_type"})


def load_oil(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily oil price on a full calendar.

    The raw file only has trading days, so weekends are missing rows, not just
    nan. Reindex to every day first, then forward-fill (carry the last known
    price) and back-fill (only affects the very first day, 2013-01-01).
    """
    oil = read_csv(data_dir, "oil.csv", parse_dates=["date"])
    oil = oil.set_index("date")["dcoilwtico"]
    calendar = pd.date_range(start, end, freq="D", name="date")
    daily = oil.reindex(calendar)
    log.info(
        "oil: %d of %d calendar days missing before filling", daily.isna().sum(), len(daily)
    )
    daily = daily.ffill().bfill().astype("float32")
    return daily.rename("oil_price").reset_index()


def load_holidays(data_dir: Path) -> pd.DataFrame:
    """Clean holiday table: one row per real event, transferred ones removed."""
    h = read_csv(data_dir, "holidays_events.csv", parse_dates=["date"])

    # A holiday with transferred=True was NOT observed on that date - it was a
    # normal working day. The day off is the matching "Transfer" row instead.
    n_transferred = int(h["transferred"].sum())
    h = h.loc[~h["transferred"]].copy()
    log.info("holidays: dropped %d transferred rows (normal days)", n_transferred)

    h["is_day_off"] = h["type"].isin(DAY_OFF_TYPES)
    h["is_event"] = h["type"].eq("Event")
    h["is_workday"] = h["type"].eq("Work Day")
    h["is_bridge"] = h["type"].eq("Bridge")
    return h


def _flags_by(h: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Collapse several holidays on the same date/place into one row (any = True)."""
    return h.groupby(["date", key], as_index=False)[cols].max()


def holiday_features(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, store_nbr) that has any holiday or event.

    National rows apply to every store, Regional rows to stores whose state
    matches locale_name, Local rows to stores whose city matches locale_name.
    """
    national = holidays.loc[holidays["locale"] == "National"]
    regional = holidays.loc[holidays["locale"] == "Regional"]
    local = holidays.loc[holidays["locale"] == "Local"]

    nat_cols = ["is_day_off", "is_event", "is_workday", "is_bridge"]
    nat = (
        _flags_by(national.assign(key="all"), "key", nat_cols)
        .drop(columns="key")
        .rename(
            columns={
                "is_day_off": "holiday_national",
                "is_event": "event_national",
                "is_workday": "workday",
                "is_bridge": "bridge",
            }
        )
    )
    nat = nat.merge(stores[["store_nbr"]], how="cross")

    reg = (
        _flags_by(regional.rename(columns={"locale_name": "state"}), "state", ["is_day_off"])
        .rename(columns={"is_day_off": "holiday_regional"})
        .merge(stores[["store_nbr", "state"]], on="state")
        .drop(columns="state")
    )
    loc = (
        _flags_by(local.rename(columns={"locale_name": "city"}), "city", ["is_day_off"])
        .rename(columns={"is_day_off": "holiday_local"})
        .merge(stores[["store_nbr", "city"]], on="city")
        .drop(columns="city")
    )

    feats = nat.merge(reg, on=["date", "store_nbr"], how="outer").merge(
        loc, on=["date", "store_nbr"], how="outer"
    )
    flag_cols = [c for c in feats.columns if c not in ("date", "store_nbr")]
    feats[flag_cols] = feats[flag_cols].fillna(False).astype(bool)
    feats["store_nbr"] = feats["store_nbr"].astype("int16")
    return feats


def load_transactions(data_dir: Path) -> pd.DataFrame:
    return read_csv(
        data_dir,
        "transactions.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "transactions": "float32"},
    )


def merge_all(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    n = len(sales)
    df = sales.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(holidays, on=["date", "store_nbr"], how="left")
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    if len(df) != n:
        raise RuntimeError(f"merge changed row count: {n} -> {len(df)}")

    flag_cols = [c for c in holidays.columns if c not in ("date", "store_nbr")]
    df[flag_cols] = df[flag_cols].fillna(False).astype(bool)

    # Day off for the store: any matching holiday, or a weekend that was not
    # turned into a working day.
    weekend = df["date"].dt.day_of_week >= 5
    df["is_day_off"] = (
        df["holiday_national"] | df["holiday_regional"] | df["holiday_local"]
        | (weekend & ~df["workday"])
    )

    for col in ["city", "state", "store_type"]:
        df[col] = df[col].astype("category")
    return df


def log_summary(name: str, df: pd.DataFrame) -> None:
    log.info(
        "%s: %d rows, %s .. %s, %d stores x %d families",
        name, len(df), df["date"].min().date(), df["date"].max().date(),
        df["store_nbr"].nunique(), df["family"].nunique(),
    )
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        log.info("%s: no missing values", name)
    for col, count in missing.items():
        log.info("%s: missing %-14s %9d (%.2f%%)", name, col, count, 100 * count / len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "bereket"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    train, test = load_sales(args.data_dir)
    stores = load_stores(args.data_dir)
    start = min(train["date"].min(), test["date"].min())
    end = max(train["date"].max(), test["date"].max())
    oil = load_oil(args.data_dir, start, end)
    holidays = holiday_features(load_holidays(args.data_dir), stores)
    transactions = load_transactions(args.data_dir)

    # transactions is the number of receipts on the SAME day - unknown at
    # prediction time and absent from test. Kept as a column so a later
    # feature step can lag it; never use it unshifted, and lag it by at least
    # MIN_LAG days (the forecast horizon).
    log.warning(
        "transactions is same-day data: use only as a lag >= %d days, never raw", MIN_LAG
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, sales in [("train", train), ("test", test)]:
        df = merge_all(sales, stores, oil, holidays, transactions)
        log_summary(name, df)
        path = args.out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
