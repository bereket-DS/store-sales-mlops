"""
Feature engineering for Fanuel's model.
Creates holiday flags and safe 16-day shifted rolling features.
"""

import pandas as pd
from src.fanuel.data_loader import HORIZON_DAYS, TARGET, check_lags

def compute_is_holiday(df, stores, holidays):
    """Merges national, regional, and local holidays, excluding transferred ones."""
    real_holidays = holidays[
        holidays["type"].isin(["Holiday", "Additional", "Bridge", "Transfer"]) &
        ~((holidays["type"] == "Holiday") & (holidays["transferred"] == True))
        ].copy()

    national_df = real_holidays[real_holidays["locale"] == "National"][["date"]].drop_duplicates()
    national_df["is_national_holiday"] = 1

    regional_df = (real_holidays[real_holidays["locale"] == "Regional"][["date", "locale_name"]]
                   .drop_duplicates().rename(columns={"locale_name": "state"}))
    regional_df["is_regional_holiday"] = 1

    local_df = (real_holidays[real_holidays["locale"] == "Local"][["date", "locale_name"]]
                .drop_duplicates().rename(columns={"locale_name": "city"}))
    local_df["is_local_holiday"] = 1

    tmp = df[["date", "store_nbr"]].copy()
    tmp = tmp.merge(stores[["store_nbr", "city", "state"]], on="store_nbr", how="left")
    tmp = tmp.merge(national_df, on="date", how="left")
    tmp = tmp.merge(regional_df, on=["date", "state"], how="left")
    tmp = tmp.merge(local_df, on=["date", "city"], how="left")

    flags = tmp[["is_national_holiday", "is_regional_holiday", "is_local_holiday"]].fillna(0)
    return (flags.sum(axis=1) > 0).astype(int)


def add_safe_rolling_features(data):
    """Calculates rolling stats shifted by HORIZON_DAYS to prevent leakage."""
    print(f"Calculating rolling features with shift={HORIZON_DAYS} (No Leakage)...")
    df = data.sort_values(["store_nbr", "family", "date"]).copy()
    check_lags([HORIZON_DAYS])  # Enforce contract rule

    # Group only by store & family for overall rolling averages
    g = df.groupby(["store_nbr", "family"])[TARGET]

    # Shift by 16 days BEFORE rolling to ensure we only use past data
    df["recent_mean_4w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(28, min_periods=1).mean())
    df["recent_mean_8w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(56, min_periods=1).mean())
    df["recent_mean_12w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(84, min_periods=1).mean())

    # Group by store, family, AND day_of_week for the dow-specific average
    gd = df.groupby(["store_nbr", "family", df["date"].dt.day_of_week])[TARGET]
    df["recent_mean_dow"] = gd.transform(lambda s: s.shift(HORIZON_DAYS).rolling(4, min_periods=1).mean())

    stat_cols = ["recent_mean_4w", "recent_mean_8w", "recent_mean_12w", "recent_mean_dow"]
    df[stat_cols] = df[stat_cols].fillna(0)
    return df


def build_features(df, stores, holidays, train_categories=None):
    """Constructs the final feature matrix for LightGBM."""
    if train_categories:
        STORE_DTYPE, FAMILY_DTYPE = train_categories
    else:
        STORE_DTYPE = pd.CategoricalDtype(categories=sorted(df["store_nbr"].unique()))
        FAMILY_DTYPE = pd.CategoricalDtype(categories=sorted(df["family"].unique()))
        train_categories = (STORE_DTYPE, FAMILY_DTYPE)

    feat = pd.DataFrame({
        "store_nbr": df["store_nbr"].astype(STORE_DTYPE),
        "family": df["family"].astype(FAMILY_DTYPE),
        "onpromotion": df["onpromotion"],
        "year": df["date"].dt.year,
        "month": df["date"].dt.month,
        "day": df["date"].dt.day,
        "weekday": df["date"].dt.day_of_week,  # Pandas updated API
    })

    for col in ["recent_mean_4w", "recent_mean_8w", "recent_mean_12w", "recent_mean_dow"]:
        feat[col] = df[col].values if col in df.columns else 0

    feat["is_holiday"] = compute_is_holiday(df, stores, holidays).values
    return feat, train_categories