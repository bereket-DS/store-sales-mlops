"""
Modes:
    python -m src.fanuel.train --validate    # Tests on frozen split + logs to MLflow
    python -m src.fanuel.train --submit      # Trains on all data + writes submission.csv
"""

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# ============================================================
# 1. FROZEN CONTRACT IMPORT SETUP
# ============================================================
def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "src" / "common" / "evaluate.py").exists():
            return candidate
    return Path.cwd()

_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.common.evaluate import (
        HORIZON_DAYS,
        TARGET,
        check_lags,
        rmsle,
        split_train_valid,
    )
except ImportError:
    print("ERROR: Could not import src.common.evaluate.")
    print(f"Make sure {_REPO_ROOT}/src/common/evaluate.py exists.")
    sys.exit(1)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = _REPO_ROOT / "data" / "raw"
OUTPUT_DIR = _REPO_ROOT / "data" / "processed" / "fanuel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_EXPERIMENT = "store-sales-fanuel"
AUTHOR = "fanuel"

LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    print("Loading data...")
    def read(name, **kwargs):
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Have you run 'dvc pull'?")

        header_cols = pd.read_csv(path, nrows=0).columns
        if "date" in header_cols:
            kwargs["parse_dates"] = ["date"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, **kwargs)

        if "date" not in df.columns or pd.api.types.is_datetime64_any_dtype(df["date"]):
            return df

        raw = df["date"].copy()
        for parser in [
            lambda s: pd.to_datetime(s, errors="coerce", dayfirst=False),
            lambda s: pd.to_datetime(s, errors="coerce", dayfirst=True),
            lambda s: pd.to_datetime(s, errors="coerce", format="mixed"),
        ]:
            res = parser(raw.astype(str).str.strip())
            if res.isna().sum() == 0:
                df["date"] = res
                return df
        raise ValueError(f"Could not parse dates for {name}")

    train = read("train.csv")
    test = read("test.csv")
    stores = read("stores.csv")
    holidays = read("holidays_events.csv")
    return train, test, stores, holidays

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def compute_is_holiday(df, stores, holidays):
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
    store_info = tmp["store_nbr"].merge(stores, on="store_nbr", how="left")
    tmp["city"] = store_info["city"].values
    tmp["state"] = store_info["state"].values

    tmp = tmp.merge(national_df, on="date", how="left")
    tmp = tmp.merge(regional_df, on=["date", "state"], how="left")
    tmp = tmp.merge(local_df, on=["date", "city"], how="left")

    flags = tmp[["is_national_holiday", "is_regional_holiday", "is_local_holiday"]].fillna(0)
    return (flags.sum(axis=1) > 0).astype(int)

def add_safe_rolling_features(data):
    print(f"Calculating rolling features with shift={HORIZON_DAYS} (No Leakage)...")
    df = data.sort_values(["store_nbr", "family", "date"]).copy()

    # Enforce contract rule explicitly
    check_lags([HORIZON_DAYS])

    g = df.groupby(["store_nbr", "family"])[TARGET]

    # Shift by exactly the horizon (16 days)
    df["recent_mean_4w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(28, min_periods=1).mean())
    df["recent_mean_8w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(56, min_periods=1).mean())
    df["recent_mean_12w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(84, min_periods=1).mean())

    gd = df.groupby(["store_nbr", "family", df["date"].dt.dayofweek])[TARGET]
    df["recent_mean_dow"] = gd.transform(lambda s: s.shift(HORIZON_DAYS).rolling(4, min_periods=1).mean())

    stat_cols = ["recent_mean_4w", "recent_mean_8w", "recent_mean_12w", "recent_mean_dow"]
    df[stat_cols] = df[stat_cols].fillna(0)
    return df

def build_features(df, stores, holidays, train_categories=None):
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
        "weekday": df["date"].dt.dayofweek,
    })

    for col in ["recent_mean_4w", "recent_mean_8w", "recent_mean_12w", "recent_mean_dow"]:
        if col in df.columns:
            feat[col] = df[col].values
        else:
            feat[col] = 0

    feat["is_holiday"] = compute_is_holiday(df, stores, holidays).values
    return feat, train_categories

# ============================================================
# MLFLOW LOGGING
# ============================================================
def log_to_mlflow(score, lgb_params, n_train, n_valid):
    if not MLFLOW_AVAILABLE:
        print("WARNING: mlflow not installed. Skipping logging.")
        return
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"lgbm-safe-rolling-{int(time.time())}"):
            mlflow.set_tag("author", AUTHOR)
            mlflow.set_tag("approach", "lgbm_log1p_safe_rolling_16d")
            mlflow.log_params(lgb_params)
            mlflow.log_metric("rmsle_val", score)
            mlflow.log_metric("n_train_rows", n_train)
            mlflow.log_metric("n_valid_rows", n_valid)
        print(f"      MLflow: Logged score {score:.5f} to '{MLFLOW_EXPERIMENT}'")
    except Exception as e:
        print(f"      WARNING: MLflow logging failed: {e}")

# ============================================================
# MAIN EXECUTION MODES
# ============================================================
def run_validation(train, stores, holidays):
    print("\n--- Starting Validation Mode ---")
    train_rolled = add_safe_rolling_features(train)

    train_split, valid_split = split_train_valid(train_rolled)
    print(f"Train rows: {len(train_split):,} | Valid rows: {len(valid_split):,}")

    X_train, cats = build_features(train_split, stores, holidays)
    y_train = np.log1p(train_split[TARGET])
    X_valid, _ = build_features(valid_split, stores, holidays, train_categories=cats)

    print(f"Training LightGBM on frozen split...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train, categorical_feature=["store_nbr", "family"])

    pred_valid = np.clip(np.expm1(model.predict(X_valid)), 0, None)
    score = rmsle(valid_split[TARGET], pred_valid)

    print(f"\n✅ Validation RMSLE: {score:.5f}")
    log_to_mlflow(score, LGB_PARAMS, len(train_split), len(valid_split))

def run_submission(train, test, stores, holidays):
    print("\n--- Starting Submission Mode ---")
    train_rolled = add_safe_rolling_features(train)

    X_submit_train, cats = build_features(train_rolled, stores, holidays)
    y_submit_train = np.log1p(train_rolled[TARGET])

    print(f"Training final model on all {len(train_rolled):,} rows...")
    final_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_model.fit(X_submit_train, y_submit_train, categorical_feature=["store_nbr", "family"])

    # Get the most recent rolling stats from training data for the test set
    latest_stats = train_rolled.groupby(["store_nbr", "family"]).agg(
        recent_mean_4w=("recent_mean_4w", "last"),
        recent_mean_8w=("recent_mean_8w", "last"),
        recent_mean_12w=("recent_mean_12w", "last"),
        recent_mean_dow=("recent_mean_dow", "last")
    ).reset_index()

    raw_test = test[["id", "date", "store_nbr", "family", "onpromotion"]].copy()
    test_target = raw_test.merge(latest_stats, on=["store_nbr", "family"], how="left").fillna(0)

    X_test, _ = build_features(test_target, stores, holidays, train_categories=cats)

    print("Generating predictions...")
    test_pred = np.clip(np.expm1(final_model.predict(X_test)), 0, None)

    submission = pd.DataFrame({"id": test["id"], "sales": test_pred})
    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"\n✅ SUCCESS! Saved to {out_path.resolve()}")
    print(f"Total rows: {len(submission):,}")
    print(f"Mean Sales: {submission['sales'].mean():.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Run submission mode")
    args = parser.parse_args()

    start_time = time.time()
    train, test, stores, holidays = load_data()

    if args.submit:
        run_submission(train, test, stores, holidays)
    else:
        run_validation(train, stores, holidays)

    print(f"\nCompleted in {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 50)
        print("ERROR ENCOUNTERED")
        print("=" * 50)
        traceback.print_exc()
        sys.exit(1)