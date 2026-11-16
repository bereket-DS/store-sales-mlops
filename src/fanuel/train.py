"""
Fanuel's Day-Specific LightGBM Model with Safe Rolling Features.

Modes:
    python -m src.fanuel.train              # Runs validation + logs to MLflow
    python -m src.fanuel.train --submit     # Trains on full data + writes submission.csv
"""

import argparse
import os
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


# ============================================================
# 1. FROZEN CONTRACT SETUP
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
        HORIZON_DAYS, TARGET, check_lags, rmsle, split_train_valid,
        TRAIN_END, VALID_START, VALID_END
    )
except ImportError:
    print(f"ERROR: Could not import src.common.evaluate from {_REPO_ROOT}.")
    sys.exit(1)


# ============================================================
# 2. HELPER FUNCTIONS (Metadata Tracking)
# ============================================================
def get_git_info() -> tuple[str, str]:
    """Fetches current git branch and commit SHA for MLflow."""
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return branch, sha
    except Exception:
        return "undefined", "undefined"


def get_raw_data_md5() -> str:
    """Reads MD5 hash from data/raw.dvc to track data lineage."""
    dvc_file = _REPO_ROOT / "data" / "raw.dvc"
    try:
        with open(dvc_file, "r") as f:
            dvc_data = yaml.safe_load(f)
        return dvc_data["outs"][0]["md5"]
    except Exception:
        return "undefined"


# ============================================================
# 3. CONFIGURATION
# ============================================================
DATA_DIR = _REPO_ROOT / "data" / "raw"
OUTPUT_DIR = _REPO_ROOT / "data" / "processed" / "fanuel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_EXPERIMENT = "store-sales-fanuel"
AUTHOR = "fanuel"

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)


# ============================================================
# 4. DATA LOADING
# ============================================================
def load_data():
    """Loads CSVs and ensures 'date' columns are parsed correctly."""
    print("Loading data...")

    def read(name, **kwargs):
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run 'dvc pull'.")

        header_cols = pd.read_csv(path, nrows=0).columns
        if "date" in header_cols:
            kwargs["parse_dates"] = ["date"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, **kwargs)

        if "date" not in df.columns or pd.api.types.is_datetime64_any_dtype(df["date"]):
            return df

        # Robust fallback parser for inconsistent date formats
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

    return read("train.csv"), read("test.csv"), read("stores.csv"), read("holidays_events.csv")


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
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

    g = df.groupby(["store_nbr", "family"])[TARGET]

    # Shift by 16 days BEFORE rolling to ensure we only use past data
    df["recent_mean_4w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(28, min_periods=1).mean())
    df["recent_mean_8w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(56, min_periods=1).mean())
    df["recent_mean_12w"] = g.transform(lambda s: s.shift(HORIZON_DAYS).rolling(84, min_periods=1).mean())

    gd = df.groupby(["store_nbr", "family", df["date"].dt.dayofweek])[TARGET]
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
        "weekday": df["date"].dt.dayofweek,
    })

    for col in ["recent_mean_4w", "recent_mean_8w", "recent_mean_12w", "recent_mean_dow"]:
        feat[col] = df[col].values if col in df.columns else 0

    feat["is_holiday"] = compute_is_holiday(df, stores, holidays).values
    return feat, train_categories


# ============================================================
# 6. MLFLOW LOGGING
# ============================================================
def log_to_mlflow(score, lgb_params, n_train, n_valid):
    """Logs params, metrics, and metadata to DagsHub MLflow."""
    branch, sha = get_git_info()
    raw_md5 = get_raw_data_md5()

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"lgbm-safe-rolling-{int(time.time())}"):
            # Metadata
            mlflow.set_tag("author", AUTHOR)
            mlflow.set_tag("approach", "lgbm_log1p_safe_rolling")
            mlflow.set_tag("git_branch", branch)
            mlflow.set_tag("git_sha", sha)
            mlflow.set_tag("model", "lgbm_safe_rolling")

            # Params
            mlflow.log_params(lgb_params)
            mlflow.log_param("window_days", HORIZON_DAYS)
            mlflow.log_param("train_end", str(TRAIN_END.date()))
            mlflow.log_param("valid_start", str(VALID_START.date()))
            mlflow.log_param("valid_end", str(VALID_END.date()))
            mlflow.log_param("raw_data_md5", raw_md5)

            # Metrics
            mlflow.log_metric("rmsle_val", score)
            mlflow.log_metric("n_train_rows", n_train)
            mlflow.log_metric("n_valid_rows", n_valid)

        print(f"MLflow: Logged score {score:.5f} to '{MLFLOW_EXPERIMENT}'")
    except Exception as e:
        print(f"WARNING: MLflow logging failed: {e}")


# ============================================================
# 7. EXECUTION MODES
# ============================================================
def run_validation(train, stores, holidays):
    """Trains on frozen split, evaluates, and logs to MLflow."""
    print("\n--- Starting Validation Mode ---")
    train_rolled = add_safe_rolling_features(train)
    train_split, valid_split = split_train_valid(train_rolled)
    print(f"Train rows: {len(train_split):,} | Valid rows: {len(valid_split):,}")

    # Prepare features and log-transform target
    X_train, cats = build_features(train_split, stores, holidays)
    y_train = np.log1p(train_split[TARGET])
    X_valid, _ = build_features(valid_split, stores, holidays, train_categories=cats)

    print("Training LightGBM on frozen split...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train, categorical_feature=["store_nbr", "family"])

    # Predict, reverse log1p, and clip negatives
    pred_valid = np.clip(np.expm1(model.predict(X_valid)), 0, None)
    score = rmsle(valid_split[TARGET], pred_valid)

    print(f"\n✅ Validation RMSLE: {score:.5f}")
    log_to_mlflow(score, LGB_PARAMS, len(train_split), len(valid_split))


def run_submission(train, test, stores, holidays):
    """Trains on all available data and generates Kaggle submission.csv."""
    print("\n--- Starting Submission Mode ---")
    train_rolled = add_safe_rolling_features(train)

    X_submit_train, cats = build_features(train_rolled, stores, holidays)
    y_submit_train = np.log1p(train_rolled[TARGET])

    print(f"Training final model on all {len(train_rolled):,} rows...")
    final_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_model.fit(X_submit_train, y_submit_train, categorical_feature=["store_nbr", "family"])

    # Fetch most recent rolling stats for the test set
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


# ============================================================
# 8. MAIN
# ============================================================
def main():
    load_dotenv()  # Load DagsHub credentials from .env

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