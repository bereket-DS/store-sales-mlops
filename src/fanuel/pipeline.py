"""
Main ML pipeline for Fanuel's model.
Orchestrates training, MLflow logging, and Kaggle submission generation.

Modes:
    python -m src.fanuel.pipeline              # Runs validation + logs to MLflow
    python -m src.fanuel.pipeline --submit     # Trains on full data + writes submission.csv
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.fanuel.data_loader import (
    _REPO_ROOT, OUTPUT_DIR, MLFLOW_EXPERIMENT, AUTHOR, LGB_PARAMS,
    get_git_info, get_raw_data_md5, load_data,
    HORIZON_DAYS, TARGET, rmsle, split_train_valid, TRAIN_END, VALID_START, VALID_END
)
from src.fanuel.feature_engineering import add_safe_rolling_features, build_features

# ============================================================
# MLFLOW LOGGING
# ============================================================
def log_to_mlflow(score, lgb_params, n_train, n_valid, model, artifact_path=None):
    """Logs params, metrics, model, and artifacts to DagsHub MLflow."""
    if not MLFLOW_AVAILABLE:
        print("WARNING: mlflow not installed. Skipping logging.")
        return

    branch, sha = get_git_info()
    raw_md5 = get_raw_data_md5()

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"lgbm-safe-rolling-{int(time.time())}"):

            # Metadata Tags
            mlflow.set_tag("author", AUTHOR)
            mlflow.set_tag("approach", "lgbm_log1p_safe_rolling")
            mlflow.set_tag("git_branch", branch)
            mlflow.set_tag("git_sha", sha)
            mlflow.set_tag("model", "lgbm_safe_rolling")

            # Parameters
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

            # Log AND register the model in a single call
            mlflow.lightgbm.log_model(
                model,
                name="lgbm_model",
                registered_model_name="store-sales-fanuel-lgbm-safe-lags"
            )

            # Artifacts
            if artifact_path and Path(artifact_path).exists():
                mlflow.log_artifact(str(artifact_path))

        print(f"MLflow: Logged score {score:.5f} to '{MLFLOW_EXPERIMENT}'")
    except Exception as e:
        print(f"WARNING: MLflow logging failed: {e}")

# ============================================================
# EXECUTION MODES
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

    print(f"\nValidation RMSLE: {score:.5f}")

    # Save validation predictions for DVC pipeline
    valid_split["pred_sales"] = pred_valid
    out_path = OUTPUT_DIR / "valid_predictions.parquet"
    valid_split.to_parquet(out_path, index=False)

    # Log to MLflow
    drift_plot_path = _REPO_ROOT / "monitoring" / "fanuel" / "distribution_drift.png"
    log_to_mlflow(score, LGB_PARAMS, len(train_split), len(valid_split), model, drift_plot_path)


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
# MAIN
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