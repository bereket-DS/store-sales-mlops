"""Tehreem's first baseline: recent mean sales for the same weekday.

Run from the repository root:
    python -m src.tehreem.baseline
    python -m src.tehreem.baseline --log-mlflow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from src.common.evaluate import TARGET, TRAIN_END, VALID_END, VALID_START, rmsle, split_train_valid

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["store_nbr", "family"]


def predict_weekday_mean(
    history: pd.DataFrame, forecast: pd.DataFrame, window_days: int = 28
) -> np.ndarray:
    """Predict from history only; forecast sales are never used.

    For example, a Monday prediction uses the last four training Mondays
    when window_days=28. Missing weekday groups use their store/family mean;
    groups without recent history use the overall recent mean.
    """
    if window_days < 7:
        raise ValueError("window_days must cover at least one week (7 days)")
    if history.empty or forecast.empty:
        raise ValueError("history and forecast must both contain rows")
    history_end = history["date"].max()
    if history_end >= forecast["date"].min():
        raise ValueError("history must end before the forecast begins to avoid leakage")

    start = history_end - pd.Timedelta(days=window_days - 1)
    recent = history.loc[history["date"] >= start, [*KEYS, "date", TARGET]].copy()
    recent["weekday"] = recent["date"].dt.day_of_week
    weekday_means = (
        recent.groupby([*KEYS, "weekday"], observed=True)[TARGET]
        .mean().rename("prediction").reset_index()
    )
    group_means = (
        recent.groupby(KEYS, observed=True)[TARGET]
        .mean().rename("fallback").reset_index()
    )

    rows = forecast[[*KEYS, "date"]].copy()
    rows["weekday"] = rows["date"].dt.day_of_week
    rows["_row"] = np.arange(len(rows))
    rows = rows.merge(weekday_means, on=[*KEYS, "weekday"], how="left", validate="many_to_one")
    rows = rows.merge(group_means, on=KEYS, how="left", validate="many_to_one")
    rows = rows.sort_values("_row")
    return (
        rows["prediction"].fillna(rows["fallback"]).fillna(recent[TARGET].mean())
        .clip(lower=0).to_numpy(dtype="float64")
    )


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def log_mlflow(report: dict, predictions_path: Path, report_path: Path) -> None:
    """Log only when explicitly requested with --log-mlflow."""
    from dotenv import load_dotenv
    import mlflow

    load_dotenv(ROOT / ".env", override=False)
    required = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
    if any(not os.getenv(key) or "<" in os.environ[key] for key in required):
        raise ValueError("Fill in your DagsHub credentials and tracking URI in .env first")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("store-sales-tehreem")
    with mlflow.start_run(run_name=f"weekday-mean-{report['window_days']}d") as run:
        mlflow.set_tags({
            "author": "tehreem",
            "git_sha": report["git_sha"],
            "git_dirty": str(report["git_dirty"]).lower(),
            "source_sha256": report["source_sha256"],
        })
        mlflow.log_params({key: report[key] for key in [
            "model", "window_days", "train_end", "valid_start", "valid_end", "raw_data_md5",
        ]})
        mlflow.log_metric("rmsle_val", report["rmsle_val"])
        mlflow.log_artifact(str(predictions_path), artifact_path="predictions")
        mlflow.log_artifact(str(report_path), artifact_path="metrics")
        # Preserve the exact implementation even before its first Git commit.
        mlflow.log_artifact(__file__, artifact_path="source")
        print(f"MLflow experiment: store-sales-tehreem; run: {run.info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window-days", type=int, default=28)
    parser.add_argument("--log-mlflow", action="store_true", help="Log this run to DagsHub")
    args = parser.parse_args()
    if args.window_days < 7:
        parser.error("--window-days must be at least 7")

    data = pd.read_csv(
        ROOT / "data/raw/train.csv",
        usecols=["id", "date", *KEYS, TARGET],
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", TARGET: "float32"},
    )
    train, valid = split_train_valid(data)
    predictions = predict_weekday_mean(train, valid, args.window_days)
    score = rmsle(valid[TARGET].to_numpy(), predictions)

    predictions_path = ROOT / "data/processed/tehreem/weekday_baseline_valid.csv"
    report_path = ROOT / "reports/tehreem/weekday_baseline_metrics.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output = valid[["id", "date", *KEYS, TARGET]].copy()
    output["prediction"] = predictions
    output.to_csv(predictions_path, index=False)
    report = {
        "model": "recent_weekday_mean",
        "window_days": args.window_days,
        "rmsle_val": score,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_end": str(TRAIN_END.date()),
        "valid_start": str(VALID_START.date()),
        "valid_end": str(VALID_END.date()),
        "raw_data_md5": yaml.safe_load((ROOT / "data/raw.dvc").read_text())["outs"][0]["md5"],
        "git_sha": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Training: through {TRAIN_END.date()} ({len(train):,} rows)")
    print(f"Validation: {VALID_START.date()} to {VALID_END.date()} ({len(valid):,} rows)")
    print(f"Recent history: {args.window_days} days; validation RMSLE: {score:.5f}")
    print(f"Predictions: {predictions_path.relative_to(ROOT)}")
    print(f"Metrics: {report_path.relative_to(ROOT)}")
    if args.log_mlflow:
        log_mlflow(report, predictions_path, report_path)


if __name__ == "__main__":
    main()
