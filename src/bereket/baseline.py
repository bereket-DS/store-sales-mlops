"""Baseline: each store x family is predicted as the mean of its last N days.

Uses the shared contract (split_train_valid, rmsle) so the score is comparable
with every other model in the repo, and logs the run to MLflow on DagsHub.

Run from the repo root:
    python -m src.bereket.baseline
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.common.evaluate import (
    TARGET,
    TRAIN_END,
    VALID_END,
    VALID_START,
    rmsle,
    split_train_valid,
)

EXPERIMENT = "store-sales-bereket"
KEYS = ["store_nbr", "family"]
REPO_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger("bereket.baseline")


def load_train(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "train.csv"
    log.info("reading %s", path)
    df = pd.read_csv(
        path,
        usecols=["date", "store_nbr", "family", TARGET],
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", TARGET: "float32"},
    )
    log.info("loaded %d rows, %s .. %s", len(df), df["date"].min().date(), df["date"].max().date())
    return df


def fit_last_n_days_mean(train: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Mean sales per store x family over the last `window_days` of training."""
    cutoff = TRAIN_END - pd.Timedelta(days=window_days - 1)
    recent = train.loc[train["date"] >= cutoff]
    log.info(
        "fitting on %s .. %s (%d days, %d rows)",
        recent["date"].min().date(), recent["date"].max().date(), window_days, len(recent),
    )
    return (
        recent.groupby(KEYS, observed=True)[TARGET]
        .mean()
        .rename("prediction")
        .reset_index()
    )


def predict(model: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    out = frame[KEYS].merge(model, on=KEYS, how="left")["prediction"]
    missing = int(out.isna().sum())
    if missing:
        log.warning("%d rows have no history in the window - predicting 0", missing)
    return out.fillna(0.0).to_numpy()


def raw_data_md5() -> str:
    """DVC hash of data/raw, so the run is tied to an exact data version."""
    meta = yaml.safe_load((REPO_ROOT / "data" / "raw.dvc").read_text())
    return meta["outs"][0]["md5"]


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--window-days", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        raise SystemExit("MLFLOW_TRACKING_PASSWORD is empty - put your DagsHub token in .env")

    df = load_train(args.data_dir)
    train, valid = split_train_valid(df)
    log.info("split: train %d rows, valid %d rows", len(train), len(valid))

    model = fit_last_n_days_mean(train, args.window_days)
    preds = predict(model, valid)
    score = rmsle(valid[TARGET], preds)
    log.info("RMSLE on validation (%s .. %s): %.5f", VALID_START.date(), VALID_END.date(), score)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"baseline-mean-last-{args.window_days}d") as run:
        mlflow.set_tags(
            {
                "author": "bereket",
                "model": "baseline",
                "git_sha": git("rev-parse", "HEAD"),
                "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            }
        )
        mlflow.log_params(
            {
                "window_days": args.window_days,
                "train_end": TRAIN_END.date().isoformat(),
                "valid_start": VALID_START.date().isoformat(),
                "valid_end": VALID_END.date().isoformat(),
                "raw_data_md5": raw_data_md5(),
            }
        )
        mlflow.log_metric("rmsle_val", score)

        tracking = mlflow.get_tracking_uri()
        exp_id = run.info.experiment_id
        log.info("MLflow run: %s/#/experiments/%s/runs/%s", tracking, exp_id, run.info.run_id)


if __name__ == "__main__":
    main()
