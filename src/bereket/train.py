"""Train a model on the in-memory features, score it on the frozen split, log to MLflow.

One script for all my models - pick one with --model. Every model gets the
same data, the same split (src/common/evaluate.py) and the same metric, so
the scores in MLflow are comparable.

    python -m src.bereket.train --model lgbm
    python -m src.bereket.train --model lgbm --no-mlflow     # quick check, nothing logged

Honest validation: early stopping uses an inner holdout (the last 16 days of
train, 2017-07-15 .. 07-30), never the validation window itself, so the
validation score is not tuned on the validation data. After early stopping
the model is refit on all of train (fit + inner) with the found number of
trees; --no-refit skips that.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.bereket.features import build_features, load_prepared
from src.common.evaluate import (
    HORIZON_DAYS,
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

INNER_START = TRAIN_END - pd.Timedelta(days=HORIZON_DAYS - 1)   # 2017-07-15

log = logging.getLogger("bereket.train")


# ---------------------------------------------------------------- models

LGBM_PARAMS = {
    "objective": "regression",      # on log1p(sales) -> optimises RMSLE
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 0,               # all cores
    "verbose": -1,
    "seed": 42,
}


def fit_predict_lgbm(fit, inner, train, valid, features, args):
    """Early-stop on the inner holdout, then (default) refit on all of train.

    The inner holdout is the 16 days right before validation - the most
    useful days to learn from. Step 1 uses them only to find the number of
    trees; step 2 retrains on fit + inner with that many trees (scaled up a
    little for the extra rows) and predicts validation with that model.
    """
    import lightgbm as lgb

    params = {**LGBM_PARAMS, "learning_rate": args.learning_rate, "num_leaves": args.num_leaves}
    dfit = lgb.Dataset(fit[features], np.log1p(fit[TARGET]), free_raw_data=False)
    dinner = lgb.Dataset(inner[features], np.log1p(inner[TARGET]), reference=dfit)
    booster = lgb.train(
        params,
        dfit,
        num_boost_round=args.num_boost_round,
        valid_sets=[dinner],
        valid_names=["inner"],
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    best = booster.best_iteration
    log.info("lgbm: best iteration %d (inner holdout)", best)

    pred_inner = np.expm1(booster.predict(inner[features], num_iteration=best))
    params_logged = {**params, "num_boost_round": args.num_boost_round,
                     "early_stopping": args.early_stopping, "best_iteration": best,
                     "refit": not args.no_refit}

    if args.no_refit:
        final, rounds = booster, best
    else:
        rounds = int(round(best * len(train) / len(fit)))
        log.info("lgbm: refit on fit + inner (%d rows) with %d trees", len(train), rounds)
        dtrain = lgb.Dataset(train[features], np.log1p(train[TARGET]))
        final = lgb.train(params, dtrain, num_boost_round=rounds)
        params_logged["refit_rounds"] = rounds

    pred_valid = np.expm1(final.predict(valid[features], num_iteration=rounds))
    importance = pd.Series(
        final.feature_importance("gain", iteration=rounds), index=features, name="gain"
    ).sort_values(ascending=False)
    return pred_inner, pred_valid, params_logged, importance


MODELS = {"lgbm": fit_predict_lgbm}


# ---------------------------------------------------------------- helpers

def zero_series(history: pd.DataFrame, days: int) -> pd.DataFrame:
    """store x family with zero sales over the last `days` days of `history`.

    Some stores simply don't sell some families (or stopped). Forecasting
    exactly 0 for them is safer than a small positive guess.
    """
    end = history["date"].max()
    recent = history.loc[history["date"] > end - pd.Timedelta(days=days)]
    total = recent.groupby(KEYS, observed=True)[TARGET].sum()
    return total[total == 0].reset_index()[KEYS]


def apply_zero_rule(frame: pd.DataFrame, pred: np.ndarray, zeros: pd.DataFrame) -> np.ndarray:
    is_zero = frame[KEYS].merge(zeros.assign(_z=True), on=KEYS, how="left")["_z"].fillna(False)
    out = np.clip(pred, 0.0, None)
    out[is_zero.to_numpy(bool)] = 0.0
    return out


def raw_data_md5() -> str:
    meta = yaml.safe_load((REPO_ROOT / "data" / "raw.dvc").read_text())
    return meta["outs"][0]["md5"]


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="lgbm")
    parser.add_argument("--processed-dir", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "bereket")
    parser.add_argument("--train-start", type=pd.Timestamp, default=pd.Timestamp("2015-01-01"),
                        help="drop older rows from fitting (older years differ a lot)")
    parser.add_argument("--zero-days", type=int, default=56,
                        help="predict 0 for series with no sales in the last N training days")
    parser.add_argument("--learning-rate", type=float, default=LGBM_PARAMS["learning_rate"])
    parser.add_argument("--num-leaves", type=int, default=LGBM_PARAMS["num_leaves"])
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--no-refit", action="store_true",
                        help="score validation with the early-stopped model instead of refitting on all of train")
    parser.add_argument("--no-mlflow", action="store_true", help="don't log the run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    if not args.no_mlflow:
        load_dotenv(REPO_ROOT / ".env")
        if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
            raise SystemExit("MLFLOW_TRACKING_PASSWORD is empty - put your DagsHub token in .env")
        if git("status", "--porcelain", "--untracked-files=no"):
            log.warning("uncommitted changes - the git_sha tag will not match the code that ran")

    start = time.perf_counter()
    df, features = build_features(load_prepared(args.processed_dir))
    train, valid = split_train_valid(df)
    train = train.loc[train["date"] >= args.train_start]
    fit = train.loc[train["date"] < INNER_START]
    inner = train.loc[train["date"] >= INNER_START]
    log.info(
        "fit %s .. %s (%d rows) | inner %s .. %s (%d) | valid %s .. %s (%d)",
        fit["date"].min().date(), fit["date"].max().date(), len(fit),
        inner["date"].min().date(), inner["date"].max().date(), len(inner),
        VALID_START.date(), VALID_END.date(), len(valid),
    )

    pred_inner, pred_valid, model_params, importance = MODELS[args.model](
        fit, inner, train, valid, features, args
    )

    # zero rule: decided from data before each window only
    pred_inner = apply_zero_rule(inner, pred_inner, zero_series(fit, args.zero_days))
    zeros = zero_series(train, args.zero_days)
    pred_valid = apply_zero_rule(valid, pred_valid, zeros)
    log.info("zero rule: %d series predicted 0 (no sales in last %d days)", len(zeros), args.zero_days)

    score_inner = rmsle(inner[TARGET], pred_inner)
    score = rmsle(valid[TARGET], pred_valid)
    minutes = (time.perf_counter() - start) / 60
    log.info("RMSLE inner holdout: %.5f", score_inner)
    log.info("RMSLE on validation (%s .. %s): %.5f   [%.1f min]",
             VALID_START.date(), VALID_END.date(), score, minutes)
    log.info("top features (gain): %s", ", ".join(importance.head(10).index))

    if args.no_mlflow:
        log.info("--no-mlflow: nothing logged")
        return

    import mlflow

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{args.model}-features-v2") as run:
        mlflow.set_tags({
            "author": "bereket",
            "model": args.model,
            "git_sha": git("rev-parse", "HEAD"),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        })
        mlflow.log_params({
            **model_params,
            "train_start": args.train_start.date().isoformat(),
            "inner_start": INNER_START.date().isoformat(),
            "train_end": TRAIN_END.date().isoformat(),
            "valid_start": VALID_START.date().isoformat(),
            "valid_end": VALID_END.date().isoformat(),
            "zero_days": args.zero_days,
            "n_features": len(features),
            "raw_data_md5": raw_data_md5(),
        })
        mlflow.log_metrics({
            "rmsle_val": score,
            "rmsle_inner": score_inner,
            "n_zero_series": len(zeros),
            "minutes": minutes,
        })
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "features.json").write_text(json.dumps(features, indent=2))
            importance.to_csv(Path(tmp) / "feature_importance.csv")
            mlflow.log_artifacts(tmp)
        log.info("MLflow run: %s/#/experiments/%s/runs/%s",
                 mlflow.get_tracking_uri(), run.info.experiment_id, run.info.run_id)


if __name__ == "__main__":
    main()
