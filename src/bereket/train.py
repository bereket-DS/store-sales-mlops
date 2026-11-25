"""Train a model on the in-memory features, score it on the frozen split, log to MLflow.

One script for all my models - pick one with --model. Every model gets the
same data, the same split (src/common/evaluate.py) and the same metric, so
the scores in MLflow are comparable.

    python -m src.bereket.train --model lgbm
    python -m src.bereket.train --model ridge
    python -m src.bereket.train --model ets
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
    return pred_inner, pred_valid, params_logged, importance, final


# Ridge: a linear model needs what trees don't - numbers for categories,
# no missing values, one scale for all columns. All of that lives inside one
# sklearn Pipeline, so the registered model takes the same raw feature frame
# as LightGBM does.
RIDGE_ONE_HOT_EXTRA = ["day_of_week", "month"]          # ints, but not "more is more"
RIDGE_LOG_PREFIXES = ("sales_lag_", "sales_roll_", "transactions_lag_")
RIDGE_LOG_EXTRA = ["onpromotion", "promo_roll_mean_14", "promo_lead_mean_7"]
# A linear model gives "August" one weight for every family. School supplies
# jump in August (back to school) and nothing else does - so each family
# gets its own month and weekday columns.
RIDGE_CROSSES = [("family", "month"), ("family", "day_of_week")]


def ridge_crosses(X: pd.DataFrame) -> pd.DataFrame:
    """family x month, family x day_of_week as text categories, for one-hot."""
    return pd.DataFrame({f"{a}_x_{b}": X[a].astype(str) + "|" + X[b].astype(str)
                         for a, b in RIDGE_CROSSES}, index=X.index)


def ridge_cross_names(transformer, input_features):
    return np.array([f"{a}_x_{b}" for a, b in RIDGE_CROSSES], dtype=object)


def ridge_pipeline(features: list[str], alpha: float):
    """one-hot categoricals + crosses | log1p skewed counts | impute | scale -> Ridge.

    log1p because the target is log1p(sales): a linear model on log sales
    should see log history, not raw history. Missing lags (the first days
    of a series) get the median plus a 0/1 "was missing" column.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    from src.bereket.features import CATEGORICAL

    one_hot = [c for c in CATEGORICAL + RIDGE_ONE_HOT_EXTRA if c in features]
    crossed = sorted({c for pair in RIDGE_CROSSES for c in pair})
    logged = [c for c in features if c.startswith(RIDGE_LOG_PREFIXES) or c in RIDGE_LOG_EXTRA]
    plain = [c for c in features if c not in one_hot and c not in logged]

    def numeric(*steps):
        return make_pipeline(*steps, SimpleImputer(strategy="median", add_indicator=True),
                             StandardScaler())

    def onehot():
        return OneHotEncoder(handle_unknown="ignore", dtype=np.float32)

    prep = ColumnTransformer(
        [
            ("onehot", onehot(), one_hot),
            ("cross", make_pipeline(FunctionTransformer(ridge_crosses, feature_names_out=ridge_cross_names), onehot()), crossed),
            ("log", numeric(FunctionTransformer(np.log1p, feature_names_out="one-to-one")), logged),
            ("num", numeric(), plain),
        ],
        sparse_threshold=1.0,       # sparse: ~900 columns, mostly one-hot zeros
    )
    return Pipeline([("prep", prep), ("ridge", Ridge(alpha=alpha))])


def fit_predict_ridge(fit, inner, train, valid, features, args):
    """Pick alpha on the inner holdout, then (default) refit on all of train."""
    y_fit = np.log1p(fit[TARGET])
    best_alpha, best_score, best_model = None, np.inf, None
    for alpha in args.alphas:
        model = ridge_pipeline(features, alpha).fit(fit[features], y_fit)
        score = rmsle(inner[TARGET], np.expm1(model.predict(inner[features])).clip(0))
        log.info("ridge: alpha %-8g inner RMSLE %.5f", alpha, score)
        if score < best_score:
            best_alpha, best_score, best_model = alpha, score, model
    log.info("ridge: best alpha %g (inner holdout)", best_alpha)

    pred_inner = np.expm1(best_model.predict(inner[features]))
    if args.no_refit:
        final = best_model
    else:
        log.info("ridge: refit on fit + inner (%d rows)", len(train))
        final = ridge_pipeline(features, best_alpha).fit(train[features], np.log1p(train[TARGET]))
    pred_valid = np.expm1(final.predict(valid[features]))

    # "importance" for a linear model: size of the coefficient on the scaled
    # inputs (one-hot columns are 0/1, the rest have std 1, so they compare).
    names = final.named_steps["prep"].get_feature_names_out()
    importance = pd.Series(np.abs(final.named_steps["ridge"].coef_), index=names,
                           name="abs_coef").sort_values(ascending=False)
    params = {"alpha": best_alpha, "alphas_tried": ",".join(map(str, args.alphas)),
              "refit": not args.no_refit, "n_inputs_after_prep": len(names)}
    return pred_inner, pred_valid, params, importance, final


# ETS (Holt-Winters): one small model per store x family, on that series'
# own past only - no features, no promotions, no holidays. A different kind
# of model from the two above, which is what an ensemble needs.
ETS_SEASON = 7          # weekly pattern
ETS_MIN_DAYS = 4 * ETS_SEASON


def ets_history(history: pd.DataFrame, days: int) -> pd.DataFrame:
    """date x (store_nbr, family) table of log1p(sales), one row per day.

    Holt-Winters needs an unbroken daily series. Dec 25 (stores closed, no
    row) is filled with the same weekday one week earlier.
    """
    start = history["date"].max() - pd.Timedelta(days=days - 1)
    recent = history.loc[history["date"] >= start, ["date", *KEYS, TARGET]]
    recent = recent.astype({"store_nbr": "int64", "family": "str"})
    wide = recent.pivot(index="date", columns=KEYS, values=TARGET)
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    wide = wide.fillna(wide.shift(ETS_SEASON)).fillna(wide.shift(-ETS_SEASON)).fillna(0.0)
    return np.log1p(wide)


def ets_forecast_one(y: np.ndarray, steps: int, trend: str | None) -> tuple[np.ndarray, bool]:
    """Holt-Winters forecast for one series; (forecast, used_fallback).

    Leading zeros (before a store sold this family) are cut off. Series that
    are too short, flat, or fail to fit get the mean of their last 4 weeks.
    """
    import warnings

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    nonzero = np.flatnonzero(y > 0)
    if len(nonzero) == 0:
        return np.zeros(steps), False
    y = y[nonzero[0]:]
    fallback = np.full(steps, y[-ETS_MIN_DAYS:].mean())
    if len(y) < ETS_MIN_DAYS or np.ptp(y[-ETS_MIN_DAYS:]) == 0:
        return fallback, True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    # convergence chatter, 1782 times
            fitted = ExponentialSmoothing(
                y, trend=trend, damped_trend=trend is not None, seasonal="add",
                seasonal_periods=ETS_SEASON, initialization_method="estimated",
            ).fit()
        forecast = fitted.forecast(steps)
    except (ValueError, np.linalg.LinAlgError):
        return fallback, True
    if not np.isfinite(forecast).all():
        return fallback, True
    return forecast, False


def ets_predict(history: pd.DataFrame, target: pd.DataFrame, args) -> tuple[np.ndarray, int]:
    """Forecast every series from the end of `history` to the rows of `target`.

    Returns (sales predictions aligned with `target`, number of fallbacks).
    """
    from joblib import Parallel, delayed, parallel_config

    wide = ets_history(history, args.ets_days)
    dates = pd.date_range(wide.index.max() + pd.Timedelta(days=1), target["date"].max(), freq="D")
    trend = None if args.ets_trend == "none" else args.ets_trend
    with parallel_config(backend="loky", inner_max_num_threads=1):   # 1 BLAS thread per worker
        results = Parallel(n_jobs=-1, batch_size=16)(
            delayed(ets_forecast_one)(wide[col].to_numpy(), len(dates), trend) for col in wide.columns
        )
    series = wide.columns.to_frame(index=False)
    long = pd.DataFrame({
        "date": np.tile(dates, len(series)),
        "store_nbr": np.repeat(series["store_nbr"].to_numpy(), len(dates)),
        "family": np.repeat(series["family"].to_numpy(), len(dates)),
        "pred": np.concatenate([forecast for forecast, _ in results]),
    })
    keys = target[["date", *KEYS]].astype({"store_nbr": "int64", "family": "str"})
    pred = keys.merge(long, on=["date", *KEYS], how="left")["pred"]
    if pred.isna().any():
        log.warning("ets: %d target rows have no history - predicting 0", int(pred.isna().sum()))
    return np.expm1(pred.fillna(0.0).to_numpy()), sum(used for _, used in results)


def fit_predict_ets(fit, inner, train, valid, features, args):
    """Inner: forecast from the end of `fit`. Validation: from the end of all
    of train. ETS always "refits" - a forecast starts where its data ends -
    so --no-refit does not apply. There is no single model object to save
    (1782 small ones), so nothing is registered."""
    pred_inner, fallback_inner = ets_predict(fit, inner, args)
    log.info("ets: inner done (%d fallback series)", fallback_inner)
    pred_valid, fallback_valid = ets_predict(train, valid, args)
    log.info("ets: validation done (%d fallback series)", fallback_valid)
    params = {"ets_trend": args.ets_trend, "ets_damped": args.ets_trend != "none",
              "ets_seasonal": "add", "ets_season": ETS_SEASON, "ets_days": args.ets_days,
              "ets_target": "log1p", "n_fallback_valid": fallback_valid}
    return pred_inner, pred_valid, params, None, None


# Every fit_predict_<name>(fit, inner, train, valid, features, args) returns
# (pred_inner, pred_valid, params, importance or None, final model or None).
MODELS = {"lgbm": fit_predict_lgbm, "ridge": fit_predict_ridge, "ets": fit_predict_ets}


# ---------------------------------------------------------------- helpers

def registered_name(model: str) -> str:
    """One registry entry per model type; each run adds a new version."""
    return f"store-sales-bereket-{model}"


def log_and_register(model_obj, model: str, input_example: pd.DataFrame) -> None:
    """Save the final model in the run and register it, using the MLflow
    flavor that matches the model's library. Add a branch here for each new
    model family (e.g. statsmodels for ETS)."""
    import mlflow

    kwargs = {"name": "model", "registered_model_name": registered_name(model),
              "input_example": input_example}
    module = type(model_obj).__module__
    if module.startswith("lightgbm"):
        mlflow.lightgbm.log_model(model_obj, **kwargs)
    elif module.startswith("sklearn"):
        mlflow.sklearn.log_model(model_obj, **kwargs)
    else:
        raise TypeError(f"no MLflow flavor wired up for {type(model_obj)} - add one in log_and_register()")
    log.info("registered model '%s'", registered_name(model))


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
    parser.add_argument("--alphas", type=lambda s: [float(a) for a in s.split(",")],
                        default=[1e3, 3e3, 1e4, 3e4, 1e5, 3e5],
                        help="ridge: comma-separated alphas to try on the inner holdout")
    parser.add_argument("--ets-days", type=int, default=728,
                        help="ets: fit each series on its last N days")
    parser.add_argument("--ets-trend", choices=["add", "none"], default="add",
                        help="ets: damped additive trend, or no trend")
    parser.add_argument("--no-refit", action="store_true",
                        help="score validation with the early-stopped model instead of refitting on all of train")
    parser.add_argument("--no-mlflow", action="store_true", help="don't log the run")
    parser.add_argument("--no-register", action="store_true",
                        help="log the run but don't save/register the model")
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

    pred_inner, pred_valid, model_params, importance, final_model = MODELS[args.model](
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
    if importance is not None:
        log.info("top features (%s): %s", importance.name, ", ".join(importance.head(10).index))

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
            if importance is not None:
                importance.to_csv(Path(tmp) / "feature_importance.csv")
            mlflow.log_artifacts(tmp)
        if args.no_register:
            log.info("--no-register: model not saved/registered")
        elif final_model is None:
            log.info("%s has no single model object to save - nothing registered", args.model)
        else:
            log_and_register(final_model, args.model, valid[features].head(5))
        log.info("MLflow run: %s/#/experiments/%s/runs/%s",
                 mlflow.get_tracking_uri(), run.info.experiment_id, run.info.run_id)


if __name__ == "__main__":
    main()
