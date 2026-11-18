# tehreem

## First experiment: a recent weekday baseline

`baseline.py` predicts each store/product-family combination using its average
sales on the same weekday over the last 28 days of training. For example, a
Monday prediction uses the four most recent training Mondays. This provides a
simple reference score before trying a more complex model.

If a weekday group is missing, prediction falls back to that store/family's
recent mean, then the overall recent mean if the group has no history.

Run these commands from the repository root:

```bash
source .venv/bin/activate
python -m src.tehreem.baseline
```

The script performs these steps:

1. Reads the shared `data/raw/train.csv`.
2. Uses `split_train_valid` from `src.common.evaluate`: training ends on
   2017-07-30; validation covers 2017-07-31 through 2017-08-15.
3. Computes weekday averages from training rows only. The same fitted averages
   predict the entire validation window, without using validation sales.
4. Scores the predictions using the shared `rmsle`; lower is better.
5. Saves predictions and a metrics report in your own folders:
   - `data/processed/tehreem/weekday_baseline_valid.csv`
   - `reports/tehreem/weekday_baseline_metrics.json`

The first local run with a 28-day window scored **0.53113 validation RMSLE**
on 28,512 validation rows. Treat this as a reference for later experiments.

The generated outputs are ignored by Git under the project rules. A later DVC
pipeline can track them. The metrics report records the raw-data hash, Git
commit, source hash, and whether the working tree contained uncommitted work.

## Track the experiment in MLflow

With your local `.env` configured, run:

```bash
python -m src.tehreem.baseline --log-mlflow
```

This logs to `store-sales-tehreem` on the shared DagsHub MLflow server, including
the validation RMSLE, window length, author, Git/source information, predictions,
metrics report, and source script. Credentials stay in your local `.env`.

## Try a different history window

```bash
python -m src.tehreem.baseline --window-days 56
```

Each run replaces the local prediction and metrics files. Use `--log-mlflow` to
keep a separate tracked run for each experiment. All experiments use the same
shared validation dates and metric.
