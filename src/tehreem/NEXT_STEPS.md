# Tehreem: commands for the trained-model experiment

Run commands from the repository root. Your working branch is
`exp/tehreem/lgbm`; the earlier weekday baseline is preserved.

## 1. Activate the environment and check your branch

```bash
cd /home/tehreem-kamal/Store_Sales_MLOps/store-sales-mlops
source .venv/bin/activate
git branch --show-current
```

The branch should be `exp/tehreem/lgbm`. Install the member dependencies if
setting up another environment:

```bash
python -m pip install -r requirements.txt -r docker/tehreem/requirements.txt
```

## 2. Check the evaluation and leakage rules

```bash
python -m pytest -q tests src/tehreem/tests
```

The additional tests check calendar gaps, separation of store/family series,
row order, rejection of unsafe lags, and unchanged forecast features when all
16 days of forecast sales are hidden.

## 3. Run the full workflow

```bash
dvc repro pipelines/tehreem/dvc.yaml
```

The four stages are:

1. `prepare`: joins stores, oil, and holidays without changing the raw rows.
2. `train`: builds 31 features in memory, selects the tree count using the
   inner holdout, refits on the outer training period, and scores validation.
3. `submit`: refits through August 15 and produces the 16-day test forecast.
4. `monitor`: compares recent training and validation input distributions.

The initial fit starts in 2016. Early stopping uses July 15–30, 2017; the
shared validation period remains July 31–August 15. The target is
`log1p(sales)`, with predictions transformed back to sales and clipped at zero.
Series with no sales in the most recent 56 training days predict zero.

Lags use a complete daily calendar. The shortest lag and rolling shift are
16 days; same-weekday averages are shifted three weeks. Same-day transactions
are not used. The wide feature matrix is never saved to disk.

Running the same DVC command again skips stages whose inputs, parameters, and
outputs are unchanged. To run just the preparation/training part:

```bash
dvc repro pipelines/tehreem/dvc.yaml:train
```

The underlying Python commands, useful for understanding each stage, are:

```bash
python -m src.tehreem.prepare
python -m src.tehreem.train
python -m src.tehreem.train --submit
python -m monitoring.tehreem.drift_report
```

Use the DVC command for further tracked experiments so it updates your lock
file automatically.

## 4. Read your results and log the saved model

```bash
python -m json.tool reports/tehreem/metrics.json
python -m src.tehreem.check_artifacts
python -m src.tehreem.train --log-existing
```

The check reloads the model, rebuilds its features, and verifies the saved
predictions, RMSLE, and submission IDs. The final command uploads the saved validation model, predictions, metrics,
parameters, feature importance, and source snapshot to `store-sales-tehreem`.
It does not train again. A source/configuration mismatch asks you to reproduce
the pipeline before logging.

Open your experiment:
https://dagshub.com/bereketweld87/store-sales-mlops.mlflow/#/experiments/2

Compare `rmsle_val` with your 56-day weekday baseline, **0.52063**. Lower is
better. The saved LightGBM model is `models/tehreem/validation_model.txt`; its
feature list and training provenance are in `models/tehreem/model_info.json`.
Runs before a code commit are marked `git_dirty=true`; the source hash and
snapshot preserve the exact implementation. Logging also records the current
`logging_git_sha`.

## 5. Make a second trained-model experiment

Edit only `pipelines/tehreem/params.yaml`. For example, change `num_leaves`
from 63 to 127 while keeping the other settings fixed, then run:

```bash
dvc repro pipelines/tehreem/dvc.yaml
python -m src.tehreem.train --log-existing
```

Record both improvements and regressions in your report. Keep the common
validation dates and metric unchanged. Later experiments can try a different
learning rate, training start date, or history window; adjust one setting at a
time so the comparison is understandable.

## 6. Inspect the submission and drift report

- `data/processed/tehreem/submission.csv`: `id,sales` for all 28,512 test rows.
- `reports/tehreem/submission_info.json`: final fit dates and provenance.
- `reports/tehreem/drift_report.json`: input-distribution comparisons.
- `reports/tehreem/drift.png`: a plot of numerical input drift.

The drift report is a batch comparison, not a live dashboard. A distribution
change alone does not show that accuracy declined. The Kaggle leaderboard
score is only known after an actual submission; record it separately from
validation RMSLE. Coordinate with the team about which CSV to submit to the
[competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).

## 7. Use the Docker environment

```bash
docker compose -f docker/tehreem/docker-compose.yml build
docker compose -f docker/tehreem/docker-compose.yml up --abort-on-container-exit --exit-code-from train
```

The container prepares data and trains the validation model. It uses four
CPUs, up to 6 GiB of memory, and your existing raw data. Its build context
contains only its Dockerfile and requirements; `.env` is supplied at runtime.
The repository is mounted read-only, with your processed-data/model/report
folders writable. Output files remain on your computer.

For a host user whose ID is not 1000, set these before using Compose:

```bash
export TEHREEM_UID=$(id -u)
export TEHREEM_GID=$(id -g)
```

To log existing artifacts from the container:

```bash
docker compose -f docker/tehreem/docker-compose.yml run --rm train python -m src.tehreem.train --log-existing
```

## 8. Push your DVC artifacts, then your code and lock file

After reviewing the score and outputs:

```bash
dvc push pipelines/tehreem/dvc.yaml
git branch --show-current
git add src/tehreem pipelines/tehreem docker/tehreem monitoring/tehreem
git diff --cached --stat
git commit -m "feat(tehreem): add LightGBM, DVC pipeline, Docker, and drift report"
git push -u origin exp/tehreem/lgbm
```

Open a pull request into `main`. Include your validation RMSLE, feature
summary, and the tests/pipeline checks you ran. Use **Create a merge commit**.
If the earlier baseline pull request is still open, merge it before merging
the LightGBM pull request.
Keep `.env` and `.dvc/config.local` local; do not use `git add -A`, re-add raw
data, or hand-edit `dvc.lock`.

For the next independent experiment after this PR merges:

```bash
git checkout main
git pull --ff-only
git checkout -b exp/tehreem/next-experiment
```
