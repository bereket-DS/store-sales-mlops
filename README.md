# Store Sales - Time Series Forecasting (MLOps practical)

Three-member project on the Kaggle competition
[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting):
daily sales for 33 product families across 54 Ecuadorian stores, scored with
**RMSLE** over a 16-day horizon (2017-08-16 .. 2017-08-31).

Members: **Bereket**, **Fanuel**, **Tehreem**.

---

## What is shared, and what is not

**Shared - exactly two things:**

1. `data/raw/` - the competition data, tracked once with DVC. Everyone pulls
   identical bytes; nobody re-downloads from Kaggle.
2. `src/common/evaluate.py` - the frozen evaluation contract.

**Not shared - everything else.** Each member writes their own data
preparation, features, model, DVC pipeline, Dockerfile, compose file and
monitoring setup, in their own folders, however they like.

```
src/common/evaluate.py      <- shared. RMSLE + the train/valid split.
data/raw/                   <- shared. DVC-tracked once.

src/<member>/               <- your code
pipelines/<member>/         <- your dvc.yaml + params.yaml
docker/<member>/            <- your Dockerfile + compose
monitoring/<member>/        <- your prometheus/grafana/drift setup
data/processed/<member>/     models/<member>/     reports/<member>/
notebooks/<member>/
```

The empty folders are tracked with `.gitkeep`, so the layout is in the repo
from the first clone and nobody has to guess where things go.

## The one rule that matters

`src/common/evaluate.py` fixes the metric and the split:

```
train  <= 2017-07-30
valid     2017-07-31 .. 2017-08-15   (16 days, mirrors the real test window)
test      2017-08-16 .. 2017-08-31
```

Three different data pipelines only produce **comparable** scores while the
RMSLE function and the validation window are identical for everyone. Import
them, do not reimplement them:

```python
from src.common.evaluate import rmsle, split_train_valid, check_lags

train, valid = split_train_valid(prepared)     # your prepared data
score = rmsle(valid["sales"], predictions)
```

All three members are CODEOWNERS of that file, so changing it needs
everyone's approval. Treat it as frozen after week one. `pytest` covers it.

`check_lags()` is there because a lag shorter than 16 days is the easiest way
to cheat yourself on this competition: it uses data that will not exist at
prediction time, so validation looks great and the leaderboard does not.

---

## Setup

### Once, by one member

```bash
git clone https://github.com/<ORG>/<REPO>.git && cd <REPO>

dvc init
bash scripts/setup_dvc_remote.sh <DAGSHUB_OWNER> <REPO>

bash scripts/download_data.sh
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc && git commit -m "track raw competition data"
dvc push && git push
```

On DagsHub: **Create -> New Repository -> Connect a repository**, point it at
the GitHub repo. The green **Remote** button shows the exact snippets for
your repo.

> Make the repo **public**. DagsHub's free plan limits collaborators on
> private projects, which is awkward for three people, and the competition
> data is public anyway. Check the course does not require otherwise.

### Once, by each member

```bash
git clone https://github.com/<ORG>/<REPO>.git && cd <REPO>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# then install whatever YOUR approach needs: lightgbm, statsmodels, torch, ...

cp .env.example .env
bash scripts/setup_dvc_remote.sh <DAGSHUB_OWNER> <REPO>

dvc pull        # identical raw data, no Kaggle account needed
dvc install     # git hook: auto `dvc checkout` when you switch branches
pytest          # the shared contract should be green before you start
```

`dvc install` is worth it. Switching branches changes `dvc.lock` but not the
files on disk; forget `dvc checkout` once and you train on stale data.

### Fill in

- `.github/CODEOWNERS` - the three GitHub handles
- `.env.example` - `<DAGSHUB_OWNER>`, `<REPO>`

---

## Daily workflow

```bash
git checkout main && git pull
git checkout -b exp/<you>/<what-you-are-trying>

# work in your own folders
dvc repro pipelines/<you>/dvc.yaml
dvc push

git add -A && git commit -m "exp: ..., RMSLE 0.4xx"
git push -u origin exp/<you>/<what>
# PR into main, one approval from either teammate
```

### Rules

1. **Only touch your own folders.** `src/common/` needs all three approvals.
2. **One `dvc.yaml` per member**, in your own `pipelines/<you>/`. Separate
   files mean separate `dvc.lock` files, which is why three people can commit
   to `main` without merge conflicts. Never put one at the repo root.
3. **Branch per experiment**, 1-4 days, then merge - including experiments
   that made things worse. A recorded failure is report material.
4. **Never squash-merge.** It collapses your commits into one authored by
   whoever merged, which erases individual attribution. Turn it off in
   Settings -> General -> Pull Requests.
5. **Never hand-edit `dvc.lock`.** On a conflict, take `main`'s version and
   re-run `dvc repro`.
6. Check `git config user.email` matches your GitHub account before your first
   commit, or your commits will not link to your profile.

---

## Processed data

`dvc push` sends your processed data to the shared remote, but a bare
`dvc pull` resolves against **your** `dvc.lock`, so you only get your own:

```bash
dvc pull                                      # raw + your own
dvc pull data/processed/fanuel/train.parquet  # only if you want it
```

Shared storage, separate data. Worth pushing: your teacher reproduces from
it, a final ensemble needs all three, and it makes "why do our scores differ"
answerable in a minute.

Keep lag and rolling features **in memory** in your training script rather
than writing them to disk. Persisting a wide feature matrix on every
parameter change is what fills a storage quota.

---

## MLflow

Each member logs to their own experiment - `store-sales-<member>` - so your
runs page is unambiguously yours. Tag runs with your name and the git SHA so
a number can always be traced back to a commit:

```python
mlflow.set_experiment("store-sales-bereket")
with mlflow.start_run(run_name="lgbm-lag28"):
    mlflow.set_tag("author", "bereket")
    mlflow.log_metric("rmsle_val", score)
```

Keeping the `author` tag means the three experiments can still be pulled into
one table for the final report:

```python
mlflow.search_runs(
    experiment_names=["store-sales-bereket", "store-sales-fanuel", "store-sales-tehreem"],
    order_by=["metrics.rmsle_val ASC"],
)
```

---

## Final checklist

- [ ] `main` has all three members' work and reproduces (`dvc repro --all-pipelines`)
- [ ] `dvc dag` shows three independent branches off shared `data/raw`
- [ ] Three MLflow experiments, each with its owner's runs
- [ ] Commit history shows sustained individual work from all three
- [ ] Each member's Docker stack starts with one command
- [ ] Each member's monitoring and drift report done
- [ ] Kaggle submission made; leaderboard score recorded honestly
- [ ] Report compares the three approaches on the frozen validation split
