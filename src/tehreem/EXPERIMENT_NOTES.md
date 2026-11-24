# Tehreem: experiment notes

## Validation results

All scores below use the shared validation window, **2017-07-31 through
2017-08-15**, and the shared RMSLE implementation. Lower is better.

| My approach | Validation RMSLE |
|---|---:|
| Same-weekday mean, last 28 training days | 0.53113 |
| Same-weekday mean, last 56 training days | 0.52063 |
| LightGBM with 31 features and log1p target | **0.42433** |

The first LightGBM run scored 0.4243349024542828, an approximately **18.5%**
RMSLE reduction from the rounded 56-day baseline reference. It is recorded in
[MLflow](https://dagshub.com/bereketweld87/store-sales-mlops.mlflow/#/experiments/2/runs/918504011612480b9bbd299ff76102b9).
The metrics for the latest reproduced run are in
`reports/tehreem/metrics.json`.
The [final reproduced model and source snapshot](https://dagshub.com/bereketweld87/store-sales-mlops.mlflow/#/experiments/2/runs/debaeae13eb84a4a89d4bf59f1f9fd6a)
are logged separately after the provenance checks were finalized.

## Model and validation procedure

- Label training begins on 2016-01-01. Earlier raw history supplies lag inputs.
- July 15–30 is an inner holdout used to select the tree count.
- The best tree count within the configured budget was 1,000. The run reached
  the maximum budget; the early-stopping patience was not exhausted.
- The model is refitted through July 30 before scoring the untouched outer
  validation window. That refit contains 1,026,432 rows.
- LightGBM learns `log1p(sales)`; predictions use `expm1` and are clipped at zero.
- An inactive-series rule predicts zero when the last 56 training days have
  no sales for that store/family.
- Final submission fitting uses the same selected tree count and all label
  rows from 2016-01-01 through 2017-08-15. The CSV contains all 28,512 test IDs.

Features include store/family/location categories, promotions, forward-filled
oil prices, locale-aware holiday flags, calendar indicators, sales lags of
16/21/28/35/56/364 calendar days, rolling statistics shifted 16 days, and a
same-weekday mean shifted three weeks. Calendar gaps are filled before lagging;
the validation-label hiding test checks that future sales cannot affect the
forecast features.

The five highest gain features in the first fit were `sales_mean_7`,
`sales_weekday_mean`, `sales_mean_28`, `sales_mean_56`, and `onpromotion`.
The full ranking is in `reports/tehreem/feature_importance.csv`.

## Team comparison references

These teammate results are reported in their project pull requests; they have
not been rerun in Tehreem's environment.

| Member and approach | Reported validation RMSLE | Source |
|---|---:|---|
| Bereket, mean of last 16 days | 0.52238 | [PR 6 comparison](https://github.com/bereket-DS/store-sales-mlops/pull/6) |
| Bereket, LightGBM | 0.42519 | [PR 6](https://github.com/bereket-DS/store-sales-mlops/pull/6) |
| Fanuel, safe-history LightGBM | 0.44685 | [PR 7](https://github.com/bereket-DS/store-sales-mlops/pull/7) |

These are scores on one fixed validation window, not Kaggle leaderboard
results. Record a leaderboard score only after the team actually submits.

## Further experiments and evidence

A useful next experiment is a larger tree budget because the first run reached
its 1,000-round cap. Change one setting at a time and preserve both good and bad
results in MLflow. Other experiments can compare leaf counts, learning rates,
or training start dates without changing the shared evaluation contract.

Keep the generated DVC lock, MLflow run links, validation metrics, feature
importance, submission metadata, and drift report as evidence for the report.
The command sequence is in [NEXT_STEPS.md](NEXT_STEPS.md).

Verified during implementation: 15 tests passed; the full DVC pipeline
completed; unchanged stages were skipped on the next run; the saved model
reproduced all validation predictions on the host and inside Docker; and the
submission matched all 28,512 test IDs. The Docker image built successfully.
The data push, Git commit/push, pull request, and Kaggle upload remain manual
steps in the command guide.
