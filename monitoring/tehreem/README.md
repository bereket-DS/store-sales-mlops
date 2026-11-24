# tehreem - monitoring

Run the DVC pipeline, or after preparation and validation run:

```bash
python -m monitoring.tehreem.drift_report
```

The batch report compares the most recent 56 training days with the shared
16-day validation window. It reports population stability indices for
promotions, oil prices, and weekday, plus categorical distribution distances
for stores and product families. Results are written to
`reports/tehreem/drift_report.json` and `reports/tehreem/drift.png`.

These describe covariate changes; they do not establish that model accuracy
declined. Validation RMSLE is included separately. See
[the command guide](../../src/tehreem/NEXT_STEPS.md) for the full workflow.
