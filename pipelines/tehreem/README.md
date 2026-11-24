# tehreem - pipelines

From the repository root:

```bash
dvc repro pipelines/tehreem/dvc.yaml
dvc push pipelines/tehreem/dvc.yaml
```

Stages: `prepare` -> `train` -> `submit`, with `monitor` using the prepared
inputs and validation metrics. Parameters live in `params.yaml`; DVC generates
`dvc.lock`. Prepared data is narrow, and lag/rolling features stay in memory.

See [the full command guide](../../src/tehreem/NEXT_STEPS.md).
