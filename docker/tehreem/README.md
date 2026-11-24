# tehreem - docker

From the repository root:

```bash
docker compose -f docker/tehreem/docker-compose.yml build
docker compose -f docker/tehreem/docker-compose.yml up --abort-on-container-exit --exit-code-from train
```

The service prepares data and trains the validation model using four CPUs and
up to 6 GiB RAM. It reads raw data from the host and writes only Tehreem's
processed-data, model, and report folders. The build context excludes project
data and credentials; the local `.env` is supplied at runtime.

The default host UID/GID is 1000. If yours differs, export `TEHREEM_UID` and
`TEHREEM_GID` as described in [the command guide](../../src/tehreem/NEXT_STEPS.md).
