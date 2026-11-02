#!/usr/bin/env bash
# Run ONCE, by one member. Everyone else uses `dvc pull`.
set -euo pipefail

COMP="store-sales-time-series-forecasting"
DEST="data/raw"

mkdir -p "$DEST"

if ! command -v kaggle >/dev/null; then
  echo "kaggle CLI not found. pip install kaggle, then put your API token in ~/.kaggle/kaggle.json"
  exit 1
fi

kaggle competitions download -c "$COMP" -p "$DEST"
unzip -o "$DEST"/*.zip -d "$DEST"
rm -f "$DEST"/*.zip

ls -lh "$DEST"
echo
echo "Now track it once for the team:"
echo "  dvc add data/raw"
echo "  git add data/raw.dvc data/.gitignore && git commit -m 'track raw competition data'"
echo "  dvc push && git push"
