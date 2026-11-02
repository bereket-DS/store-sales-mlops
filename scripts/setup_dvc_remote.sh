#!/usr/bin/env bash
# Each member runs this once, with their OWN DagsHub token.
# Credentials go to .dvc/config.local, which is gitignored.
set -euo pipefail

DAGSHUB_OWNER="${1:-<DAGSHUB_OWNER>}"
REPO="${2:-<REPO>}"

read -rsp "DagsHub token: " TOKEN
echo

if ! dvc remote list | grep -q '^origin'; then
  dvc remote add -d origin "s3://dvc"
  dvc remote modify origin endpointurl "https://dagshub.com/${DAGSHUB_OWNER}/${REPO}.s3"
fi

dvc remote modify origin --local access_key_id     "$TOKEN"
dvc remote modify origin --local secret_access_key "$TOKEN"

echo "Done. Check with: dvc remote list && dvc status --cloud"
echo "If this fails, copy the exact snippet from the green 'Remote' button on DagsHub."
