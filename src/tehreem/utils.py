"""Paths, parameters, and provenance shared only by Tehreem's scripts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / 'data/processed/tehreem/prepared.parquet'
MODELS = ROOT / 'models/tehreem'
REPORTS = ROOT / 'reports/tehreem'
PROCESSED = ROOT / 'data/processed/tehreem'
DEFAULT_PARAMS = ROOT / 'pipelines/tehreem/params.yaml'


def read_params(path: str | Path = DEFAULT_PARAMS) -> dict:
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def provenance(params_path: str | Path = DEFAULT_PARAMS) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ['git', '-c', f'safe.directory={ROOT}', *args],
                cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
                env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unavailable'

    digest = hashlib.sha256()
    sources = [ROOT / 'src/tehreem' / name for name in
               ['prepare.py', 'features.py', 'train.py', 'tracking.py', 'utils.py']]
    sources += [ROOT / 'src/common/evaluate.py', ROOT / 'docker/tehreem/requirements.txt']
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    parameters = read_params(params_path)
    model_parameters = {name: parameters[name] for name in ['features', 'training']}
    digest.update(json.dumps(model_parameters, sort_keys=True).encode())
    return {
        'author': 'tehreem',
        'git_sha': git('rev-parse', 'HEAD'),
        'git_dirty': bool(git('status', '--porcelain')),
        'source_sha256': digest.hexdigest(),
        'raw_data_md5': yaml.safe_load((ROOT / 'data/raw.dvc').read_text())['outs'][0]['md5'],
    }
