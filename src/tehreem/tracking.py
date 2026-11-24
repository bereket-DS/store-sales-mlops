"""Log a saved validation experiment without repeating model training."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
import mlflow

from src.tehreem.utils import MODELS, PROCESSED, REPORTS, ROOT, provenance


def log_validation():
    metrics_path = REPORTS / 'metrics.json'
    info_path = MODELS / 'model_info.json'
    if not metrics_path.exists() or not info_path.exists():
        raise ValueError('Run training or dvc repro before logging existing artifacts')
    metrics = json.loads(metrics_path.read_text())
    info = json.loads(info_path.read_text())
    params_path = ROOT / info['params_file']
    current = provenance(params_path)
    if current['source_sha256'] != info['source_sha256']:
        raise ValueError('Source/configuration changed after training. Run dvc repro again before logging.')
    load_dotenv(ROOT / '.env', override=False)
    for key in ['MLFLOW_TRACKING_URI', 'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD']:
        if not os.getenv(key) or '<' in os.environ[key]:
            raise ValueError(f'Configure {key} in your local .env')
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment('store-sales-tehreem')
    with mlflow.start_run(run_name=f'lgbm-{info["best_iteration"]}trees') as run:
        mlflow.set_tags({key: str(info[key]) for key in [
            'author', 'git_sha', 'git_dirty', 'source_sha256', 'raw_data_md5',
        ]})
        mlflow.set_tag('logging_git_sha', current['git_sha'])
        mlflow.log_params({
            **info['training_config']['lightgbm'], 'model': info['model'],
            'train_start': info['train_start'], 'train_end': info['train_end'],
            'valid_start': info['valid_start'], 'valid_end': info['valid_end'],
            'best_iteration': info['best_iteration'], 'feature_count': len(info['features']),
            'lag_days': ','.join(map(str, info['feature_config']['lags'])),
            **{f'version_{key}': value for key, value in info['versions'].items()},
        })
        mlflow.log_metrics(metrics)
        for path, category in [
            (MODELS / 'validation_model.txt', 'model'), (info_path, 'model'),
            (metrics_path, 'reports'), (REPORTS / 'feature_importance.csv', 'reports'),
            (PROCESSED / 'lgbm_valid_predictions.parquet', 'predictions'),
        ]:
            mlflow.log_artifact(str(path), artifact_path=category)
        for path in sorted((ROOT / 'src/tehreem').glob('*.py')):
            mlflow.log_artifact(str(path), artifact_path='source')
        mlflow.log_artifact(str(params_path), artifact_path='source')
        mlflow.log_artifact(str(ROOT / 'docker/tehreem/requirements.txt'), artifact_path='source')
        mlflow.log_artifact(str(ROOT / 'src/common/evaluate.py'), artifact_path='source/shared')
        print(f'MLflow experiment: store-sales-tehreem; run: {run.info.run_id}', flush=True)
        print(f'Validation RMSLE: {metrics["rmsle_val"]:.5f}', flush=True)
