"""Reload the saved model and verify its predictions and submission IDs."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.evaluate import rmsle, split_train_valid
from src.tehreem.features import build_features
from src.tehreem.train import predict_sales
from src.tehreem.utils import MODELS, PREPARED, PROCESSED, REPORTS, ROOT, provenance


def main():
    info = json.loads((MODELS / 'model_info.json').read_text())
    metrics = json.loads((REPORTS / 'metrics.json').read_text())
    if info['source_sha256'] != provenance(ROOT / info['params_file'])['source_sha256']:
        raise ValueError('Model source/configuration changed: reproduce the pipeline first')
    frame, features = build_features(pd.read_parquet(PREPARED), info['feature_config'])
    if features != info['features']:
        raise ValueError('The model feature list does not match the current feature builder')
    train, valid = split_train_valid(frame)
    train = train.loc[train['date'] >= pd.Timestamp(info['train_start'])]
    model = lgb.Booster(model_file=str(MODELS / 'validation_model.txt'))
    if model.feature_name() != features:
        raise ValueError('Saved model feature names do not match the feature list')
    model.params['num_threads'] = int(info['training_config']['lightgbm']['num_threads'])
    predictions = predict_sales(model, valid, features, train, int(info['training_config']['zero_history_days']))
    saved = pd.read_parquet(PROCESSED / 'lgbm_valid_predictions.parquet')
    np.testing.assert_array_equal(valid['id'], saved['id'])
    np.testing.assert_allclose(predictions, saved['prediction'], rtol=1e-10, atol=1e-10)
    score = rmsle(valid['sales'], predictions)
    np.testing.assert_allclose(score, metrics['rmsle_val'], rtol=1e-10, atol=1e-10)
    submission = pd.read_csv(PROCESSED / 'submission.csv')
    expected = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=['id']).sort_values('id')
    np.testing.assert_array_equal(submission['id'], expected['id'])
    if not submission['id'].is_unique or not np.isfinite(submission['sales']).all() or (submission['sales'] < 0).any():
        raise ValueError('Submission predictions must be unique, finite, and nonnegative')
    print(f'Saved-model reload verified: RMSLE {score:.5f}; {len(valid):,} validation predictions match.', flush=True)
    print(f'Submission verified: {len(submission):,} exact test IDs and finite nonnegative sales.', flush=True)


if __name__ == '__main__':
    main()
