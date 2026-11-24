"""Train and validate LightGBM, or refit it to generate a Kaggle submission."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.evaluate import TRAIN_END, VALID_START, VALID_END, rmsle, split_train_valid
from src.tehreem.features import KEYS, build_features
from src.tehreem.utils import (
    DEFAULT_PARAMS, MODELS, PREPARED, PROCESSED, REPORTS, ROOT,
    provenance, read_params, write_json,
)


def predict_sales(model, rows, features, history, zero_days):
    predictions = np.maximum(0, np.expm1(model.predict(rows[features])))
    if zero_days > 0:
        start = history['date'].max() - pd.Timedelta(days=zero_days - 1)
        totals = history.loc[history['date'] >= start].groupby(KEYS, observed=True)['sales'].sum()
        inactive = totals.eq(0).reindex(pd.MultiIndex.from_frame(rows[KEYS]), fill_value=False)
        predictions[inactive.to_numpy()] = 0
    if not np.isfinite(predictions).all():
        raise ValueError('Model produced non-finite predictions')
    return predictions


def dataset(rows, features):
    return lgb.Dataset(rows[features], label=np.log1p(rows['sales'].to_numpy(dtype='float64')))


def validate(frame, features, config, params_path):
    train, valid = split_train_valid(frame)
    settings = config['training']
    train = train.loc[train['date'] >= pd.Timestamp(settings['start_date'])].copy()
    inner_start = TRAIN_END - pd.Timedelta(days=int(settings['inner_days']) - 1)
    fit = train.loc[train['date'] < inner_start]
    inner = train.loc[train['date'] >= inner_start]
    if fit.empty or inner.empty or valid.empty:
        raise ValueError('Training configuration leaves an empty fit, inner, or validation period')
    print(f'Inner fit: {len(fit):,} rows; early stopping: {inner_start.date()} through {TRAIN_END.date()}', flush=True)
    dfit = dataset(fit, features)
    dinner = dataset(inner, features)
    inner_model = lgb.train(
        settings['lightgbm'], dfit, num_boost_round=int(settings['max_rounds']),
        valid_sets=[dinner], valid_names=['inner'],
        callbacks=[lgb.early_stopping(int(settings['early_stopping_rounds'])), lgb.log_evaluation(100)],
    )
    rounds = inner_model.best_iteration or int(settings['max_rounds'])
    inner_predictions = predict_sales(inner_model, inner, features, fit, int(settings['zero_history_days']))
    inner_score = rmsle(inner['sales'], inner_predictions)
    del inner_model, dfit, dinner

    print(f'Refitting {rounds} trees on {len(train):,} rows through {TRAIN_END.date()}', flush=True)
    model = lgb.train(settings['lightgbm'], dataset(train, features), num_boost_round=rounds)
    predictions = predict_sales(model, valid, features, train, int(settings['zero_history_days']))
    score = rmsle(valid['sales'], predictions)
    for directory in [MODELS, REPORTS, PROCESSED]:
        directory.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS / 'validation_model.txt'))
    output = valid[['id', 'date', *KEYS, 'sales']].copy()
    output['prediction'] = predictions
    output.to_parquet(PROCESSED / 'lgbm_valid_predictions.parquet', index=False)
    pd.DataFrame({
        'feature': features, 'gain': model.feature_importance(importance_type='gain'),
    }).sort_values('gain', ascending=False).to_csv(REPORTS / 'feature_importance.csv', index=False)
    metrics = {
        'rmsle_val': score, 'rmsle_inner': inner_score,
        'baseline_reference_rmsle': 0.52063, 'best_iteration': int(rounds),
        'train_rows': len(train), 'valid_rows': len(valid),
    }
    info = {
        **provenance(params_path), 'model': 'lightgbm_log1p', 'features': features,
        'params_file': str(Path(params_path).resolve().relative_to(ROOT)),
        'feature_config': config['features'], 'training_config': settings,
        'train_start': str(train['date'].min().date()), 'train_end': str(TRAIN_END.date()),
        'inner_start': str(inner_start.date()),
        'valid_start': str(VALID_START.date()), 'valid_end': str(VALID_END.date()),
        'best_iteration': int(rounds),
        'versions': {name: version(name) for name in ['lightgbm', 'pandas', 'numpy']},
    }
    write_json(REPORTS / 'metrics.json', metrics)
    write_json(MODELS / 'model_info.json', info)
    print(f'Validation RMSLE: {score:.5f}; 56-day baseline reference: 0.52063', flush=True)
    print('Saved validation model, model metadata, predictions, metrics, and feature importance.', flush=True)


def submit(frame, features, config, params_path):
    info_path = MODELS / 'model_info.json'
    if not info_path.exists():
        raise ValueError('Run validation first to select the tree count using the inner holdout')
    info = json.loads(info_path.read_text())
    if (features != info['features'] or config['features'] != info['feature_config']
            or config['training'] != info['training_config']):
        raise ValueError('Configuration changed; rerun validation before generating a submission')
    train = frame.loc[frame['sales'].notna() & (frame['date'] >= pd.Timestamp(config['training']['start_date']))]
    test = frame.loc[frame['sales'].isna()].sort_values('id')
    if train.empty or test.empty or train['date'].max() >= test['date'].min():
        raise ValueError('Unexpected training/test boundaries')
    print(f'Final fit: {len(train):,} rows through {train["date"].max().date()}, {info["best_iteration"]} trees', flush=True)
    model = lgb.train(config['training']['lightgbm'], dataset(train, features), num_boost_round=info['best_iteration'])
    predictions = predict_sales(model, test, features, train, int(config['training']['zero_history_days']))
    submission = pd.DataFrame({'id': test['id'].to_numpy(), 'sales': predictions})
    expected = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=['id']).sort_values('id')
    if not submission['id'].is_unique or not np.array_equal(submission['id'], expected['id']):
        raise ValueError('Submission IDs do not match raw test IDs')
    model.save_model(str(MODELS / 'submission_model.txt'))
    submission.to_csv(PROCESSED / 'submission.csv', index=False)
    write_json(REPORTS / 'submission_info.json', {
        **provenance(params_path), 'rows': len(submission), 'best_iteration': info['best_iteration'],
        'train_end': str(train['date'].max().date()),
        'test_start': str(test['date'].min().date()), 'test_end': str(test['date'].max().date()),
    })
    print(f'Created {len(submission):,} predictions -> data/processed/tehreem/submission.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', default=str(DEFAULT_PARAMS))
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--log-mlflow', action='store_true', help='Log the validation run after training')
    parser.add_argument('--log-existing', action='store_true', help='Log saved validation artifacts without retraining')
    args = parser.parse_args()
    if args.log_existing:
        if args.submit:
            parser.error('--log-existing logs validation artifacts; use it separately from --submit')
        from src.tehreem.tracking import log_validation
        log_validation()
        return
    if args.submit and args.log_mlflow:
        parser.error('Log the saved validation run separately with --log-existing')
    if not PREPARED.exists():
        parser.error('Run python -m src.tehreem.prepare first')
    config = read_params(args.params)
    print('Building features in memory; no wide feature matrix is saved.', flush=True)
    frame, features = build_features(pd.read_parquet(PREPARED), config['features'])
    print(f'Built {len(features)} features for {len(frame):,} rows.', flush=True)
    if args.submit:
        submit(frame, features, config, args.params)
    else:
        validate(frame, features, config, args.params)
        if args.log_mlflow:
            from src.tehreem.tracking import log_validation
            log_validation()


if __name__ == '__main__':
    main()
