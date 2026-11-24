"""Compare recent training covariates with the fixed validation window."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.common.evaluate import TRAIN_END, VALID_START, VALID_END
from src.tehreem.utils import DEFAULT_PARAMS, PREPARED, REPORTS, read_params, write_json


def population_stability_index(reference, current, bins=10):
    reference = np.asarray(reference, dtype='float64')
    current = np.asarray(current, dtype='float64')
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        raise ValueError('Both drift windows need finite observations')
    boundaries = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(boundaries) == 1:
        value = boundaries[0]
        boundaries = np.array([-np.inf, np.nextafter(value, -np.inf), np.nextafter(value, np.inf), np.inf])
    else:
        boundaries[0], boundaries[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=boundaries)[0].astype(float)
    actual = np.histogram(current, bins=boundaries)[0].astype(float)
    expected = (expected + 0.5) / (expected.sum() + 0.5 * len(expected))
    actual = (actual + 0.5) / (actual.sum() + 0.5 * len(actual))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', default=str(DEFAULT_PARAMS))
    args = parser.parse_args()
    settings = read_params(args.params)['monitoring']
    columns = ['date', 'onpromotion', 'oil_price', 'store_nbr', 'family']
    frame = pd.read_parquet(PREPARED, columns=columns)
    frame['weekday'] = frame['date'].dt.day_of_week
    start = TRAIN_END - pd.Timedelta(days=int(settings['reference_days']) - 1)
    reference = frame.loc[frame['date'].between(start, TRAIN_END)]
    current = frame.loc[frame['date'].between(VALID_START, VALID_END)]
    psi = {name: population_stability_index(reference[name], current[name], int(settings['bins']))
           for name in ['onpromotion', 'oil_price', 'weekday']}
    category_distances = {}
    for name in ['store_nbr', 'family']:
        a = reference[name].value_counts(normalize=True)
        b = current[name].value_counts(normalize=True)
        a, b = a.align(b, fill_value=0)
        category_distances[name] = float((a - b).abs().sum() / 2)
    metrics = json.loads((REPORTS / 'metrics.json').read_text())
    write_json(REPORTS / 'drift_report.json', {
        'reference_start': str(start.date()), 'reference_end': str(TRAIN_END.date()),
        'current_start': str(VALID_START.date()), 'current_end': str(VALID_END.date()),
        'reference_rows': len(reference), 'current_rows': len(current),
        'population_stability_index': psi, 'categorical_total_variation': category_distances,
        'validation_rmsle': metrics['rmsle_val'],
        'interpretation': 'Larger values indicate more distribution change. These are descriptive covariate comparisons, not proof that model accuracy declined.',
    })
    os.environ.setdefault('MPLCONFIGDIR', str(REPORTS / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(list(psi), list(psi.values()), color='#2563eb')
    axis.bar_label(bars, fmt='%.3f', padding=3)
    axis.set(title='Training vs validation covariate drift', ylabel='Population stability index')
    figure.tight_layout()
    figure.savefig(REPORTS / 'drift.png', dpi=150)
    plt.close(figure)
    print('Created reports/tehreem/drift_report.json and reports/tehreem/drift.png', flush=True)


if __name__ == '__main__':
    main()
