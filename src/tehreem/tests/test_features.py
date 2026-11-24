"""Forecast-origin leakage and calendar alignment are model correctness rules."""

import numpy as np
import pandas as pd
import pytest

from src.tehreem.features import build_features


@pytest.fixture
def settings():
    return {'lags': [16, 28], 'rolling_windows': [7, 28], 'rolling_shift': 16,
            'weekday_shift_weeks': 3, 'weekday_window_weeks': 8}


@pytest.fixture
def prepared():
    rows = []
    for store, family in [(1, 'A'), (2, 'B')]:
        for day, date in enumerate(pd.date_range('2016-12-01', '2017-02-28')):
            if store == 1 and date == pd.Timestamp('2016-12-25'):
                continue
            rows.append({
                'id': len(rows), 'date': date, 'store_nbr': store, 'family': family,
                'sales': float(store * 1000 + day), 'onpromotion': 0,
                'city': f'city-{store}', 'state': 'state', 'store_type': 'D',
                'cluster': store, 'oil_price': 50.0, 'holiday_national': 0,
                'holiday_regional': 0, 'holiday_local': 0, 'is_event': 0, 'is_workday': 0,
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def test_missing_date_is_a_calendar_gap_and_series_stay_separate(prepared, settings):
    frame, _ = build_features(prepared, settings)
    same_day = frame.loc[frame['date'] == pd.Timestamp('2017-01-10')]
    assert same_day.loc[same_day['store_nbr'] == 1, 'sales_lag_16'].iloc[0] == 0
    assert same_day.loc[same_day['store_nbr'] == 2, 'sales_lag_16'].iloc[0] == 2024
    np.testing.assert_array_equal(frame['id'], prepared['id'])


def test_hiding_all_forecast_sales_does_not_change_forecast_features(prepared, settings):
    first, features = build_features(prepared, settings)
    hidden = prepared.copy()
    forecast = hidden['date'] >= pd.Timestamp('2017-02-13')
    hidden.loc[forecast, 'sales'] = np.nan
    second, _ = build_features(hidden, settings)
    pd.testing.assert_frame_equal(first.loc[forecast, features], second.loc[forecast, features])
    assert 'sales' not in features and 'id' not in features and 'date' not in features


@pytest.mark.parametrize('key,value', [('lags', [7]), ('rolling_shift', 7), ('weekday_shift_weeks', 2)])
def test_unsafe_history_configuration_is_rejected(prepared, settings, key, value):
    settings[key] = value
    with pytest.raises(ValueError, match='leak'):
        build_features(prepared, settings)
