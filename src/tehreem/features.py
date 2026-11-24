"""Build calendar and leakage-safe sales features in memory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.evaluate import check_lags

KEYS = ['store_nbr', 'family']
CATEGORICALS = ['store_nbr', 'family', 'city', 'state', 'store_type', 'cluster']
KNOWN_FEATURES = [
    *CATEGORICALS, 'onpromotion', 'oil_price', 'holiday_national',
    'holiday_regional', 'holiday_local', 'is_event', 'is_workday',
]


def build_features(prepared: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    lags = [int(value) for value in config['lags']]
    shift = int(config['rolling_shift'])
    week_shift = int(config['weekday_shift_weeks'])
    week_window = int(config['weekday_window_weeks'])
    check_lags([*lags, shift, week_shift * 7])
    windows = [int(value) for value in config['rolling_windows']]
    if not windows or min(windows) < 1 or week_window < 1:
        raise ValueError('Rolling windows must be positive')
    if prepared.duplicated(['date', *KEYS]).any():
        raise ValueError('Duplicate store/family/date rows would corrupt lags')

    frame = prepared.copy()
    for column in CATEGORICALS:
        frame[column] = frame[column].astype('category')
    dates = frame['date'].dt
    calendar_features = {
        'year': dates.year.astype('int16'),
        'month': dates.month.astype('int8'),
        'day': dates.day.astype('int8'),
        'weekday': dates.day_of_week.astype('int8'),
        'is_weekend': (dates.day_of_week >= 5).astype('int8'),
        'payday': ((dates.day == 15) | dates.is_month_end).astype('int8'),
        'trend_days': (frame['date'] - pd.Timestamp('2013-01-01')).dt.days.astype('int16'),
    }
    for name, values in calendar_features.items():
        frame[name] = values
    features = [*KNOWN_FEATURES, *calendar_features]

    # A complete date index makes shift(16) mean 16 calendar days, including
    # missing Christmas dates. Observed test labels remain unknown (NaN).
    calendar = pd.date_range(prepared['date'].min(), prepared['date'].max())
    sales = prepared.pivot(index='date', columns=KEYS, values='sales').reindex(calendar)
    observed_ids = prepared.pivot(index='date', columns=KEYS, values='id').reindex(calendar)
    sales = sales.mask(observed_ids.isna(), 0).astype('float32')
    del observed_ids
    date_positions = calendar.get_indexer(prepared['date'])
    pair_positions = sales.columns.get_indexer(pd.MultiIndex.from_frame(prepared[KEYS]))
    if (date_positions < 0).any() or (pair_positions < 0).any():
        raise ValueError('Failed to align the sales calendar with original rows')

    def add(name: str, values: pd.DataFrame | np.ndarray) -> None:
        array = values.to_numpy() if isinstance(values, pd.DataFrame) else values
        frame[name] = array[date_positions, pair_positions].astype('float32')
        features.append(name)

    for lag in lags:
        add(f'sales_lag_{lag}', sales.shift(lag))
    shifted = sales.shift(shift)
    for window in windows:
        add(f'sales_mean_{window}', shifted.rolling(window, min_periods=1).mean())
    add('sales_std_28', shifted.rolling(28, min_periods=2).std())

    weekly = np.full(sales.shape, np.nan, dtype='float32')
    for weekday in range(7):
        indices = np.flatnonzero(calendar.day_of_week == weekday)
        values = sales.iloc[indices].shift(week_shift).rolling(week_window, min_periods=1).mean()
        weekly[indices] = values.to_numpy(dtype='float32')
    add('sales_weekday_mean', weekly)
    return frame, features
