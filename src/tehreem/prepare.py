"""Join raw competition inputs without duplicating or dropping sales rows."""

from __future__ import annotations

import pandas as pd

from src.tehreem.utils import PREPARED, ROOT


def add_holidays(frame: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    holidays = holidays.loc[~holidays['transferred'].fillna(False)].copy()
    days_off = holidays.loc[holidays['type'].isin(['Holiday', 'Transfer', 'Additional', 'Bridge'])]
    frame['holiday_national'] = frame['date'].isin(
        days_off.loc[days_off['locale'] == 'National', 'date']
    ).astype('int8')
    for locale, store_column, name in [
        ('Regional', 'state', 'holiday_regional'), ('Local', 'city', 'holiday_local'),
    ]:
        keys = days_off.loc[days_off['locale'] == locale, ['date', 'locale_name']].drop_duplicates()
        keys = keys.rename(columns={'locale_name': store_column})
        keys[name] = 1
        frame = frame.merge(keys, on=['date', store_column], how='left', validate='many_to_one')
        frame[name] = frame[name].fillna(0).astype('int8')
    for holiday_type, name in [('Event', 'is_event'), ('Work Day', 'is_workday')]:
        dates = holidays.loc[
            (holidays['locale'] == 'National') & (holidays['type'] == holiday_type), 'date'
        ]
        frame[name] = frame['date'].isin(dates).astype('int8')
    return frame


def main() -> None:
    raw = ROOT / 'data/raw'
    types = {'id': 'int32', 'store_nbr': 'int16', 'family': 'category',
             'sales': 'float32', 'onpromotion': 'int32'}
    train = pd.read_csv(raw / 'train.csv', parse_dates=['date'], dtype=types)
    test = pd.read_csv(raw / 'test.csv', parse_dates=['date'], dtype={k:v for k,v in types.items() if k != 'sales'})
    frame = pd.concat([train, test], ignore_index=True)
    original_ids = frame['id'].to_numpy(copy=True)
    if frame.duplicated(['date', 'store_nbr', 'family']).any():
        raise ValueError('Raw sales have duplicate store/family/date keys')

    stores = pd.read_csv(raw / 'stores.csv').rename(columns={'type': 'store_type'})
    frame = frame.merge(stores, on='store_nbr', how='left', validate='many_to_one')
    oil = pd.read_csv(raw / 'oil.csv', parse_dates=['date']).set_index('date')['dcoilwtico']
    calendar = pd.date_range(frame['date'].min(), frame['date'].max())
    # Forward filling uses earlier prices only; the first unknown price becomes 0.
    oil = oil.reindex(calendar).ffill().fillna(0).astype('float32')
    frame['oil_price'] = frame['date'].map(oil)
    holidays = pd.read_csv(raw / 'holidays_events.csv', parse_dates=['date'])
    frame = add_holidays(frame, holidays)
    for column in ['family', 'city', 'state', 'store_type']:
        frame[column] = frame[column].astype('category')
    if len(frame) != len(original_ids) or not (frame['id'].to_numpy() == original_ids).all():
        raise ValueError('Preparation changed the raw row count or row order')
    if frame[['city', 'state', 'store_type', 'cluster']].isna().any().any():
        raise ValueError('Store metadata is missing after the join')
    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PREPARED, index=False)
    print(f'Prepared {len(train):,} training and {len(test):,} test rows -> {PREPARED.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
