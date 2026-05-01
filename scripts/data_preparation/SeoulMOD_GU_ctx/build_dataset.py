"""Augment a SeoulMOD GU-level dataset with a holiday-state channel.

Output: datasets/SeoulMOD_GU_{year}_ctx/
  data.dat   : (T, N=625, F=9) where F = [6 modes, tod, dow, holiday_state]
               holiday_state in {0=workday, 1=weekend, 2=holiday},
               stored as state / 3 so the model recovers the index via
               (val * 3).long().
  desc.json
  od_pairs.json (copied from source)

Reads:
  datasets/SeoulMOD_GU_{year}/{data.dat, desc.json, od_pairs.json}
  datasets/calendar.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HOLIDAY_STATES = 3  # 0=workday, 1=weekend, 2=holiday


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2024)
    args = parser.parse_args()

    src = REPO / 'datasets' / f'SeoulMOD_GU_{args.year}'
    out = REPO / 'datasets' / f'SeoulMOD_GU_{args.year}_ctx'
    cal_path = REPO / 'datasets' / 'calendar.csv'
    out.mkdir(parents=True, exist_ok=True)

    with open(src / 'desc.json') as f:
        desc = json.load(f)
    T, N, F = tuple(desc['shape'])
    print(f'src shape: T={T}, N={N}, F={F}')
    src_mm = np.memmap(src / 'data.dat', dtype='float32', mode='r', shape=(T, N, F))

    start_date = desc['start_date']
    end_date = desc['end_date']
    print(f'date range: {start_date} ~ {end_date}')

    cal = pd.read_csv(cal_path)
    cal['date'] = pd.to_datetime(cal['date'])
    days = pd.date_range(start=start_date, end=end_date, freq='D')
    cal_idx = cal.set_index('date').reindex(days)
    if cal_idx[['is_weekend', 'is_holiday']].isna().any().any():
        missing = cal_idx[cal_idx['is_holiday'].isna()].index.tolist()
        raise ValueError(f'calendar missing dates: {missing[:5]}...')

    is_holiday = cal_idx['is_holiday'].astype(int).to_numpy()
    is_weekend = cal_idx['is_weekend'].astype(int).to_numpy()
    state_per_day = np.where(is_holiday == 1, 2,
                             np.where(is_weekend == 1, 1, 0)).astype(np.int8)
    print(f'days: {len(days)}, state counts (workday/weekend/holiday): '
          f'{(state_per_day == 0).sum()}, {(state_per_day == 1).sum()}, {(state_per_day == 2).sum()}')

    state_per_hour = np.repeat(state_per_day, 24)
    if state_per_hour.shape[0] != T:
        raise ValueError(f'hour count mismatch: {state_per_hour.shape[0]} vs T={T}')

    state_norm = state_per_hour.astype(np.float32) / HOLIDAY_STATES
    holiday_chan = np.broadcast_to(state_norm[:, None, None], (T, N, 1)).astype(np.float32)

    new_F = F + 1
    out_arr = np.concatenate([np.array(src_mm), holiday_chan], axis=-1).astype(np.float32)
    print(f'output shape: {out_arr.shape}')

    out_path = out / 'data.dat'
    mm = np.memmap(out_path, dtype=np.float32, mode='w+', shape=out_arr.shape)
    mm[:] = out_arr[:]
    mm.flush()
    del mm
    print(f'wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)')

    new_desc = dict(desc)
    new_desc['name'] = f'SeoulMOD_GU_{args.year}_ctx'
    new_desc['domain'] = desc['domain'] + ' + holiday-state context'
    new_desc['shape'] = list(out_arr.shape)
    new_desc['num_features'] = new_F
    new_desc['feature_description'] = list(desc['feature_description']) + ['holiday_state']
    with open(out / 'desc.json', 'w') as f:
        json.dump(new_desc, f, indent=2)

    shutil.copy(src / 'od_pairs.json', out / 'od_pairs.json')
    print('desc + od_pairs copied. done.')


if __name__ == '__main__':
    main()
