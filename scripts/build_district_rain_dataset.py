"""Augment a SeoulMMOD district-scale dataset with origin/destination rainfall channels.

Pipeline:
  1. Load hourly district rainfall for the requested year.
  2. For each (origin district, destination district, hour) bucket,
     attach (rain_o, rain_d).
  3. Z-score using train portion (first 70% of T) after log1p.
  4. Concat to existing (T, 625, 8) -> (T, 625, 10) =
       [6 modes, rain_o, rain_d, tod, dow].

Rainfall input: datasets/rainfall/seoul_rain_hourly_{year}.csv
  timestamp, district_cd, rain_mm

Output: datasets/SeoulMMOD_District_{year}_rain/
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
TRAIN_RATIO = 0.7
STEPS_PER_DAY = 24


def load_rain_year(rain_path: Path) -> pd.DataFrame:
    print(f'Loading hourly rainfall from {rain_path}...')
    rain = pd.read_csv(rain_path, dtype={'district_cd': str})
    required = {'timestamp', 'district_cd', 'rain_mm'}
    missing = required - set(rain.columns)
    if missing:
        raise ValueError(f'rainfall file missing columns {sorted(missing)}: {rain_path}')
    rain['timestamp'] = pd.to_datetime(rain['timestamp'], format='mixed')
    rain['rain_mm'] = pd.to_numeric(rain['rain_mm'], errors='coerce').fillna(0.0)
    print(f'  rows: {len(rain):,}; rain_mm mean={rain.rain_mm.mean():.4f}, max={rain.rain_mm.max():.2f}')
    return rain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2024)
    args = parser.parse_args()

    src = REPO / 'datasets' / f'SeoulMMOD_District_{args.year}'
    out = REPO / 'datasets' / f'SeoulMMOD_District_{args.year}_rain'
    rain_path = REPO / 'datasets' / 'rainfall' / f'seoul_rain_hourly_{args.year}.csv'
    district_meta_path = REPO / 'datasets' / 'district_metadata.csv'
    out.mkdir(parents=True, exist_ok=True)

    with open(src / 'desc.json') as f:
        desc = json.load(f)
    T, N, F = tuple(desc['shape'])
    print(f'src: T={T}, N={N}, F={F}')
    src_mm = np.memmap(src / 'data.dat', dtype='float32', mode='r', shape=(T, N, F))

    with open(src / 'od_pairs.json') as f:
        od = json.load(f)
    pairs = od['od_pairs']

    district_meta = pd.read_csv(district_meta_path, dtype={'district_cd': str})
    code_col = 'district_cd'
    missing_cols = [col for col in [code_col] if col not in district_meta.columns]
    if missing_cols:
        raise ValueError(f'district metadata missing columns {missing_cols}: {district_meta_path}')

    all_districts = sorted(district_meta[code_col].astype(str).tolist())
    district_idx = {cd: i for i, cd in enumerate(all_districts)}
    print(f'district order (first 5): {all_districts[:5]}, count: {len(all_districts)}')

    base = Date(args.year, 1, 1)
    rain_mat = np.zeros((T, len(all_districts)), dtype=np.float32)
    rain = load_rain_year(rain_path)
    unknown_districts = sorted(set(rain['district_cd']) - set(district_idx))
    if unknown_districts:
        raise ValueError(f'rainfall file has unknown district_cd values: {unknown_districts[:5]}')
    n_filled = 0
    for _, row in rain.iterrows():
        cd = str(row['district_cd'])
        gi = district_idx[cd]
        t_idx = int((row['timestamp'].to_pydatetime().date() - base).days * STEPS_PER_DAY
                    + row['timestamp'].hour)
        if t_idx < 0 or t_idx >= T:
            continue
        rain_mat[t_idx, gi] = float(row['rain_mm'])
        n_filled += 1
    print(f'rain matrix filled: {n_filled:,} '
          f'(out of {T*len(all_districts):,} cells, {100*n_filled/(T*len(all_districts)):.1f}%)')
    print(f'  per-district mean: {rain_mat.mean(axis=0)[:5]}, overall mean: {rain_mat.mean():.4f}')

    rain_pair = np.zeros((T, len(pairs), 2), dtype=np.float32)
    for idx, (o, d) in enumerate(pairs):
        if o not in district_idx or d not in district_idx:
            print(f'  WARN: pair {idx} ({o},{d}) district not in mapping')
            continue
        rain_pair[:, idx, 0] = rain_mat[:, district_idx[o]]
        rain_pair[:, idx, 1] = rain_mat[:, district_idx[d]]
    print(f'rain_pair built: shape {rain_pair.shape}')

    # Z-score after log1p, using train portion only.
    train_end = int(T * TRAIN_RATIO)
    for ch_name, ch in [('rain_o', 0), ('rain_d', 1)]:
        x_train = rain_pair[:train_end, :, ch]
        x_train = np.log1p(np.maximum(x_train, 0.0))
        mu = float(x_train.mean())
        sd = float(x_train.std()) + 1e-6
        rain_pair[:, :, ch] = (np.log1p(np.maximum(rain_pair[:, :, ch], 0.0)) - mu) / sd
        print(f'  {ch_name}: log1p train mu={mu:.4f} sd={sd:.4f}; '
              f'after z-score min={rain_pair[:, :, ch].min():.3f} max={rain_pair[:, :, ch].max():.3f}')

    # Concat: [6 modes, rain_o, rain_d, tod, dow]
    new_F = F + 2
    out_arr = np.zeros((T, N, new_F), dtype=np.float32)
    out_arr[:, :, :6] = src_mm[:, :, :6]
    out_arr[:, :, 6:8] = rain_pair
    out_arr[:, :, 8:10] = src_mm[:, :, 6:8]

    out_path = out / 'data.dat'
    mm = np.memmap(out_path, dtype=np.float32, mode='w+', shape=out_arr.shape)
    mm[:] = out_arr[:]
    mm.flush()
    del mm
    print(f'wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) shape {out_arr.shape}')

    new_desc = dict(desc)
    new_desc['name'] = f'SeoulMMOD_District_{args.year}_rain'
    new_desc['domain'] = desc['domain'] + ' + origin/dest hourly rainfall (z-score of log1p mm)'
    new_desc['shape'] = list(out_arr.shape)
    new_desc['num_features'] = new_F
    new_desc['feature_description'] = (
        list(desc['feature_description'][:6])
        + ['rain_origin_zscore', 'rain_dest_zscore']
        + list(desc['feature_description'][6:])
    )
    with open(out / 'desc.json', 'w') as f:
        json.dump(new_desc, f, indent=2)
    shutil.copy(src / 'od_pairs.json', out / 'od_pairs.json')
    print('done.')


if __name__ == '__main__':
    main()
