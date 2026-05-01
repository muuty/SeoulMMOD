"""Augment a SeoulMMOD district-scale dataset with origin/destination hourly rainfall channels.

Pipeline:
  1. Load 12 monthly Seoul AWS rain CSVs (10-min interval, per-station)
     for the requested year.
  2. Aggregate to (district, hour) hourly rainfall (mm/h):
       - sum the 6 ten-minute readings per station-hour,
       - mean across stations within the district.
  3. Map district name -> 5-digit admin code via district_centroids.csv.
  4. For each (origin district, destination district, hour) bucket,
     attach (rain_o, rain_d).
  5. Z-score using train portion (first 70% of T) after log1p.
  6. Concat to existing (T, 625, 8) -> (T, 625, 10) =
       [6 modes, rain_o, rain_d, tod, dow].

Output: datasets/SeoulMMOD_District_{year}_rain/
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date as Date
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
TRAIN_RATIO = 0.7
STEPS_PER_DAY = 24


def load_rain_year(rain_glob: str) -> pd.DataFrame:
    files = sorted(glob(rain_glob))
    print(f'Loading {len(files)} rain CSVs from {rain_glob}...')
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding='cp949')
        df.columns = ['station_code', 'station_name', 'district_code', 'district_name', 'rain_10min', 'ts']
        dfs.append(df[['station_name', 'district_name', 'rain_10min', 'ts']])
    rain = pd.concat(dfs, ignore_index=True)
    rain['ts'] = pd.to_datetime(rain['ts'], format='mixed')
    rain['rain_10min'] = pd.to_numeric(rain['rain_10min'], errors='coerce').fillna(0.0)
    rain['date'] = rain['ts'].dt.date
    rain['hour'] = rain['ts'].dt.hour
    print(f'  total rows: {len(rain):,}')
    return rain


def aggregate_to_district_hour(rain: pd.DataFrame) -> pd.DataFrame:
    print('Aggregating to (district, date, hour) hourly mm...')
    per_station = (
        rain.groupby(['station_name', 'district_name', 'date', 'hour'], as_index=False)
        ['rain_10min'].sum()
    )
    per_district_hour = (
        per_station.groupby(['district_name', 'date', 'hour'], as_index=False)
        ['rain_10min'].mean()
    )
    per_district_hour = per_district_hour.rename(columns={'rain_10min': 'rain_mm'})
    print(f'  {len(per_district_hour):,} (district, date, hour) rows; '
          f'rain_mm mean={per_district_hour.rain_mm.mean():.4f}, max={per_district_hour.rain_mm.max():.2f}')
    return per_district_hour


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2024)
    args = parser.parse_args()

    src = REPO / 'datasets' / f'SeoulMMOD_District_{args.year}'
    out = REPO / 'datasets' / f'SeoulMMOD_District_{args.year}_rain'
    rain_glob = str(REPO / 'datasets' / 'rainfall' / f'seoul_rain_{args.year}*.csv')
    district_cent_path = REPO / 'datasets' / 'district_centroids.csv'
    out.mkdir(parents=True, exist_ok=True)

    with open(src / 'desc.json') as f:
        desc = json.load(f)
    T, N, F = tuple(desc['shape'])
    print(f'src: T={T}, N={N}, F={F}')
    src_mm = np.memmap(src / 'data.dat', dtype='float32', mode='r', shape=(T, N, F))

    with open(src / 'od_pairs.json') as f:
        od = json.load(f)
    pairs = od['od_pairs']

    district_cent = pd.read_csv(district_cent_path)
    code_col = 'district_cd' if 'district_cd' in district_cent.columns else 'gu_cd'
    name_col = 'district_name' if 'district_name' in district_cent.columns else 'gu_name'
    district_cent['district_cd_str'] = district_cent[code_col].astype(str)
    name_to_cd = dict(zip(district_cent[name_col], district_cent['district_cd_str']))
    print(f'name_to_cd: {len(name_to_cd)} mappings')

    all_districts = sorted(name_to_cd.values())
    district_idx = {cd: i for i, cd in enumerate(all_districts)}
    print(f'district order (first 5): {all_districts[:5]}, count: {len(all_districts)}')

    rain = load_rain_year(rain_glob)
    per_district_hour = aggregate_to_district_hour(rain)

    base = Date(args.year, 1, 1)
    rain_mat = np.zeros((T, len(all_districts)), dtype=np.float32)
    n_filled = 0
    for _, row in per_district_hour.iterrows():
        district_name = row['district_name']
        if district_name not in name_to_cd:
            continue
        cd = name_to_cd[district_name]
        if cd not in district_idx:
            continue
        gi = district_idx[cd]
        day_idx = (row['date'] - base).days
        if day_idx < 0:
            continue
        t_idx = day_idx * STEPS_PER_DAY + int(row['hour'])
        if t_idx >= T:
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
