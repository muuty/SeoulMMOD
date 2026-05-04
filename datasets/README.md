# Datasets

Place the released SeoulMMOD dataset contents in this directory.

Expected raw layout:

```text
datasets/
  raw/
    od_flow_2023.parquet
    od_flow_2024.parquet
    od_flow_2025.parquet
  calendar.csv
  district_poi.csv
  district_metadata.csv
  rainfall/
    seoul_rain_hourly_2023.csv
    seoul_rain_hourly_2024.csv
    seoul_rain_hourly_2025.csv
```

The dataset release is distributed separately from this code repository.
Configuration files load datasets by directory name. For example,
`baselines/STAEformer/SeoulMMOD_District_2024_allmode_h24.py` expects
`datasets/SeoulMMOD_District_2024/`.

If processed parquet files are available, build the base datasets from the
repository root with:

```bash
python scripts/build_basicts_dataset.py \
  --input-dir datasets/raw --output-dir datasets --level district --years 2024

python scripts/build_basicts_dataset.py \
  --input-dir datasets/raw --output-dir datasets --level subdistrict --years 2024
```

Processed datasets are written under `datasets/`, for example:

```text
datasets/
  SeoulMMOD_District_2024/
    data.dat
    desc.json
  SeoulMMOD_Subdistrict_2024/
    data.dat
    desc.json
```

The calendar, POI, and rainfall files are only needed for contextual
experiments.
