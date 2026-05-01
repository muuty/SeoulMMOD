# Datasets

Place processed SeoulMMOD datasets in this directory.

Expected layout:

```text
datasets/
  SeoulMOD_GU_2024/
    data.dat
    desc.json
  SeoulMOD_2024/
    data.dat
    desc.json
  calendar.csv
  gu_features.csv
  gu_centroids.csv
  rainfall/
    seoul_rain_2024*.csv
```

The dataset release is distributed separately from this code repository.
Configuration files load datasets by directory name. For example,
`baselines/STAEformer/SeoulMOD_GU_2024_allmode.py` expects
`datasets/SeoulMOD_GU_2024/`.

If processed parquet files are available, build the base datasets from the
repository root with:

```bash
python scripts/data_preparation/build_basicts_dataset.py \
  --input-dir raw/parquet --output-dir datasets --level gu --years 2024

python scripts/data_preparation/build_basicts_dataset.py \
  --input-dir raw/parquet --output-dir datasets --level dong --years 2024
```

The calendar, POI, and rainfall files are only needed for contextual
experiments.
