# Datasets

Place processed SeoulMMOD datasets in this directory.

Expected layout:

```text
datasets/
  SeoulMMOD_District_2024/
    data.dat
    desc.json
  SeoulMMOD_Subdistrict_2024/
    data.dat
    desc.json
  calendar.csv
  district_features.csv
  district_centroids.csv
  rainfall/
    seoul_rain_2024*.csv
```

The dataset release is distributed separately from this code repository.
Configuration files load datasets by directory name. For example,
`baselines/STAEformer/SeoulMMOD_District_2024_allmode_h24.py` expects
`datasets/SeoulMMOD_District_2024/`.

If processed parquet files are available, build the base datasets from the
repository root with:

```bash
python scripts/build_basicts_dataset.py \
  --input-dir raw/parquet --output-dir datasets --level district --years 2024

python scripts/build_basicts_dataset.py \
  --input-dir raw/parquet --output-dir datasets --level subdistrict --years 2024
```

The calendar, POI, and rainfall files are only needed for contextual
experiments.
