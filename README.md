# SeoulMMOD

Code for SeoulMMOD benchmark experiments.

Data, checkpoints, and logs are not included.

## Structure

```text
baselines/       BasicTS baseline configs and model code
basicts/         BasicTS framework code used in the benchmark
datasets/        dataset placeholder and data notes
experiments/     training and evaluation scripts
od_baselines/    MPGCN, ODCRN, and ODMixer source code
scripts/         data builders and OD baseline runner
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

Place processed datasets under `datasets/`.

```text
datasets/
  SeoulMMOD_District_2024/
  SeoulMMOD_Subdistrict_2024/
```

To build BasicTS datasets from released parquet files:

```bash
python scripts/data_preparation/build_basicts_dataset.py \
  --input-dir raw/parquet \
  --output-dir datasets \
  --level district \
  --years 2024
```

Use `--level subdistrict` for the subdistrict-scale dataset.

## Training

```bash
python experiments/train.py \
  -c baselines/STAEformer/SeoulMMOD_District_2024_allmode.py \
  -g 0
```

Use another config under `baselines/` for a different model.

## OD Baselines

```bash
python scripts/run_od_native.py \
  --model odmixer \
  --level district \
  --horizon 24 \
  --data_dir datasets/SeoulMMOD_District_2024 \
  --gpu 0
```

`--model` supports `mpgcn`, `odcrn`, and `odmixer`.

## Context

```bash
python scripts/data_preparation/SeoulMMOD_District_ctx/build_dataset.py --year 2024
python scripts/data_preparation/SeoulMMOD_District_rain/build_dataset.py --year 2024
```

These scripts use optional files such as `calendar.csv`,
`district_features.csv`, `district_centroids.csv`, and rainfall CSVs under
`datasets/`.

## Citation

Please cite the original papers for external components used in this repository.

| Component | Source | Paper |
|---|---|---|
| BasicTS | https://github.com/zezhishao/BasicTS | Shao et al., 2024 |
| MPGCN | https://github.com/underdoc-wang/MPGCN | Shi et al., 2020 |
| ODCRN | https://github.com/deepkashiwa20/ODCRN | Jiang et al., 2021 |
| ODMixer | https://github.com/KLatitude/ODMixer | Liu et al., 2024 |
