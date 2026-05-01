# SeoulMMOD Code

Code for SeoulMMOD benchmark experiments.

Datasets, checkpoints, and logs are not included.

## Table of Contents

- [Structure](#structure)
- [Setup](#setup)
- [Data](#data)
- [BasicTS Baselines](#basicts-baselines)
- [OD Baselines](#od-baselines)
- [Contextual Variants](#contextual-variants)
- [Cross-Year Evaluation](#cross-year-evaluation)
- [Source Code and Citations](#source-code-and-citations)

## Structure

```text
baselines/                 BasicTS baseline configs and model code
basicts/                   BasicTS framework code used in the benchmark
datasets/                  processed datasets and optional metadata
experiments/               train, evaluate, inference, cross-year scripts
od_baselines/              MPGCN, ODCRN, and ODMixer source code
scripts/                   OD baseline runner and data builders
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place BasicTS datasets under `datasets/`.

```text
datasets/
  SeoulMOD_GU_2024/
    data.dat
    desc.json
    od_pairs.json
  SeoulMOD_2024/
    data.dat
    desc.json
    od_pairs.json
```

To build datasets from released parquet files:

```bash
python scripts/data_preparation/build_basicts_dataset.py \
  --input-dir raw/parquet \
  --output-dir datasets \
  --level gu \
  --years 2024

python scripts/data_preparation/build_basicts_dataset.py \
  --input-dir raw/parquet \
  --output-dir datasets \
  --level dong \
  --years 2024
```

The builder expects `od_flow_{year}.parquet`.

## BasicTS Baselines

```bash
python experiments/train.py \
  -c baselines/STAEformer/SeoulMOD_GU_2024_allmode.py \
  -g 0
```

Use another config under `baselines/` to run a different model.

## OD Baselines

MPGCN, ODCRN, and ODMixer use one runner.

```bash
python scripts/run_od_native.py \
  --model odmixer \
  --level gu \
  --horizon 24 \
  --data_dir datasets/SeoulMOD_GU_2024 \
  --gpu 0
```

Set `--model` to `mpgcn`, `odcrn`, or `odmixer`.

## Contextual Variants

Build GU-level contextual datasets:

```bash
python scripts/data_preparation/SeoulMOD_GU_ctx/build_dataset.py --year 2024
python scripts/data_preparation/SeoulMOD_GU_rain/build_dataset.py --year 2024
```

Expected metadata:

```text
datasets/calendar.csv
datasets/gu_features.csv
datasets/gu_centroids.csv
datasets/rainfall/seoul_rain_2024*.csv
```

Example:

```bash
python experiments/train.py \
  -c baselines/STID/holiday_SeoulMOD_GU_2024_h24.py \
  -g 0
```

## Cross-Year Evaluation

```bash
python experiments/eval_crossyear.py \
  -c baselines/STAEformer/SeoulMOD_GU_2024_allmode.py \
  --test-dataset SeoulMOD_GU_2025
```

## Source Code and Citations

| Component | Source | Citation |
|---|---|---|
| BasicTS | https://github.com/zezhishao/BasicTS | Shao et al., 2024. Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis |
| MPGCN | https://github.com/underdoc-wang/MPGCN | Shi et al., 2020. Predicting Origin-Destination Flow via Multi-Perspective Graph Convolutional Network |
| ODCRN | https://github.com/deepkashiwa20/ODCRN | Jiang et al., 2021. Countrywide Origin-Destination Matrix Prediction and Its Application for COVID-19 |
| ODMixer | https://github.com/KLatitude/ODMixer | Liu et al., 2024. Fine-grained Spatial-temporal MLP Architecture for Metro Origin-Destination Prediction |

Please cite the original papers when using these components.
