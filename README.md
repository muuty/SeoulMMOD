# SeoulMMOD Code

This repository contains the benchmark code for SeoulMMOD, including
BasicTS-based baselines, OD-native model runners, and scripts used to
reproduce the experiments in the paper. It is intentionally compact:
datasets, trained checkpoints, and experiment logs are not included.

## Contents

- `basicts/`: the BasicTS framework code used by the benchmark.
- `baselines/`: model definitions and SeoulMMOD configuration files for
  Historical Average, Latest Value, STAEformer, STID, AGCRN, Graph WaveNet,
  MTGNN, iTransformer, SOFTS, MTSMixer, and TimeMixer.
- `scripts/run_od_native.py`: the OD-matrix runner for MPGCN, ODCRN, and
  ODMixer.
- `od_baselines/`: compact source copies for MPGCN, ODCRN, and ODMixer used by
  the OD-native runner.
- `scripts/data_preparation/`: scripts that build the base GU/Dong BasicTS
  datasets and contextual dataset variants.
- `experiments/`: BasicTS train/evaluate entry points.
- `raw/`: optional location for raw or parquet source files used by the
  conversion scripts. `raw/` is ignored by git and can be created as needed.

## Setup

```bash
conda create -n seoulmmod python=3.10
conda activate seoulmmod
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version if the default resolver
does not choose the desired wheel.

The code was tested with Python 3.10. The benchmark uses dataset metadata when
loading configuration files, so training and evaluation commands require the
processed dataset directories described below.

External baseline and framework code is included only where it is needed by the
benchmark runner, with original notices kept where available. See
[Source Code and Citations](#source-code-and-citations) for upstream links,
licenses, and citation information.

## Source Code and Citations

This repository includes compact source copies or derived code from the
following projects.

| Component | Use in this repository | Source | License or notice | Citation |
|---|---|---|---|---|
| BasicTS | Training framework and baseline runner code under `basicts/` | https://github.com/zezhishao/BasicTS | Apache-2.0 | Shao et al., 2024 |
| MPGCN | OD-native baseline source under `od_baselines/MPGCN/` | https://github.com/underdoc-wang/MPGCN | MIT license included in `od_baselines/MPGCN/LICENSE` | Shi et al., 2020 |
| ODCRN | OD-native baseline source under `od_baselines/ODCRN/` | https://github.com/deepkashiwa20/ODCRN | Upstream repository does not include a separate license file. The source copy is included for benchmark reproducibility. | Jiang et al., 2021 |
| ODMixer | OD-native baseline source under `od_baselines/ODMixer/` | https://github.com/KLatitude/ODMixer | Upstream repository does not include a separate license file. The source copy is included for benchmark reproducibility. | Han et al., 2024 |

Please cite the original model papers when using the corresponding baselines.

## Data Layout

Place processed BasicTS datasets under `datasets/`. Each dataset directory
should contain:

```text
datasets/
  SeoulMOD_GU_2024/
    data.dat
    desc.json
  SeoulMOD_2024/
    data.dat
    desc.json
```

The BasicTS configs load datasets by name, so
`baselines/STAEformer/SeoulMOD_GU_2024_allmode.py` expects
`datasets/SeoulMOD_GU_2024/`, while
`baselines/STAEformer/SeoulMOD_2024_allmode.py` expects
`datasets/SeoulMOD_2024/`.

The data release is distributed separately from this code repository. If the
processed parquet files are available under `raw/parquet/`, build the base
datasets with:

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

The builder expects `od_flow_{year}.parquet`, where each file contains hourly
OD flows by transport mode. Keep raw source files under `raw/`. The `raw/`
directory is ignored by git.

## BasicTS Baselines

Run a district-scale model with:

```bash
python experiments/train.py \
  -c baselines/STAEformer/SeoulMOD_GU_2024_allmode.py \
  -g 0
```

The same entry point works for other BasicTS configs under `baselines/`.
Checkpoints are written to `checkpoints/`, and logs are written to `logs/`.

## OD-Native Baselines

MPGCN, ODCRN, and ODMixer use the matrix-shaped OD runner:

```bash
python scripts/run_od_native.py \
  --model odmixer \
  --level gu \
  --horizon 24 \
  --data_dir datasets/SeoulMOD_GU_2024 \
  --gpu 0
```

Use `--model mpgcn` or `--model odcrn` for the other OD-native baselines.

## Contextual Variants

The contextual ablation in the appendix uses STID extended with calendar,
rainfall, or POI signals on auxiliary dataset variants. Build the variants
from the base GU dataset (defaults to year 2024):

```bash
python scripts/data_preparation/SeoulMOD_GU_ctx/build_dataset.py --year 2024
python scripts/data_preparation/SeoulMOD_GU_rain/build_dataset.py --year 2024
```

The `_ctx` build adds a holiday-state channel and reads
`datasets/calendar.csv`. The `_rain` build aggregates the Seoul AWS rain
CSVs in `datasets/rainfall/` and adds origin/destination rainfall channels.
POI is loaded directly inside the model from `datasets/gu_features.csv` and
does not need a dataset rebuild.

Train a contextual variant with:

```bash
python experiments/train.py \
  -c baselines/STID/holiday_SeoulMOD_GU_2024_h24.py \
  -g 0
```

## Cross-Year Evaluation

Train on the 2024 calendar and evaluate per-season MAE on the 2025
calendar (Section 4.5 of the paper):

```bash
python experiments/eval_crossyear.py \
  -c baselines/STAEformer/SeoulMOD_GU_2024_allmode.py \
  --test-dataset SeoulMOD_GU_2025
```
