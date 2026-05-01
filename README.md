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

To build BasicTS datasets from released parquet files:

```bash
python scripts/data_preparation/build_basicts_dataset.py \
  --input-dir raw/parquet \
  --output-dir datasets \
  --level district \
  --years 2024
```

Use `--level subdistrict` for the subdistrict-scale dataset.

Processed datasets are placed under `datasets/`.

```text
datasets/
  SeoulMMOD_District_2024/
  SeoulMMOD_Subdistrict_2024/
```

## Training

```bash
python experiments/train.py \
  -c baselines/STAEformer/SeoulMMOD_District_2024_allmode_h24.py \
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

| Component | Source | Reference |
|---|---|---|
| BasicTS | https://github.com/zezhishao/BasicTS | Yubo Liang, Zezhi Shao, Fei Wang, Zhao Zhang, Tao Sun, and Yongjun Xu. BasicTS: An Open Source Fair Multivariate Time Series Prediction Benchmark. Bench 2022, LNCS 13852, 87-102, 2023. |
| MPGCN | https://github.com/underdoc-wang/MPGCN | Hongzhi Shi, Quanming Yao, Qi Guo, Yaguang Li, Lingyu Zhang, Jieping Ye, Yong Li, and Yan Liu. Predicting Origin-Destination Flow via Multi-Perspective Graph Convolutional Network. ICDE 2020, 1818-1821, 2020. |
| ODCRN | https://github.com/deepkashiwa20/ODCRN | Renhe Jiang, Zhaonan Wang, Zekun Cai, Chuang Yang, Zipei Fan, Tianqi Xia, Go Matsubara, Hiroto Mizuseki, Xuan Song, and Ryosuke Shibasaki. Countrywide Origin-Destination Matrix Prediction and Its Application for COVID-19. ECML PKDD 2021 Applied Data Science Track, LNCS, 319-334, 2021. |
| ODMixer | https://github.com/KLatitude/ODMixer | Yang Liu, Binglin Chen, Yongsen Zheng, Lechao Cheng, Guanbin Li, and Liang Lin. ODMixer: Fine-Grained Spatial-Temporal MLP for Metro Origin-Destination Prediction. IEEE Transactions on Knowledge and Data Engineering, 2025. |
