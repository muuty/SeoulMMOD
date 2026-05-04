"""Cross-year eval for CD-style (ODPairDataset) models: 2024-trained → 2025 test, per season."""
import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime, timedelta

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

MODE_NAMES = ["metro_bus", "local_bus", "subway", "walk", "car", "other"]
SEASONS = {
    'spring': (3, 4, 5),
    'summer': (6, 7, 8),
    'fall': (9, 10, 11),
    'winter': (12, 1, 2),
}


def load_config(path: str):
    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.').replace('\\', '.')
    cfg_name = path.split('.')[-1]
    return __import__(path, fromlist=[cfg_name]).CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cfg", required=True, help="2024 training config")
    ap.add_argument("--test-dataset", required=True, help="2025 dataset name (e.g. SeoulMMOD_District_2025)")
    ap.add_argument("--gpus", default="0")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.cfg)

    # Scaler computed on 2024 training set (keep 2024 stats)
    scaler_param = dict(cfg.SCALER.PARAM)
    scaler = cfg.SCALER.TYPE(**scaler_param)
    mean = scaler.mean.item()
    std = scaler.std.item()

    # Substitute test dataset name (2025) — reuse entire ODPairDataset pipeline
    ds_param = dict(cfg.DATASET.PARAM)
    ds_param['dataset_name'] = args.test_dataset
    ds_param['mode'] = 'test'
    dataset = cfg.DATASET.TYPE(**ds_param)
    # Override to use ALL of 2025 as test window (skip train/val split on 2025)
    dataset.t_start = 0
    dataset.t_end = dataset.total_time
    dataset.n_time_samples = dataset.t_end - dataset.t_start - dataset.input_len - dataset.output_len + 1
    loader = DataLoader(dataset, batch_size=cfg.TEST.DATA.BATCH_SIZE, shuffle=False, num_workers=2)

    # Model + checkpoint (2024 weights)
    ckpt_dir = cfg.TRAIN.CKPT_SAVE_DIR
    subdirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
    if not subdirs:
        sys.exit(f"No subdirs in {ckpt_dir}")
    subdir = sorted(subdirs)[-1]
    ckpt_path = os.path.join(ckpt_dir, subdir, f"{cfg.MODEL.NAME}_best_val_MAE.pt")

    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    forward_features = cfg.MODEL.get("FORWARD_FEATURES", [0])
    target_features = cfg.MODEL.get("TARGET_FEATURES", [0])
    print(f"Model: {cfg.MODEL.NAME}", file=sys.stderr)
    print(f"2024 ckpt: {ckpt_path}", file=sys.stderr)
    print(f"2025 test: {args.test_dataset}", file=sys.stderr)
    print(f"Test samples: {len(dataset)}", file=sys.stderr)

    # 2025 dataset start_date (for season mapping)
    desc = dataset.description
    start_date = datetime.strptime(desc.get('start_date', '2025-01-01'), '%Y-%m-%d')
    t_start_offset = dataset.t_start  # first absolute index in test mode
    input_len = dataset.input_len

    abs_errors = []  # per-sample MAE for each batch element
    months = []  # month of the last input hour (t + input_len - 1)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_all = batch["inputs"].float()
            y_all = batch["target"].float()

            x_sel = (x_all[..., forward_features].clone() - mean) / std
            y_4_dec = (y_all[..., forward_features].clone() - mean) / std
            y_4_dec[..., 0] = 0.0

            pred = model(history_data=x_sel.to(device), future_data=y_4_dec.to(device),
                         batch_seen=None, epoch=None, train=False)
            if isinstance(pred, dict):
                pred = pred.get("prediction", list(pred.values())[0])
            pred = pred.cpu().float().numpy()
            pred = pred * std + mean
            target = y_all[..., target_features].numpy()

            # Per-sample MAE: mean over (L, N, C)
            mae_per_sample = np.mean(np.abs(pred - target), axis=(1, 2, 3))
            abs_errors.extend(mae_per_sample.tolist())

            # Month from time_index
            time_idx = batch["time_index"].numpy()
            for t in time_idx:
                hour_offset = int(t)
                dt = start_date + timedelta(hours=hour_offset)
                months.append(dt.month)

    abs_errors = np.array(abs_errors)
    months = np.array(months)

    print()
    print(f"=== Cross-year: {cfg.MODEL.NAME} ({cfg.DATASET.NAME} -> {args.test_dataset}) ===")
    print(f"Overall MAE: {np.mean(abs_errors):.4f}")
    results_seasonal = {}
    for season, season_months in SEASONS.items():
        mask = np.isin(months, season_months)
        if mask.sum() > 0:
            smae = float(np.mean(abs_errors[mask]))
            print(f"  {season:<8} MAE: {smae:.4f}  (n={mask.sum()})")
            results_seasonal[season] = smae

    out = {
        "model": cfg.MODEL.NAME,
        "train_dataset": cfg.DATASET.NAME,
        "test_dataset": args.test_dataset,
        "overall_mae": float(np.mean(abs_errors)),
        "seasonal": results_seasonal,
        "checkpoint": ckpt_path,
    }
    out_dir = os.path.join(REPO, "benchmark_runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"crossyear_{cfg.MODEL.NAME}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
