"""Cross-year evaluation: load 2024-trained model, test on 2025 by season."""
import os, sys, json, argparse, torch, numpy as np
from torch.utils.data import DataLoader

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

SEASONS = {
    'spring': (3, 4, 5),
    'summer': (6, 7, 8),
    'fall': (9, 10, 11),
    'winter': (12, 1, 2),
}

def load_config(path):
    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.').replace('\\', '.')
    cfg_name = path.split('.')[-1]
    return __import__(path, fromlist=[cfg_name]).CFG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cfg", required=True, help="2024 training config")
    ap.add_argument("--test-dataset", required=True, help="2025 dataset name")
    ap.add_argument("--gpus", default="0")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.cfg)

    # Scaler from 2024 training
    scaler = cfg.SCALER.TYPE(**dict(cfg.SCALER.PARAM))
    is_multi = hasattr(scaler, '_multi') and scaler._multi
    if is_multi:
        means = scaler.mean.numpy()
        stds = scaler.std.numpy()
        channels = scaler._channels
    else:
        means = np.array([scaler.mean.item()])
        stds = np.array([scaler.std.item()])
        channels = [scaler.target_channel]

    # Load 2025 test dataset (full year)
    from basicts.data import TimeSeriesForecastingDataset
    test_desc_path = f'datasets/{args.test_dataset}/desc.json'
    with open(test_desc_path) as f:
        test_desc = json.load(f)

    # Use ALL of 2025 as test (ratio trick: train=0, val=0, test=1)
    # Actually we need valid ratio structure. Use mode='test' with full data.
    # Simpler: load memmap directly
    test_shape = tuple(test_desc['shape'])
    test_data = np.memmap(f'datasets/{args.test_dataset}/data.dat', dtype='float32', mode='r', shape=test_shape)
    T, N, C = test_shape

    INPUT_LEN = cfg.DATASET.PARAM.get('input_len', 24)
    OUTPUT_LEN = cfg.DATASET.PARAM.get('output_len', 24)
    forward_features = cfg.MODEL.get('FORWARD_FEATURES', [0])
    target_features = cfg.MODEL.get('TARGET_FEATURES', [0])

    # Find checkpoint
    ckpt_dir = cfg.TRAIN.CKPT_SAVE_DIR
    subdirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
    subdir = sorted(subdirs)[-1]
    ckpt_path = os.path.join(ckpt_dir, subdir, f"{cfg.MODEL.NAME}_best_val_MAE.pt")

    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"Model: {cfg.MODEL.NAME}", file=sys.stderr)
    print(f"Checkpoint: {ckpt_path}", file=sys.stderr)
    print(f"Test dataset: {args.test_dataset} {test_shape}", file=sys.stderr)

    # Compute start date
    from datetime import datetime, timedelta
    start_date = datetime.strptime(test_desc.get('start_date', '2025-01-01'), '%Y-%m-%d')

    # Sliding window inference
    all_maes = []  # list of (t_idx, mae_per_sample)
    all_timestamps = []

    n_samples = T - INPUT_LEN - OUTPUT_LEN + 1
    BATCH = 32

    with torch.no_grad():
        for start in range(0, n_samples, BATCH):
            end = min(start + BATCH, n_samples)
            batch_x = []
            batch_y = []
            for t in range(start, end):
                x = test_data[t:t+INPUT_LEN, :, :]  # [IL, N, C]
                y = test_data[t+INPUT_LEN:t+INPUT_LEN+OUTPUT_LEN, :, :]  # [OL, N, C]
                batch_x.append(x)
                batch_y.append(y)

            batch_x = torch.tensor(np.array(batch_x), dtype=torch.float32)  # [B, IL, N, C]
            batch_y = np.array(batch_y)  # [B, OL, N, C]

            # Select features + normalize
            x_sel = batch_x[..., forward_features].clone()
            for i, ch in enumerate(channels):
                if ch in forward_features:
                    pos = forward_features.index(ch)
                    x_sel[..., pos] = (x_sel[..., pos] - means[i]) / stds[i]

            # Create dummy future_data for xformer models
            fut_sel = torch.zeros(x_sel.shape[0], OUTPUT_LEN, x_sel.shape[2], x_sel.shape[3], dtype=torch.float32)
            pred = model(history_data=x_sel.to(device), future_data=fut_sel.to(device),
                        batch_seen=None, epoch=None, train=False)
            if isinstance(pred, dict):
                pred = pred.get('prediction', list(pred.values())[0])
            pred = pred.cpu().numpy()

            # Inverse transform
            y_target = batch_y[..., target_features]
            for i in range(pred.shape[-1]):
                idx = min(i, len(means)-1)
                pred[..., i] = pred[..., i] * stds[idx] + means[idx]

            mae = np.mean(np.abs(pred - y_target), axis=(1,2,3))  # per sample
            for j, t in enumerate(range(start, end)):
                dt = start_date + timedelta(hours=t + INPUT_LEN)
                all_timestamps.append(dt)
                all_maes.append(mae[j])

            if (start // BATCH) % 100 == 0:
                print(f"  {start}/{n_samples}", file=sys.stderr)

    all_maes = np.array(all_maes)
    months = np.array([dt.month for dt in all_timestamps])

    print(f"\n=== Cross-year results: {cfg.MODEL.NAME} on {args.test_dataset} ===")
    print(f"Overall MAE: {np.mean(all_maes):.3f}")
    for season, season_months in SEASONS.items():
        mask = np.isin(months, season_months)
        if mask.sum() > 0:
            print(f"  {season:<8} MAE: {np.mean(all_maes[mask]):.3f}  (n={mask.sum()})")

    # Save
    results = {
        'model': cfg.MODEL.NAME,
        'train_dataset': cfg.DATASET.NAME,
        'test_dataset': args.test_dataset,
        'overall_mae': float(np.mean(all_maes)),
        'seasonal': {s: float(np.mean(all_maes[np.isin(months, m)])) for s, m in SEASONS.items()},
    }
    out_dir = os.path.join(REPO, 'benchmark_runs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'crossyear_{cfg.MODEL.NAME}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
