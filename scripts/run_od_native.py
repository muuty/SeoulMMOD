"""Unified trainer for OD-native baselines on SeoulMOD.

Loads SeoulMOD GU/Dong memmap, reshapes to [T, N, N, C], trains selected model
under T=H (input_len = output_len = horizon) protocol with multi-mode joint
input/output (6 channels), and reports MAE/RMSE/MAPE averaged over horizon.

Outputs test_metrics.json compatible with the BasicTS-style checkpoint layout.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO, 'od_baselines', 'MPGCN'))
sys.path.insert(0, os.path.join(REPO, "od_baselines", "ODCRN", "model", "ODCRN"))
sys.path.insert(0, os.path.join(REPO, "od_baselines", "ODMixer"))


def _import_models():
    import MPGCN as MPGCN_mod
    import GCN as GCN_mod
    import ODCRN as ODCRN_mod
    return MPGCN_mod, GCN_mod, ODCRN_mod


class ODSeqDataset(Dataset):
    def __init__(self, flow, indices, input_len, output_len, O_dyn_G=None, D_dyn_G=None, need_dyn=False):
        self.flow = flow  # tensor or np.memmap [T, N, N, C]
        self.indices = indices
        self.input_len = input_len
        self.output_len = output_len
        self.O_dyn_G = O_dyn_G
        self.D_dyn_G = D_dyn_G
        self.need_dyn = need_dyn

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.flow[i:i + self.input_len]
        y = self.flow[i + self.input_len:i + self.input_len + self.output_len]
        # (L, N, N, C) float32 tensor
        x = torch.from_numpy(np.array(x, dtype=np.float32))
        y = torch.from_numpy(np.array(y, dtype=np.float32))
        if self.need_dyn:
            key = (i + self.input_len) % self.O_dyn_G.shape[-1]  # hour of day
            O_G = self.O_dyn_G[:, :, key]
            D_G = self.D_dyn_G[:, :, key]
            return x, y, O_G, D_G
        return x, y


def build_adjacency(N):
    adj = np.ones((N, N), dtype=np.float32)
    deg = adj.sum(axis=1, keepdims=True)
    return adj / deg


def build_dynamic_graphs(od_data, train_end_idx, perceived_period=24):
    """Build periodic O/D cosine-similarity graphs over all modes summed.
    od_data: [T, N, N, C], return [N, N, period] for O and D similarity."""
    N = od_data.shape[1]
    # Use summed flow across modes to build similarity
    flow_sum = od_data[:train_end_idx].sum(axis=-1)  # [T, N, N]
    num_periods = train_end_idx // perceived_period
    flow_sum = flow_sum[:num_periods * perceived_period]
    O_dyn = np.zeros((N, N, perceived_period), dtype=np.float32)
    D_dyn = np.zeros((N, N, perceived_period), dtype=np.float32)
    for t in range(perceived_period):
        avg = flow_sum[t::perceived_period].mean(axis=0)  # [N, N]
        row_norm = np.linalg.norm(avg, axis=1, keepdims=True) + 1e-8
        col_norm = np.linalg.norm(avg, axis=0, keepdims=True) + 1e-8
        row_n = avg / row_norm
        col_n = avg / col_norm
        O_dyn[:, :, t] = row_n @ row_n.T
        D_dyn[:, :, t] = col_n.T @ col_n
    return O_dyn, D_dyn


def make_mpgcn(N, input_dim, output_dim, out_horizon, hidden_dim, K):
    """Wrapper MPGCN-style model that outputs (B, out_horizon, N, N, output_dim)."""
    from MPGCN import MPGCN as _MPGCN, BDGCN

    class MPGCNMulti(nn.Module):
        def __init__(self):
            super().__init__()
            self.M = 2
            self.K = K
            self.num_nodes = N
            self.lstm_hidden_dim = hidden_dim
            self.lstm_num_layers = 1
            self.gcn_num_layers = 3
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.out_horizon = out_horizon
            self.branches = nn.ModuleList()
            for m in range(self.M):
                branch = nn.ModuleDict()
                branch['temporal'] = nn.LSTM(
                    input_size=input_dim, hidden_size=hidden_dim,
                    num_layers=1, batch_first=True)
                branch['spatial'] = nn.ModuleList()
                for n in range(self.gcn_num_layers):
                    cur_in = hidden_dim if n == 0 else hidden_dim
                    branch['spatial'].append(
                        BDGCN(K=K, input_dim=cur_in, hidden_dim=hidden_dim,
                              use_bias=True, activation=nn.ReLU))
                branch['fc'] = nn.Sequential(
                    nn.Linear(hidden_dim, out_horizon * output_dim),
                )
                self.branches.append(branch)

        def init_hidden_list(self, batch_size, device):
            hidden = []
            for _ in range(self.M):
                h = (torch.zeros(self.lstm_num_layers, batch_size * (self.num_nodes ** 2),
                                 self.lstm_hidden_dim, device=device),
                     torch.zeros(self.lstm_num_layers, batch_size * (self.num_nodes ** 2),
                                 self.lstm_hidden_dim, device=device))
                hidden.append(h)
            return hidden

        def forward(self, x_seq, G_list):
            # x_seq: (B, T, N, N, C_in)
            B, T, No, Nd, C = x_seq.shape
            device = x_seq.device
            hidden_list = self.init_hidden_list(B, device)
            lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(B * No * Nd, T, C)
            branch_out = []
            for m in range(self.M):
                lstm_out, hidden_list[m] = self.branches[m]['temporal'](lstm_in, hidden_list[m])
                gcn_in = lstm_out[:, -1, :].reshape(B, No, Nd, self.lstm_hidden_dim)
                for n in range(self.gcn_num_layers):
                    gcn_in = self.branches[m]['spatial'][n](gcn_in, G_list[m])
                fc_out = self.branches[m]['fc'](gcn_in)  # (B, N, N, H*C_out)
                branch_out.append(fc_out)
            ensemble = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)
            # (B, N, N, H*C_out) -> (B, H, N, N, C_out)
            ensemble = ensemble.view(B, No, Nd, self.out_horizon, self.output_dim)
            ensemble = ensemble.permute(0, 3, 1, 2, 4).contiguous()
            return ensemble

    return MPGCNMulti()


def make_odcrn(N, input_dim, out_horizon, hidden_dim, K):
    import ODCRN as ODCRN_mod
    model = ODCRN_mod.ODCRN(
        num_nodes=N, K=K, input_dim=input_dim,
        hidden_dim=hidden_dim, out_horizon=out_horizon,
        num_layers=2, DGCbool=True, use_bias=True, activation=None)
    return model


def make_odmixer(N, input_dim, output_dim, out_horizon, in_steps,
                 hidden_dim, layer_nums=2, dropout=0.1):
    """Wrapper that adapts ODMixer to (B, T, N, N, M) -> (B, N, N, H, M).

    M=input_dim mode channels are folded along the time axis: ODMixer's
    input_seq becomes T*M and out_steps becomes H*M, then reshape back.
    ODMixer's bidirectional trend learning module needs a `prev_od`
    historical window, which our protocol does not supply; we pass
    zeros so the per-baseline comparison is apples-to-apples. ODMixer
    does not require any graph input.
    """
    import torch as _torch
    from odmixer_arch import ODMixerBackbone

    class ODMixerMulti(nn.Module):
        def __init__(self):
            super().__init__()
            self.N = N
            self.M = input_dim
            self.H = out_horizon
            self.T = in_steps
            self.backbone = ODMixerBackbone(
                num_nodes=N,
                input_seq=in_steps * input_dim,
                hid_dim=hidden_dim,
                layer_nums=layer_nums,
                dropout=dropout,
                out_steps=out_horizon * output_dim,
            )

        def forward(self, x_seq):
            # x_seq: (B, T, N, N, M)
            B, T, _, _, M = x_seq.shape
            # Stack mode along time -> (B, T*M, N, N).
            x = x_seq.permute(0, 1, 4, 2, 3).reshape(B, T * M, self.N, self.N)
            prev = _torch.zeros_like(x)
            od_out, _ = self.backbone(x, prev)            # (B, H*M, N, N)
            # (B, H*M, N, N) -> (B, H, M, N, N) -> (B, H, N, N, M)
            return od_out.reshape(B, self.H, M, self.N, self.N).permute(0, 1, 3, 4, 2)

    return ODMixerMulti()


def compute_metrics(pred, target):
    """MAE, RMSE, MAPE on full tensor (averaged over all elements and horizon steps)."""
    diff = np.abs(pred - target)
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    mask = target > 1.0  # flow > 1 — skip near-zero
    if mask.any():
        mape = float((diff[mask] / target[mask]).mean() * 100.0)
    else:
        mape = float('nan')
    return mae, rmse, mape


def compute_metrics_per_horizon(pred, target):
    """Per-horizon metrics: pred/target shape [N_samples, H, ...]."""
    out = {}
    H = pred.shape[1]
    for h in range(H):
        p = pred[:, h]
        t = target[:, h]
        mae, rmse, mape = compute_metrics(p, t)
        out[f'h{h+1}'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['odcrn', 'mpgcn', 'odmixer'], required=True)
    p.add_argument('--data_dir', type=str,
                   default=os.path.join(REPO, 'datasets', 'SeoulMOD_GU_2024'))
    p.add_argument('--level', type=str, choices=['gu', 'dong'], default='gu')
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--input_len', type=int, default=None,
                   help='Defaults to horizon (T=H protocol)')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--hidden_dim', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--K_cheby', type=int, default=2,
                   help='random_walk_diffusion order; final K = order+1')
    p.add_argument('--output_root', type=str,
                   default=os.path.join(REPO, 'checkpoints'))
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--log_transform', action='store_true', default=True)
    p.add_argument('--no_log_transform', dest='log_transform', action='store_false')
    p.add_argument('--max_train_samples', type=int, default=None)
    args = p.parse_args()

    if args.input_len is None:
        args.input_len = args.horizon

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # Load data
    with open(os.path.join(args.data_dir, 'desc.json')) as f:
        desc = json.load(f)
    shape = tuple(desc['shape'])
    T_total, num_pairs, C_total = shape
    N = int(np.sqrt(num_pairs))
    assert N * N == num_pairs, f'num_pairs {num_pairs} not a square'
    print(f'[data] shape={shape}, N={N}, T_total={T_total}')

    raw = np.memmap(os.path.join(args.data_dir, 'data.dat'),
                    dtype='float32', mode='r', shape=shape)
    # 6 flow modes (indices 0..5)
    C_flow = 6
    flow_flat = np.array(raw[:, :, :C_flow], dtype=np.float32)  # [T, N*N, 6]
    flow = flow_flat.reshape(T_total, N, N, C_flow)

    # Normalize: ZScore on training portion
    T_in, T_out = args.input_len, args.horizon
    total_seq = T_total - T_in - T_out + 1
    train_end_seq = int(total_seq * 0.7)
    val_end_seq = int(total_seq * 0.8)
    print(f'[split] total_seq={total_seq}, train={train_end_seq}, val={val_end_seq - train_end_seq}, test={total_seq - val_end_seq}')

    train_end_t = train_end_seq + T_in  # last training frame
    train_data = flow[:train_end_t]
    if args.log_transform:
        flow_proc = np.log1p(flow)
        train_proc = np.log1p(train_data)
    else:
        flow_proc = flow.copy()
        train_proc = train_data

    mean = train_proc.mean(axis=(0, 1, 2), keepdims=True)  # per-channel mean
    std = train_proc.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    flow_norm = (flow_proc - mean) / std
    print(f'[norm] mean={mean.squeeze()}, std={std.squeeze()}')

    # Train/val/test index splits
    train_idx = np.arange(0, train_end_seq)
    val_idx = np.arange(train_end_seq, val_end_seq)
    test_idx = np.arange(val_end_seq, total_seq)
    if args.max_train_samples is not None and len(train_idx) > args.max_train_samples:
        train_idx = train_idx[:args.max_train_samples]

    # Static adjacency (uniform, row-normalized)
    MPGCN_mod, GCN_mod, ODCRN_mod = _import_models()
    adj = build_adjacency(N)
    adj_processor = GCN_mod.Adj_Processor('random_walk_diffusion', args.K_cheby)
    K_final = args.K_cheby + 1
    static_G = adj_processor.process(
        torch.from_numpy(adj).float().unsqueeze(0)).squeeze(0).to(device)  # (K, N, N)

    need_dyn = (args.model == 'mpgcn')
    O_dyn_G, D_dyn_G = None, None
    if need_dyn:
        print('[dyn] building dynamic O/D graphs...')
        t0 = time.time()
        O_dyn_G, D_dyn_G = build_dynamic_graphs(flow, train_end_t, perceived_period=24)
        print(f'[dyn] done in {time.time()-t0:.1f}s')

    # Datasets
    train_ds = ODSeqDataset(flow_norm, train_idx, T_in, T_out,
                            O_dyn_G, D_dyn_G, need_dyn=need_dyn)
    val_ds = ODSeqDataset(flow_norm, val_idx, T_in, T_out,
                          O_dyn_G, D_dyn_G, need_dyn=need_dyn)
    test_ds = ODSeqDataset(flow_norm, test_idx, T_in, T_out,
                           O_dyn_G, D_dyn_G, need_dyn=need_dyn)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f'[dataset] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}')

    # Model
    if args.model == 'odcrn':
        model = make_odcrn(N=N, input_dim=C_flow, out_horizon=T_out,
                           hidden_dim=args.hidden_dim, K=K_final).to(device)
    elif args.model == 'odmixer':
        model = make_odmixer(N=N, input_dim=C_flow, output_dim=C_flow,
                             out_horizon=T_out, in_steps=T_in,
                             hidden_dim=args.hidden_dim).to(device)
    else:
        model = make_mpgcn(N=N, input_dim=C_flow, output_dim=C_flow, out_horizon=T_out,
                           hidden_dim=args.hidden_dim, K=K_final).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f'[model] {args.model}, params={n_params}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()  # MAE

    def run_batch(batch, train_mode):
        if need_dyn:
            x, y, O_G, D_G = batch
            # Adj_Processor uses torch.eye on CPU — keep O_G/D_G on CPU, then move
            O_proc = adj_processor.process(O_G).to(device, non_blocking=True)
            D_proc = adj_processor.process(D_G).to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            dyn_G = (O_proc, D_proc)
            pred = model(x_seq=x, G_list=[static_G, dyn_G])
        else:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.model == 'odmixer':
                pred = model(x_seq=x)
            else:
                pred = model(G=static_G, X_seq=x)
        return pred, y

    # Train loop
    best_val = float('inf')
    best_state = None
    patience = args.patience
    run_name = args.run_name or f'{args.model}_allmode_2024_{args.level}'
    # BasicTS-style output dir
    subdir_name = f'SeoulMOD_{"GU_" if args.level == "gu" else ""}2024_{args.epochs}_{T_in}_{T_out}'
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    out_dir = os.path.join(args.output_root, run_name, subdir_name, ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f'[out_dir] {out_dir}')

    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(out_dir, 'train.log')
    log_f = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f'[start] {datetime.now().isoformat()}')
    t_train0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            pred, y = run_batch(batch, True)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            tr_loss += loss.item() * y.shape[0]
            tr_n += y.shape[0]
        tr_loss /= max(tr_n, 1)

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                pred, y = run_batch(batch, False)
                va_loss += loss_fn(pred, y).item() * y.shape[0]
                va_n += y.shape[0]
        va_loss /= max(va_n, 1)
        elapsed = time.time() - t0
        improved = va_loss < best_val
        marker = '*' if improved else ' '
        log(f'  Epoch {epoch:3d}: train={tr_loss:.5f}, val={va_loss:.5f} ({elapsed:.1f}s) {marker}')
        if improved:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                log(f'  Early stop at epoch {epoch}')
                break
    log(f'[train_done] total {time.time()-t_train0:.1f}s')

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        ckpt_path = os.path.join(out_dir, f'{args.model}_best.pt')
        torch.save({'model_state_dict': best_state,
                    'best_val_loss': float(best_val),
                    'args': vars(args)}, ckpt_path)
        log(f'[ckpt_saved] {ckpt_path}')

    # Test
    model.eval()
    all_pred, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred, y = run_batch(batch, False)
            all_pred.append(pred.cpu().numpy())
            all_y.append(y.cpu().numpy())
    pred_norm = np.concatenate(all_pred, axis=0)  # [Nt, H, N, N, C]
    y_norm = np.concatenate(all_y, axis=0)

    # Inverse transform
    pred_unnorm = pred_norm * std + mean
    y_unnorm = y_norm * std + mean
    if args.log_transform:
        pred_raw = np.expm1(pred_unnorm)
        y_raw = np.expm1(y_unnorm)
    else:
        pred_raw = pred_unnorm
        y_raw = y_unnorm
    pred_raw = np.clip(pred_raw, 0, None)

    # Metrics: overall + per-horizon + per-mode + overall WAPE
    overall_mae, overall_rmse, overall_mape = compute_metrics(pred_raw, y_raw)
    per_h = compute_metrics_per_horizon(pred_raw, y_raw)
    overall_wape = float(np.abs(pred_raw - y_raw).sum() / (np.abs(y_raw).sum() + 1e-8))

    # Per-mode: last axis is C_flow=6
    mode_names = ['metro_bus', 'local_bus', 'subway', 'walk', 'car', 'other']
    per_mode = {}
    for mi, name in enumerate(mode_names):
        p = pred_raw[..., mi]
        t = y_raw[..., mi]
        mae_m = float(np.abs(p - t).mean())
        rmse_m = float(np.sqrt(((p - t) ** 2).mean()))
        wape_m = float(np.abs(p - t).sum() / (np.abs(t).sum() + 1e-8))
        per_mode[name] = {'MAE': mae_m, 'RMSE': rmse_m, 'WAPE': wape_m}

    log(f'[test] Overall: MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, '
        f'WAPE={overall_wape*100:.2f}%, MAPE={overall_mape:.4f}')
    for h in range(T_out):
        m = per_h[f'h{h+1}']
        log(f'[test] h{h+1}: MAE={m["MAE"]:.4f}, RMSE={m["RMSE"]:.4f}, MAPE={m["MAPE"]:.4f}')
    for name in mode_names:
        pm = per_mode[name]
        log(f'[test] mode {name:<10} MAE={pm["MAE"]:>8.4f} RMSE={pm["RMSE"]:>8.4f} WAPE={pm["WAPE"]*100:>6.2f}%')

    metrics_out = {
        'overall': {'MAE': overall_mae, 'RMSE': overall_rmse,
                    'WAPE': overall_wape, 'MAPE': overall_mape},
        'per_horizon': per_h,
        'per_mode': per_mode,
        'horizon': T_out,
        'input_len': T_in,
        'model': args.model,
        'level': args.level,
    }
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)
    log(f'[metrics_saved] {out_dir}/test_metrics.json')
    log_f.close()


if __name__ == '__main__':
    main()
