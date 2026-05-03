"""Unified trainer + evaluator for OD-native baselines on SeoulMMOD.

Loads the BasicTS-style memmap, normalises with per-channel ZScore on a
log-transformed training portion, trains the selected model under the
T=H protocol with multi-mode joint input/output (6 channels), and runs
a streaming test pass that computes overall, per-horizon, and per-mode
MAE / RMSE / wMAPE / MAPE without holding the full prediction tensor in
memory. Outputs `test_metrics.json` next to the checkpoint.

Usage
-----
Train + test:
    python scripts/run_od_native.py --model odmixer --level subdistrict \\
        --horizon 24 --data_dir datasets/SeoulMMOD_Subdistrict_2024 --gpu 0

Test-only on a saved checkpoint (replaces the old eval_odmixer_dong.py):
    python scripts/run_od_native.py --eval_only --ckpt_dir <out_dir> --gpu 0
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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO, 'od_baselines', 'MPGCN'))
sys.path.insert(0, os.path.join(REPO, 'od_baselines', 'ODCRN', 'model', 'ODCRN'))
sys.path.insert(0, os.path.join(REPO, 'od_baselines', 'ODMixer'))


# ============================================================================
# Constants
# ============================================================================

C_FLOW = 6  # 6 transport modes (channels in data.dat[..., :6])
MODE_NAMES = ['metro_bus', 'local_bus', 'subway', 'walk', 'car', 'other']


# ============================================================================
# Dataset
# ============================================================================

class ODSeqDataset(Dataset):
    """Sliding-window OD sequences with optional periodic graph index."""

    def __init__(self, flow, indices, input_len, output_len,
                 O_dyn_G=None, D_dyn_G=None, need_dyn=False):
        self.flow = flow                  # [T, N, N, C]
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
        x = torch.from_numpy(np.array(
            self.flow[i:i + self.input_len], dtype=np.float32))
        y = torch.from_numpy(np.array(
            self.flow[i + self.input_len:i + self.input_len + self.output_len],
            dtype=np.float32))
        if self.need_dyn:
            key = (i + self.input_len) % self.O_dyn_G.shape[-1]
            return x, y, self.O_dyn_G[:, :, key], self.D_dyn_G[:, :, key]
        return x, y


# ============================================================================
# Data pipeline
# ============================================================================

def load_flow(data_dir):
    """Memmap [T, N*N, C] -> ndarray [T, N, N, C_FLOW]."""
    with open(os.path.join(data_dir, 'desc.json')) as f:
        desc = json.load(f)
    shape = tuple(desc['shape'])
    T_total, num_pairs, _ = shape
    N = int(np.sqrt(num_pairs))
    assert N * N == num_pairs, f'num_pairs {num_pairs} not a square'
    raw = np.memmap(os.path.join(data_dir, 'data.dat'),
                    dtype='float32', mode='r', shape=shape)
    flow = np.array(raw[:, :, :C_FLOW], dtype=np.float32).reshape(
        T_total, N, N, C_FLOW)
    return flow, T_total, N


def make_splits(T_total, T_in, T_out, ratios=(0.7, 0.1, 0.2)):
    """Sequence-level train/val/test indices and the last training frame."""
    total_seq = T_total - T_in - T_out + 1
    train_end_seq = int(total_seq * ratios[0])
    val_end_seq = int(total_seq * (ratios[0] + ratios[1]))
    train_idx = np.arange(0, train_end_seq)
    val_idx = np.arange(train_end_seq, val_end_seq)
    test_idx = np.arange(val_end_seq, total_seq)
    train_end_t = train_end_seq + T_in
    return train_idx, val_idx, test_idx, train_end_t


def normalize(flow, train_end_t, log_transform):
    """Per-channel ZScore on the training portion (optionally log1p first)."""
    train_data = flow[:train_end_t]
    if log_transform:
        flow_proc = np.log1p(flow)
        train_proc = np.log1p(train_data)
    else:
        flow_proc = flow.copy()
        train_proc = train_data
    mean = train_proc.mean(axis=(0, 1, 2), keepdims=True)
    std = train_proc.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    flow_norm = (flow_proc - mean) / std
    return flow_norm, mean, std


def build_dynamic_graphs(od_data, train_end_t, perceived_period=24):
    """Periodic O/D cosine-similarity graphs over modes summed.

    Returns (O_dyn, D_dyn) with shape [N, N, period].
    """
    N = od_data.shape[1]
    flow_sum = od_data[:train_end_t].sum(axis=-1)            # [T, N, N]
    num_periods = train_end_t // perceived_period
    flow_sum = flow_sum[:num_periods * perceived_period]
    O_dyn = np.zeros((N, N, perceived_period), dtype=np.float32)
    D_dyn = np.zeros((N, N, perceived_period), dtype=np.float32)
    for t in range(perceived_period):
        avg = flow_sum[t::perceived_period].mean(axis=0)     # [N, N]
        row_norm = np.linalg.norm(avg, axis=1, keepdims=True) + 1e-8
        col_norm = np.linalg.norm(avg, axis=0, keepdims=True) + 1e-8
        row_n = avg / row_norm
        col_n = avg / col_norm
        O_dyn[:, :, t] = row_n @ row_n.T
        D_dyn[:, :, t] = col_n.T @ col_n
    return O_dyn, D_dyn


def make_dataloaders(flow_norm, splits, T_in, T_out, batch_size,
                     dyn_graphs, num_workers=2):
    train_idx, val_idx, test_idx = splits
    O_G, D_G = dyn_graphs if dyn_graphs is not None else (None, None)
    need_dyn = dyn_graphs is not None

    def _ds(idx):
        return ODSeqDataset(flow_norm, idx, T_in, T_out,
                            O_G, D_G, need_dyn=need_dyn)

    train_loader = DataLoader(_ds(train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(_ds(val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(_ds(test_idx), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ============================================================================
# Adjacency
# ============================================================================

def process_static_adjacency(N, K_cheby, device):
    """Uniform row-normalised adjacency, processed by the MPGCN/ODCRN
    Adj_Processor. Returns (processor, static_G_on_device, K_final)."""
    import GCN
    adj = np.ones((N, N), dtype=np.float32)
    adj = adj / adj.sum(axis=1, keepdims=True)
    proc = GCN.Adj_Processor('random_walk_diffusion', K_cheby)
    static_G = proc.process(
        torch.from_numpy(adj).float().unsqueeze(0)).squeeze(0).to(device)
    K_final = K_cheby + 1
    return proc, static_G, K_final


# ============================================================================
# Models
# ============================================================================

class MPGCNMulti(nn.Module):
    """MPGCN with two branches (static + dynamic graph) producing
    (B, out_horizon, N, N, output_dim)."""

    def __init__(self, N, input_dim, output_dim, out_horizon, hidden_dim, K):
        super().__init__()
        from MPGCN import BDGCN
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
        for _ in range(self.M):
            branch = nn.ModuleDict()
            branch['temporal'] = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=1, batch_first=True)
            branch['spatial'] = nn.ModuleList()
            for _ in range(self.gcn_num_layers):
                branch['spatial'].append(
                    BDGCN(K=K, input_dim=hidden_dim, hidden_dim=hidden_dim,
                          use_bias=True, activation=nn.ReLU))
            branch['fc'] = nn.Sequential(
                nn.Linear(hidden_dim, out_horizon * output_dim))
            self.branches.append(branch)

    def _init_hidden(self, batch_size, device):
        hidden = []
        for _ in range(self.M):
            h = (torch.zeros(self.lstm_num_layers,
                             batch_size * (self.num_nodes ** 2),
                             self.lstm_hidden_dim, device=device),
                 torch.zeros(self.lstm_num_layers,
                             batch_size * (self.num_nodes ** 2),
                             self.lstm_hidden_dim, device=device))
            hidden.append(h)
        return hidden

    def forward(self, x_seq, G_list):
        B, T, No, Nd, C = x_seq.shape
        hidden_list = self._init_hidden(B, x_seq.device)
        lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(B * No * Nd, T, C)
        branch_out = []
        for m in range(self.M):
            lstm_out, hidden_list[m] = self.branches[m]['temporal'](
                lstm_in, hidden_list[m])
            gcn_in = lstm_out[:, -1, :].reshape(
                B, No, Nd, self.lstm_hidden_dim)
            for n in range(self.gcn_num_layers):
                gcn_in = self.branches[m]['spatial'][n](gcn_in, G_list[m])
            branch_out.append(self.branches[m]['fc'](gcn_in))
        ensemble = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)
        ensemble = ensemble.view(B, No, Nd, self.out_horizon, self.output_dim)
        return ensemble.permute(0, 3, 1, 2, 4).contiguous()


class ODMixerMulti(nn.Module):
    """ODMixer adapted to (B, T, N, N, M) -> (B, H, N, N, M).

    M mode channels are folded along the time axis. Bidirectional trend
    learning needs `prev_od`; we pass zeros so the per-baseline comparison
    is apples-to-apples. ODMixer needs no graph input.
    """

    def __init__(self, N, input_dim, output_dim, out_horizon, in_steps,
                 hidden_dim, layer_nums=2, dropout=0.1):
        super().__init__()
        from odmixer_arch import ODMixerBackbone
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
        B, T, _, _, M = x_seq.shape
        x = x_seq.permute(0, 1, 4, 2, 3).reshape(B, T * M, self.N, self.N)
        prev = torch.zeros_like(x)
        od_out, _ = self.backbone(x, prev)
        return od_out.reshape(B, self.H, M, self.N, self.N).permute(
            0, 1, 3, 4, 2)


def build_model(model_name, N, T_in, T_out, hidden_dim, K_final):
    if model_name == 'odcrn':
        import ODCRN
        return ODCRN.ODCRN(
            num_nodes=N, K=K_final, input_dim=C_FLOW,
            hidden_dim=hidden_dim, out_horizon=T_out,
            num_layers=2, DGCbool=True, use_bias=True, activation=None)
    if model_name == 'odmixer':
        return ODMixerMulti(N, C_FLOW, C_FLOW, T_out, T_in, hidden_dim)
    if model_name == 'mpgcn':
        return MPGCNMulti(N, C_FLOW, C_FLOW, T_out, hidden_dim, K_final)
    raise ValueError(f'unknown model {model_name}')


def model_forward(model, batch, *, model_name, static_G, adj_processor,
                  device, need_dyn):
    """Unified forward dispatch returning (pred, y) on `device`."""
    if need_dyn:
        x, y, O_G, D_G = batch
        O_proc = adj_processor.process(O_G).to(device, non_blocking=True)
        D_proc = adj_processor.process(D_G).to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x_seq=x, G_list=[static_G, (O_proc, D_proc)])
    else:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if model_name == 'odmixer':
            pred = model(x_seq=x)
        else:
            pred = model(G=static_G, X_seq=x)
    return pred, y


# ============================================================================
# Streaming metrics
# ============================================================================

class StreamingMetrics:
    """Accumulates MAE / RMSE / wMAPE / MAPE for overall, per-horizon, and
    per-mode slices without storing tensors. Call `update` per batch with
    raw-scale (denormalised) numpy arrays."""

    def __init__(self, H, M, mode_names):
        self.H = H
        self.M = M
        self.mode_names = mode_names
        self.overall = self._zeros()
        self.per_h = [self._zeros() for _ in range(H)]
        self.per_mode = [self._zeros() for _ in range(M)]

    @staticmethod
    def _zeros():
        return {'sum_abs': 0.0, 'sum_sq': 0.0, 'sum_t_abs': 0.0,
                'n': 0, 'sum_pct': 0.0, 'pct_n': 0}

    @staticmethod
    def _accum(slot, diff, sq, t_abs, mask, t):
        slot['sum_abs'] += float(diff.sum())
        slot['sum_sq'] += float(sq.sum())
        slot['sum_t_abs'] += float(t_abs.sum())
        slot['n'] += int(diff.size)
        if mask.any():
            slot['sum_pct'] += float((diff[mask] / t[mask]).sum())
            slot['pct_n'] += int(mask.sum())

    def update(self, pred_raw, y_raw):
        diff = np.abs(pred_raw - y_raw)
        sq = diff * diff
        t_abs = np.abs(y_raw)
        mask = y_raw > 1.0  # skip near-zero entries for MAPE
        self._accum(self.overall, diff, sq, t_abs, mask, y_raw)
        for h in range(self.H):
            self._accum(self.per_h[h],
                        diff[:, h], sq[:, h], t_abs[:, h],
                        mask[:, h], y_raw[:, h])
        for mi in range(self.M):
            self._accum(self.per_mode[mi],
                        diff[..., mi], sq[..., mi], t_abs[..., mi],
                        mask[..., mi], y_raw[..., mi])

    @staticmethod
    def _summary(slot, include_wmape=True):
        out = {
            'MAE': slot['sum_abs'] / slot['n'],
            'RMSE': float(np.sqrt(slot['sum_sq'] / slot['n'])),
            'MAPE': (slot['sum_pct'] / slot['pct_n'] * 100.0)
                    if slot['pct_n'] > 0 else float('nan'),
        }
        if include_wmape:
            out['wMAPE'] = slot['sum_abs'] / (slot['sum_t_abs'] + 1e-8)
        return out

    def finalize(self):
        return {
            'overall': self._summary(self.overall),
            'per_horizon': {f'h{h+1}': self._summary(self.per_h[h],
                                                     include_wmape=False)
                            for h in range(self.H)},
            'per_mode': {self.mode_names[mi]: self._summary(self.per_mode[mi])
                         for mi in range(self.M)},
        }


def denormalize(pred_norm, y_norm, mean, std, log_transform):
    pred = pred_norm * std + mean
    y = y_norm * std + mean
    if log_transform:
        pred = np.expm1(pred)
        y = np.expm1(y)
    return np.clip(pred, 0, None), y


# ============================================================================
# Training / testing pipelines
# ============================================================================

def run_one_pass(model, loader, *, training, optimizer, loss_fn,
                 forward_kwargs, clip_norm=None):
    model.train(training)
    total_loss = 0.0
    total_n = 0
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            if training:
                optimizer.zero_grad()
            pred, y = model_forward(model, batch, **forward_kwargs)
            loss = loss_fn(pred, y)
            if training:
                loss.backward()
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            total_loss += loss.item() * y.shape[0]
            total_n += y.shape[0]
    return total_loss / max(total_n, 1)


def fit(model, train_loader, val_loader, *, epochs, patience, lr,
        forward_kwargs, log):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    best_val = float('inf')
    best_state = None
    pat = patience
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        tr = run_one_pass(model, train_loader, training=True,
                          optimizer=optimizer, loss_fn=loss_fn,
                          forward_kwargs=forward_kwargs, clip_norm=3.0)
        va = run_one_pass(model, val_loader, training=False,
                          optimizer=optimizer, loss_fn=loss_fn,
                          forward_kwargs=forward_kwargs)
        improved = va < best_val
        log(f'  Epoch {epoch:3d}: train={tr:.5f}, val={va:.5f} '
            f'({time.time()-t_ep:.1f}s) {"*" if improved else " "}')
        if improved:
            best_val = va
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            pat = patience
        else:
            pat -= 1
            if pat <= 0:
                log(f'  Early stop at epoch {epoch}')
                break
    log(f'[train_done] total {time.time()-t0:.1f}s')
    return best_val, best_state


def streaming_test(model, test_loader, *, mean, std, log_transform,
                   T_out, forward_kwargs, log):
    metrics = StreamingMetrics(H=T_out, M=C_FLOW, mode_names=MODE_NAMES)
    model.eval()
    n_batches = len(test_loader)
    log_every = max(1, n_batches // 10)
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            pred, y = model_forward(model, batch, **forward_kwargs)
            pred_raw, y_raw = denormalize(
                pred.cpu().numpy(), y.cpu().numpy(), mean, std, log_transform)
            metrics.update(pred_raw, y_raw)
            if (bi + 1) % log_every == 0 or bi == n_batches - 1:
                el = time.time() - t0
                eta = el / (bi + 1) * (n_batches - bi - 1)
                log(f'[test] batch {bi+1}/{n_batches}  '
                    f'elapsed={el:.1f}s  eta={eta:.1f}s')
    log(f'[test_pass] {time.time()-t0:.1f}s')
    return metrics.finalize()


def log_test_summary(metrics, T_out, log):
    o = metrics['overall']
    log(f'[test] Overall: MAE={o["MAE"]:.4f}, RMSE={o["RMSE"]:.4f}, '
        f'wMAPE={o["wMAPE"]*100:.2f}%, MAPE={o["MAPE"]:.4f}')
    for h in range(T_out):
        m = metrics['per_horizon'][f'h{h+1}']
        log(f'[test] h{h+1}: MAE={m["MAE"]:.4f}, '
            f'RMSE={m["RMSE"]:.4f}, MAPE={m["MAPE"]:.4f}')
    for name in MODE_NAMES:
        pm = metrics['per_mode'][name]
        log(f'[test] mode {name:<10} MAE={pm["MAE"]:>8.4f} '
            f'RMSE={pm["RMSE"]:>8.4f} wMAPE={pm["wMAPE"]*100:>6.2f}%')


# ============================================================================
# Output dir / logging
# ============================================================================

def setup_output_dir(args, out_dir=None):
    """Create out_dir and return (out_dir, log_fn, log_file)."""
    if out_dir is None:
        run_name = args.run_name or f'{args.model}_allmode_2024_{args.level}'
        dataset_tag = 'District' if args.level == 'district' else 'Subdistrict'
        T_in = args.input_len or args.horizon
        subdir = f'SeoulMMOD_{dataset_tag}_2024_{args.epochs}_{T_in}_{args.horizon}'
        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        out_dir = os.path.join(args.output_root, run_name, subdir, ts)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, 'train.log')
    log_file = open(log_path, 'a')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    return out_dir, log, log_file


def save_metrics(metrics, T_out, T_in, model_name, level, out_dir):
    metrics_out = {
        **metrics,
        'horizon': T_out,
        'input_len': T_in,
        'model': model_name,
        'level': level,
    }
    out_path = os.path.join(out_dir, 'test_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    return out_path


# ============================================================================
# Main pipelines
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['odcrn', 'mpgcn', 'odmixer'])
    p.add_argument('--data_dir', type=str,
                   default=os.path.join(REPO, 'datasets', 'SeoulMMOD_District_2024'))
    p.add_argument('--level', type=str, choices=['district', 'subdistrict'],
                   default='district')
    p.add_argument('--horizon', type=int)
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
    p.add_argument('--no_log_transform', dest='log_transform',
                   action='store_false')
    p.add_argument('--max_train_samples', type=int, default=None)
    # Test-only mode
    p.add_argument('--eval_only', action='store_true',
                   help='Skip training; load --ckpt_dir/{model}_best.pt + '
                        'args.json and run streaming test only')
    p.add_argument('--ckpt_dir', type=str, default=None,
                   help='Required with --eval_only')
    args = p.parse_args()

    if args.eval_only:
        if not args.ckpt_dir:
            p.error('--eval_only requires --ckpt_dir')
        # Inflate args from saved args.json
        with open(os.path.join(args.ckpt_dir, 'args.json')) as f:
            saved = json.load(f)
        gpu_override = args.gpu
        for k, v in saved.items():
            if not hasattr(args, k) or k in ('gpu', 'eval_only', 'ckpt_dir'):
                continue
            setattr(args, k, v)
        args.gpu = gpu_override
    else:
        if not args.model or args.horizon is None:
            p.error('--model and --horizon are required for training')

    if args.input_len is None:
        args.input_len = args.horizon
    return args


def run_pipeline(args):
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    T_in, T_out = args.input_len, args.horizon
    out_dir = args.ckpt_dir if args.eval_only else None
    out_dir, log, log_file = setup_output_dir(args, out_dir=out_dir)
    try:
        log(f'[start] {datetime.now().isoformat()}  device={device}  '
            f'mode={"eval_only" if args.eval_only else "train+test"}')

        # ---- data ------------------------------------------------------
        flow, T_total, N = load_flow(args.data_dir)
        log(f'[data] N={N}, T_total={T_total}')
        train_idx, val_idx, test_idx, train_end_t = make_splits(
            T_total, T_in, T_out)
        if args.max_train_samples is not None and \
                len(train_idx) > args.max_train_samples:
            train_idx = train_idx[:args.max_train_samples]
        log(f'[split] train={len(train_idx)}, val={len(val_idx)}, '
            f'test={len(test_idx)}')

        flow_norm, mean, std = normalize(flow, train_end_t, args.log_transform)
        log(f'[norm] mean={mean.squeeze()}  std={std.squeeze()}')

        need_dyn = args.model == 'mpgcn'
        dyn_graphs = None
        if need_dyn and not args.eval_only:
            log('[dyn] building dynamic O/D graphs...')
            t0 = time.time()
            dyn_graphs = build_dynamic_graphs(
                flow, train_end_t, perceived_period=24)
            log(f'[dyn] done in {time.time()-t0:.1f}s')
        elif need_dyn:
            # eval-only with mpgcn would need dyn graphs too; rebuild
            dyn_graphs = build_dynamic_graphs(flow, train_end_t)

        train_loader, val_loader, test_loader = make_dataloaders(
            flow_norm, (train_idx, val_idx, test_idx), T_in, T_out,
            args.batch_size, dyn_graphs)

        # ---- model + adjacency -----------------------------------------
        adj_processor, static_G, K_final = process_static_adjacency(
            N, args.K_cheby, device)
        model = build_model(args.model, N, T_in, T_out,
                            args.hidden_dim, K_final).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log(f'[model] {args.model}, params={n_params}')

        forward_kwargs = dict(
            model_name=args.model, static_G=static_G,
            adj_processor=adj_processor, device=device, need_dyn=need_dyn)

        # ---- train (skipped in eval_only) ------------------------------
        if not args.eval_only:
            with open(os.path.join(out_dir, 'args.json'), 'w') as f:
                json.dump({k: v for k, v in vars(args).items()
                           if k not in ('eval_only', 'ckpt_dir')}, f, indent=2)
            best_val, best_state = fit(
                model, train_loader, val_loader,
                epochs=args.epochs, patience=args.patience, lr=args.lr,
                forward_kwargs=forward_kwargs, log=log)
            if best_state is not None:
                model.load_state_dict(
                    {k: v.to(device) for k, v in best_state.items()})
                ckpt_path = os.path.join(out_dir, f'{args.model}_best.pt')
                torch.save({'model_state_dict': best_state,
                            'best_val_loss': float(best_val),
                            'args': vars(args)}, ckpt_path)
                log(f'[ckpt_saved] {ckpt_path}')
        else:
            ckpt_path = os.path.join(out_dir, f'{args.model}_best.pt')
            ckpt = torch.load(ckpt_path, map_location='cpu',
                              weights_only=False)
            state = ckpt['model_state_dict']
            model.load_state_dict({k: v.to(device) for k, v in state.items()})
            log(f'[ckpt_loaded] {ckpt_path}  '
                f'best_val={ckpt.get("best_val_loss"):.5f}')

        # ---- streaming test --------------------------------------------
        metrics = streaming_test(
            model, test_loader, mean=mean, std=std,
            log_transform=args.log_transform, T_out=T_out,
            forward_kwargs=forward_kwargs, log=log)
        log_test_summary(metrics, T_out, log)
        out_path = save_metrics(metrics, T_out, T_in, args.model,
                                args.level, out_dir)
        log(f'[metrics_saved] {out_path}')
    finally:
        log_file.close()


def main():
    run_pipeline(parse_args())


if __name__ == '__main__':
    main()
