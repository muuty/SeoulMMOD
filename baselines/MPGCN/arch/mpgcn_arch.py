from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn


def random_walk_supports(adj: torch.Tensor, order: int) -> torch.Tensor:
    if adj.ndim == 2:
        adj = adj.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    adj = adj.float()
    row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    x = (adj / row_sum).transpose(-1, -2)
    eye = torch.eye(adj.shape[-1], dtype=adj.dtype, device=adj.device)
    eye = eye.expand(adj.shape[0], -1, -1)
    supports = [eye]
    if order >= 1:
        supports.append(x)
    for k in range(2, order + 1):
        supports.append(2 * torch.matmul(x, supports[k - 1]) - supports[k - 2])
    out = torch.stack(supports, dim=1)
    return out.squeeze(0) if squeeze else out


def uniform_static_supports(num_nodes: int, order: int) -> torch.Tensor:
    adj = torch.ones(num_nodes, num_nodes, dtype=torch.float32) / num_nodes
    return random_walk_supports(adj, order)


def build_periodic_dynamic_graphs(
    data_path: str,
    desc_path: str,
    num_nodes: int,
    n_modes: int,
    train_ratio: float,
    period: int = 24,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    with open(desc_path, "r") as f:
        shape = tuple(json.load(f)["shape"])
    train_end = int(shape[0] * train_ratio)
    raw = np.memmap(data_path, dtype="float32", mode="r", shape=shape)
    acc = np.zeros((period, num_nodes, num_nodes), dtype=np.float64)
    counts = np.zeros(period, dtype=np.int64)
    for start in range(0, train_end, chunk_size):
        end = min(start + chunk_size, train_end)
        chunk = np.asarray(raw[start:end, :, :n_modes], dtype=np.float32)
        chunk = chunk.sum(axis=-1).reshape(end - start, num_nodes, num_nodes)
        for offset in range(end - start):
            key = (start + offset) % period
            acc[key] += chunk[offset]
            counts[key] += 1
    counts[counts == 0] = 1

    O_dyn = np.zeros((period, num_nodes, num_nodes), dtype=np.float32)
    D_dyn = np.zeros((period, num_nodes, num_nodes), dtype=np.float32)
    for key in range(period):
        avg = acc[key] / counts[key]
        row_norm = np.linalg.norm(avg, axis=1, keepdims=True) + 1e-8
        col_norm = np.linalg.norm(avg, axis=0, keepdims=True) + 1e-8
        row_n = avg / row_norm
        col_n = avg / col_norm
        O_dyn[key] = row_n @ row_n.T
        D_dyn[key] = col_n.T @ col_n
    return torch.from_numpy(O_dyn), torch.from_numpy(D_dyn)


class BDGCN(nn.Module):
    def __init__(self, K: int, input_dim: int, hidden_dim: int,
                 use_bias: bool = True, activation=None) -> None:
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.W = nn.Parameter(torch.empty(self.input_dim * (self.K ** 2), self.hidden_dim))
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim))
            nn.init.constant_(self.b, val=0.0)

    def forward(self, X: torch.Tensor, G: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feat_set = []
        if isinstance(G, torch.Tensor):
            if self.K != G.shape[-3]:
                raise ValueError(f"Expected {self.K} supports, got {G.shape[-3]}")
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum("bncl,nm->bmcl", X, G[o])
                    mode_2_prod = torch.einsum("bmcl,cd->bmdl", mode_1_prod, G[d])
                    feat_set.append(mode_2_prod)
        elif isinstance(G, tuple):
            if len(G) != 2 or self.K != G[0].shape[-3] or self.K != G[1].shape[-3]:
                raise ValueError("Dynamic graph supports must be ((B,K,N,N), (B,K,N,N))")
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum("bncl,bnm->bmcl", X, G[0][:, o])
                    mode_2_prod = torch.einsum("bmcl,bcd->bmdl", mode_1_prod, G[1][:, d])
                    feat_set.append(mode_2_prod)
        else:
            raise TypeError(f"Unsupported graph type: {type(G)}")

        feat = torch.cat(feat_set, dim=-1)
        out = torch.einsum("bmdk,kh->bmdh", feat, self.W)
        if self.use_bias:
            out = out + self.b
        return self.activation(out) if self.activation is not None else out


class MPGCNCore(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, output_dim: int,
                 hidden_dim: int, k_supports: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_branches = 2
        self.num_layers = 3
        self.branches = nn.ModuleList()
        for _ in range(self.num_branches):
            branch = nn.ModuleDict()
            branch["temporal"] = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
            branch["spatial"] = nn.ModuleList([
                BDGCN(K=k_supports, input_dim=hidden_dim, hidden_dim=hidden_dim,
                      use_bias=True, activation=nn.ReLU)
                for _ in range(self.num_layers)
            ])
            branch["fc"] = nn.Linear(hidden_dim, output_dim)
            self.branches.append(branch)

    def _init_hidden(self, batch_size: int, device: torch.device):
        return [
            (
                torch.zeros(1, batch_size * self.num_nodes * self.num_nodes, self.hidden_dim, device=device),
                torch.zeros(1, batch_size * self.num_nodes * self.num_nodes, self.hidden_dim, device=device),
            )
            for _ in range(self.num_branches)
        ]

    def forward(self, x_seq: torch.Tensor, graph_list: list) -> torch.Tensor:
        batch_size, seq_len, num_o, num_d, channels = x_seq.shape
        hidden_list = self._init_hidden(batch_size, x_seq.device)
        lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(batch_size * num_o * num_d, seq_len, channels)
        branch_out = []
        for branch_idx, branch in enumerate(self.branches):
            lstm_out, hidden_list[branch_idx] = branch["temporal"](lstm_in, hidden_list[branch_idx])
            gcn_in = lstm_out[:, -1].reshape(batch_size, num_o, num_d, self.hidden_dim)
            for layer in branch["spatial"]:
                gcn_in = layer(gcn_in, graph_list[branch_idx])
            branch_out.append(branch["fc"](gcn_in))
        out = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)
        return out.unsqueeze(1)


class MPGCNAdapter(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        output_len: int,
        hidden_dim: int = 32,
        K_cheby: int = 2,
        data_path: str | None = None,
        desc_path: str | None = None,
        train_ratio: float = 0.7,
        period: int = 24,
    ) -> None:
        super().__init__()
        if data_path is None or desc_path is None:
            raise ValueError("MPGCNAdapter requires explicit data_path and desc_path")
        if output_dim != input_dim:
            raise ValueError("MPGCNAdapter autoregressive rollout requires output_dim == input_dim")
        self.output_len = output_len
        self.period = period
        self.K_cheby = K_cheby
        self.register_buffer("static_graph", uniform_static_supports(num_nodes, K_cheby))
        O_dyn, D_dyn = build_periodic_dynamic_graphs(
            data_path=data_path,
            desc_path=desc_path,
            num_nodes=num_nodes,
            n_modes=input_dim,
            train_ratio=train_ratio,
            period=period,
        )
        self.register_buffer("O_dyn", O_dyn)
        self.register_buffer("D_dyn", D_dyn)
        self.model = MPGCNCore(num_nodes, input_dim, output_dim, hidden_dim, K_cheby + 1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                time_index: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if time_index is None:
            raise ValueError("MPGCNAdapter requires time_index from ODMatrixDataset")
        keys = (time_index.long() + 1).remainder(self.period)
        O_supports = random_walk_supports(self.O_dyn[keys], self.K_cheby)
        D_supports = random_walk_supports(self.D_dyn[keys], self.K_cheby)
        graph_list = [self.static_graph, (O_supports, D_supports)]
        cur = history_data
        preds = []
        for _ in range(self.output_len):
            step = self.model(cur, graph_list)
            preds.append(step)
            cur = torch.cat([cur[:, 1:], step], dim=1)
        return torch.cat(preds, dim=1)
