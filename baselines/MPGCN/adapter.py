from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from ..od_graph_utils import load_static_supports, random_walk_supports, uniform_static_supports
from .arch.mpgcn_arch import BDGCN


def build_periodic_dynamic_graphs(
    data_path: str,
    desc_path: str,
    num_nodes: int,
    n_modes: int,
    train_ratio: float,
    input_len: int,
    output_len: int,
    period: int = 24,
    chunk_size: int = 128,
    dynamic_graph_type: str = "distance",
) -> tuple[torch.Tensor, torch.Tensor]:
    with open(desc_path, "r") as f:
        shape = tuple(json.load(f)["shape"])
    train_end = int((shape[0] - input_len - output_len + 1) * train_ratio) + input_len
    train_end = (train_end // period) * period
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
        o_sim = row_n @ row_n.T
        d_sim = col_n.T @ col_n
        if dynamic_graph_type == "similarity":
            O_dyn[key] = o_sim
            D_dyn[key] = d_sim
        elif dynamic_graph_type == "distance":
            o_dist = np.clip(1.0 - o_sim, 0.0, 2.0)
            d_dist = np.clip(1.0 - d_sim, 0.0, 2.0)
            np.fill_diagonal(o_dist, 0.0)
            np.fill_diagonal(d_dist, 0.0)
            O_dyn[key] = o_dist
            D_dyn[key] = d_dist
        else:
            raise ValueError(f"Unsupported dynamic_graph_type: {dynamic_graph_type}")
    return torch.from_numpy(O_dyn), torch.from_numpy(D_dyn)


class MPGCNAdapter(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        output_len: int,
        data_path: str,
        desc_path: str,
        adj_path: str | None = None,
        hidden_dim: int = 32,
        K_cheby: int = 2,
        train_ratio: float = 0.7,
        input_len: int | None = None,
        period: int = 24,
        static_graph_type: str = "adjacency",
        dynamic_graph_type: str = "distance",
    ) -> None:
        super().__init__()
        self.output_len = output_len
        self.period = period
        self.K_cheby = K_cheby
        if static_graph_type == "uniform":
            static_graph = uniform_static_supports(num_nodes, K_cheby)
        elif static_graph_type == "adjacency":
            if adj_path is None:
                raise ValueError("adj_path is required for adjacency static graph")
            static_graph = load_static_supports(adj_path, K_cheby)
        else:
            raise ValueError(f"Unsupported static_graph_type: {static_graph_type}")
        self.register_buffer("static_graph", static_graph)
        O_dyn, D_dyn = build_periodic_dynamic_graphs(
            data_path=data_path,
            desc_path=desc_path,
            num_nodes=num_nodes,
            n_modes=input_dim,
            train_ratio=train_ratio,
            input_len=input_len or output_len,
            output_len=output_len,
            period=period,
            dynamic_graph_type=dynamic_graph_type,
        )
        self.register_buffer("O_dyn", O_dyn)
        self.register_buffer("D_dyn", D_dyn)
        self.model = MPGCNMulti(num_nodes, input_dim, output_dim, output_len, hidden_dim, K_cheby + 1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                time_index: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        keys = (time_index.long() + 1).remainder(self.period)
        O_supports = random_walk_supports(self.O_dyn[keys], self.K_cheby)
        D_supports = random_walk_supports(self.D_dyn[keys], self.K_cheby)
        graph_list = [self.static_graph, (O_supports, D_supports)]
        return self.model(history_data, graph_list)


class MPGCNMulti(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, output_dim: int,
                 output_len: int, hidden_dim: int, k_supports: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.output_len = output_len
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
            branch["fc"] = nn.Linear(hidden_dim, output_len * output_dim)
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
        out = out.view(batch_size, num_o, num_d, self.output_len, self.output_dim)
        return out.permute(0, 3, 1, 2, 4).contiguous()
