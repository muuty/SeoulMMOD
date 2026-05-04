from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from .mpgcn import MPGCNCore
from .odcrn import ODCRN
from .odmixer import ODMixerBackbone


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


def load_desc_shape(desc_path: str) -> tuple[int, int, int]:
    with open(desc_path, "r") as f:
        return tuple(json.load(f)["shape"])


def build_periodic_dynamic_graphs(
    data_path: str,
    desc_path: str,
    num_nodes: int,
    n_modes: int,
    train_ratio: float,
    period: int = 24,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = load_desc_shape(desc_path)
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


class ODCRNAdapter(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        output_len: int,
        hidden_dim: int = 32,
        K_cheby: int = 2,
        num_layers: int = 2,
        dgc: bool = True,
    ) -> None:
        super().__init__()
        if output_dim != input_dim:
            raise ValueError("ODCRNAdapter requires output_dim == input_dim")
        if num_layers != 2:
            raise ValueError("ODCRNAdapter follows the original two-graph setup and requires num_layers=2")
        self.register_buffer("static_graph", uniform_static_supports(num_nodes, K_cheby))
        self.model = ODCRN(
            num_nodes=num_nodes,
            K=K_cheby + 1,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_horizon=output_len,
            num_layers=num_layers,
            DGCbool=dgc,
            use_bias=True,
            activation=None,
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        return self.model(G=self.static_graph, X_seq=history_data)


class ODMixerAdapter(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 32,
        layer_nums: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_dim != input_dim:
            raise ValueError("ODMixerAdapter autoregressive rollout requires output_dim == input_dim")
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_len = output_len
        self.backbone = ODMixerBackbone(
            num_nodes=num_nodes,
            input_seq=input_len * input_dim,
            hidden_dim=hidden_dim,
            layer_nums=layer_nums,
            dropout=dropout,
            out_steps=output_dim,
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        cur = history_data
        preds = []
        for _ in range(self.output_len):
            batch_size, input_len, _, _, input_dim = cur.shape
            od = cur.permute(0, 1, 4, 2, 3).reshape(
                batch_size, input_len * input_dim, self.num_nodes, self.num_nodes
            )
            prev_od = torch.zeros_like(od)
            od_out, _ = self.backbone(od, prev_od)
            step = od_out.reshape(
                batch_size, self.output_dim, self.num_nodes, self.num_nodes
            ).permute(0, 2, 3, 1).unsqueeze(1)
            preds.append(step)
            cur = torch.cat([cur[:, 1:], step], dim=1)
        return torch.cat(preds, dim=1).contiguous()
