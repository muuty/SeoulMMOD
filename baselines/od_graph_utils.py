from __future__ import annotations

import numpy as np
import torch


def random_walk_supports(adj: torch.Tensor, order: int) -> torch.Tensor:
    batched = adj.ndim == 3
    if not batched:
        adj = adj.unsqueeze(0)

    adj = adj.float()
    row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    transition = (adj / row_sum).transpose(-1, -2)
    eye = torch.eye(adj.shape[-1], dtype=adj.dtype, device=adj.device)
    eye = eye.expand(adj.shape[0], -1, -1)
    supports = [eye]
    if order >= 1:
        supports.append(transition)
    for k in range(2, order + 1):
        supports.append(2 * torch.matmul(transition, supports[k - 1]) - supports[k - 2])
    supports = torch.stack(supports, dim=1)
    return supports if batched else supports.squeeze(0)


def load_static_supports(adj_path: str, order: int) -> torch.Tensor:
    adj = torch.from_numpy(np.load(adj_path).astype(np.float32))
    return random_walk_supports(adj, order)


def uniform_static_supports(num_nodes: int, order: int) -> torch.Tensor:
    adj = torch.ones(num_nodes, num_nodes, dtype=torch.float32)
    adj = adj / adj.sum(dim=-1, keepdim=True)
    return random_walk_supports(adj, order)
