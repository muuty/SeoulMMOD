from __future__ import annotations

import torch
from torch import nn


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
