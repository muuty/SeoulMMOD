from __future__ import annotations

import torch
from torch import nn

from ..od_graph_utils import load_static_supports
from .arch.odcrn_arch import ODCRN


class ODCRNAdapter(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        output_len: int,
        adj_path: str,
        hidden_dim: int = 32,
        K_cheby: int = 2,
        num_layers: int = 2,
        dgc: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("static_graph", load_static_supports(adj_path, K_cheby))
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
