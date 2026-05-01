"""Multi-mode wrapper around MTGNN.

MTGNN's end_conv_2 produces [B, out_dim, N, 1]. To predict L*M
(horizon * modes) values per node, we set out_dim=L*M and reshape into
[B, L, N, M].
"""
import torch
from torch import nn

from .mtgnn_arch import MTGNN


class MTGNNMulti(nn.Module):
    def __init__(self, num_nodes, output_len, num_modes,
                 gcn_true=True, buildA_true=True, gcn_depth=2,
                 predefined_A=None, static_feat=None, dropout=0.3,
                 subgraph_size=20, node_dim=40, dilation_exponential=1,
                 conv_channels=32, residual_channels=32,
                 skip_channels=64, end_channels=128,
                 seq_length=12, in_dim=6, layers=3,
                 propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super().__init__()
        self.output_len = output_len
        self.num_modes = num_modes
        self.mtgnn = MTGNN(
            gcn_true=gcn_true,
            buildA_true=buildA_true,
            gcn_depth=gcn_depth,
            num_nodes=num_nodes,
            predefined_A=predefined_A,
            static_feat=static_feat,
            dropout=dropout,
            subgraph_size=subgraph_size,
            node_dim=node_dim,
            dilation_exponential=dilation_exponential,
            conv_channels=conv_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            seq_length=seq_length,
            in_dim=in_dim,
            out_dim=output_len * num_modes,
            layers=layers,
            propalpha=propalpha,
            tanhalpha=tanhalpha,
            layer_norm_affline=layer_norm_affline,
        )

    def forward(self, history_data, idx=None, **kwargs):
        # history_data: [B, L_in, N, C_in=6]
        out = self.mtgnn(history_data=history_data, idx=idx, **kwargs)
        # out: [B, out_dim, N, 1] = [B, L*M, N, 1]
        B, Lflat, N, _ = out.shape
        assert Lflat == self.output_len * self.num_modes
        out = out.squeeze(-1)                            # [B, L*M, N]
        out = out.view(B, self.output_len, self.num_modes, N)
        out = out.permute(0, 1, 3, 2).contiguous()       # [B, L, N, M]
        return out
