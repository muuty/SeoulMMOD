"""Multi-mode wrapper around GraphWaveNet.

GraphWaveNet's end_conv_2 produces [B, out_dim, N, 1]. To predict L*M
(horizon * modes) values per node, we set out_dim=L*M and reshape into
[B, L, N, M].
"""
import torch
from torch import nn

from .gwnet_arch import GraphWaveNet


class GraphWaveNetMulti(nn.Module):
    def __init__(self, num_nodes, output_len, num_modes,
                 supports=None, gcn_bool=True, addaptadj=True,
                 aptinit=None, in_dim=6,
                 residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, dropout=0.3):
        super().__init__()
        self.output_len = output_len
        self.num_modes = num_modes
        self.gwnet = GraphWaveNet(
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=aptinit,
            in_dim=in_dim,
            out_dim=output_len * num_modes,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
        )

    def forward(self, history_data, future_data=None, batch_seen=None,
                epoch=None, train=True, **kwargs):
        # history_data: [B, L_in, N, C_in=6]
        out = self.gwnet(history_data=history_data,
                         future_data=future_data,
                         batch_seen=batch_seen,
                         epoch=epoch, train=train, **kwargs)
        # GWNet output shape: [B, out_dim, N, T_out]
        # When L_in == receptive_field T_out=1; when L_in > receptive_field, T_out>1.
        # We collapse T_out by taking the last step (the rightmost is the most
        # context-rich one) and then reshape into (L, M).
        B, Lflat, N, T_out = out.shape
        assert Lflat == self.output_len * self.num_modes, \
            f'expected {self.output_len * self.num_modes} flat steps, got {Lflat}'
        if T_out != 1:
            out = out[..., -1:]
        out = out.squeeze(-1)                           # [B, L*M, N]
        out = out.view(B, self.output_len, self.num_modes, N)
        out = out.permute(0, 1, 3, 2).contiguous()      # [B, L, N, M]
        return out
