"""ODMixer architecture (Han et al., 2024), cleanly extracted from
KLatitude/ODMixer with the cfg/logger dependencies stripped so the
classes accept plain kwargs.

Original input/output:
  forward(sequence) where sequence is a dict with 'od' and 'prev_od',
  each shaped (B, T, N, N). Returns two tensors of shape (B, 1, N, N).

This file keeps the original layer set (CM, MM, ODIM, BTL,
SingleInteractModule, MixerLayer, ODMixer) but the ODMixer class now
takes its hyperparameters as constructor kwargs. The wrapper that
adapts ODMixer to our multi-mode multi-step protocol lives in
run_od_native.py (`make_odmixer`).
"""
import torch
from torch import nn
from torch.nn import functional as F


class SingleInteractModule(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, output_dim),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, output_dim),
        )
        self.conv1d = nn.Conv1d(2, 1, 1)

    def forward(self, x, y):
        shape = x.shape
        x_to_y, y_to_x = x.reshape(shape[0], -1), y.reshape(shape[0], -1)
        z = torch.cat((x_to_y.unsqueeze(-1), y_to_x.unsqueeze(-1)), -1)
        z = torch.squeeze(self.conv1d(z.permute(0, 2, 1)), dim=1)
        z = z.reshape(shape)
        gate = torch.sigmoid(self.linear1(z))
        output = self.linear2(x) * gate
        return output + x


class BTL(nn.Module):
    """Bidirectional trend learning between current and previous OD."""

    def __init__(self, input_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.up_interact = SingleInteractModule(input_dim, hid_dim, input_dim, dropout)
        self.down_interact = SingleInteractModule(input_dim, hid_dim, input_dim, dropout)

    def forward(self, x, y):
        return self.up_interact(x, y), self.down_interact(y, x)


class MixerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.ffn(x)


class CM(nn.Module):
    """Channel mixer: mixes across the feature axis (last dim).
    LayerNorm normalizes over the (N, N, hid) volume per sample,
    matching the original ODMixer where input_dim == num_nodes.
    """

    def __init__(self, num_nodes, hid_dim, dropout=0.1):
        super().__init__()
        self.mixer = MixerLayer(hid_dim, 2 * hid_dim, hid_dim, dropout)
        self.ln = nn.LayerNorm([num_nodes, num_nodes, hid_dim])

    def forward(self, x):
        feat = self.mixer(x)
        return self.ln(feat + x)


class MM(nn.Module):
    """Mode mixer: mixes across origin and destination node axes."""

    def __init__(self, num_nodes, hid_dim, dropout=0.1):
        super().__init__()
        self.origin_mixer = MixerLayer(num_nodes, 2 * hid_dim, num_nodes, dropout)
        self.des_mixer = MixerLayer(num_nodes, 2 * hid_dim, num_nodes, dropout)
        self.ln = nn.LayerNorm([num_nodes, num_nodes, hid_dim])

    def forward(self, x):
        # x: (B, N, N, hid)
        # Mix along destination axis (the second N): permute so that
        # dim becomes the last axis for the mixer linear, then back.
        origin_feat = x.permute(0, 1, 3, 2)              # (B, N, hid, N)
        origin_feat = self.origin_mixer(origin_feat)     # mix last axis
        origin_feat = origin_feat.permute(0, 1, 3, 2)    # back to (B, N, N, hid)
        # Mix along origin axis (the first N).
        des_feat = x.permute(0, 2, 3, 1)                 # (B, N, hid, N)
        des_feat = self.des_mixer(des_feat)
        des_feat = des_feat.permute(0, 3, 1, 2)          # back to (B, N, N, hid)
        feat = origin_feat + des_feat
        return self.ln(feat + x)


class ODIM(nn.Module):
    def __init__(self, num_nodes, hid_dim, dropout=0.1):
        super().__init__()
        self.cm = CM(num_nodes, hid_dim, dropout)
        self.mm = MM(num_nodes, hid_dim, dropout)

    def forward(self, x):
        return self.mm(self.cm(x))


class ODMixerBackbone(nn.Module):
    """ODMixer backbone with constructor kwargs (no cfg/logger).

    Input shape : (B, input_seq, N, N) for both `od` and `prev_od`.
    Output shape: (B, out_steps, N, N) for both current and previous
    branches. The multi-mode multi-step adaptation (folding M modes
    into input_seq and inflating out_steps to H*M) is done in the
    wrapper, not here; this class is the faithful original architecture
    aside from cfg/logger plumbing.
    """

    def __init__(self, num_nodes, input_seq, hid_dim,
                 layer_nums=2, dropout=0.1, out_steps=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_seq = input_seq
        self.hid_dim = hid_dim
        self.layer_nums = layer_nums
        self.dropout = dropout
        self.out_steps = out_steps

        self.emb_layer = nn.Linear(input_seq, hid_dim)
        self.encoder_layer = nn.ModuleList(
            [ODIM(num_nodes, hid_dim, dropout) for _ in range(layer_nums)]
        )
        self.trend_layer = nn.ModuleList(
            [BTL(hid_dim, hid_dim, dropout) for _ in range(layer_nums)]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.PReLU(),
            nn.Linear(hid_dim // 2, out_steps),
        )

    def forward(self, od, prev_od):
        """od / prev_od: (B, T, N, N). Returns (od_out, prev_out),
        each shaped (B, out_steps, N, N)."""
        # (B, T, N, N) -> (B, N, N, T) -> emb -> (B, N, N, hid)
        od = od.permute(0, 2, 3, 1)
        prev_od = prev_od.permute(0, 2, 3, 1)
        od_feat = self.emb_layer(od)
        prev_od_feat = self.emb_layer(prev_od)

        for i in range(self.layer_nums):
            od_feat = self.encoder_layer[i](od_feat)
            prev_od_feat = self.encoder_layer[i](prev_od_feat)
            prev_od_feat, od_feat = self.trend_layer[i](prev_od_feat, od_feat)

        od_out = self.output_layer(od_feat)              # (B, N, N, out_steps)
        prev_out = self.output_layer(prev_od_feat)
        # (B, N, N, out_steps) -> (B, out_steps, N, N)
        return od_out.permute(0, 3, 1, 2), prev_out.permute(0, 3, 1, 2)
