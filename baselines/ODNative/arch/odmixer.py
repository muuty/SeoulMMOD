from __future__ import annotations

import torch
from torch import nn


class ODMixerSingleInteract(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.conv1d = nn.Conv1d(2, 1, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_to_y = x.reshape(shape[0], -1)
        y_to_x = y.reshape(shape[0], -1)
        z = torch.cat((x_to_y.unsqueeze(-1), y_to_x.unsqueeze(-1)), dim=-1)
        z = torch.squeeze(self.conv1d(z.permute(0, 2, 1)), dim=1)
        z = z.reshape(shape)
        gate = torch.sigmoid(self.linear1(z))
        return self.linear2(x) * gate + x


class ODMixerBTL(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.up_interact = ODMixerSingleInteract(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.down_interact = ODMixerSingleInteract(hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.up_interact(x, y), self.down_interact(y, x)


class ODMixerLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class ODMixerCM(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.mixer = ODMixerLayer(hidden_dim, 2 * hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm([num_nodes, num_nodes, hidden_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mixer(x) + x)


class ODMixerMM(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.origin_mixer = ODMixerLayer(num_nodes, 2 * hidden_dim, num_nodes, dropout)
        self.dest_mixer = ODMixerLayer(num_nodes, 2 * hidden_dim, num_nodes, dropout)
        self.norm = nn.LayerNorm([num_nodes, num_nodes, hidden_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_feat = self.origin_mixer(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        dest_feat = self.dest_mixer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.norm(origin_feat + dest_feat + x)


class ODMixerODIM(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.cm = ODMixerCM(num_nodes, hidden_dim, dropout)
        self.mm = ODMixerMM(num_nodes, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mm(self.cm(x))


class ODMixerBackbone(nn.Module):
    def __init__(self, num_nodes: int, input_seq: int, hidden_dim: int,
                 layer_nums: int = 2, dropout: float = 0.1,
                 out_steps: int = 1) -> None:
        super().__init__()
        self.emb_layer = nn.Linear(input_seq, hidden_dim)
        self.encoder_layer = nn.ModuleList([
            ODMixerODIM(num_nodes, hidden_dim, dropout) for _ in range(layer_nums)
        ])
        self.trend_layer = nn.ModuleList([
            ODMixerBTL(hidden_dim, dropout) for _ in range(layer_nums)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, out_steps),
        )

    def forward(self, od: torch.Tensor, prev_od: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        od_feat = self.emb_layer(od.permute(0, 2, 3, 1))
        prev_od_feat = self.emb_layer(prev_od.permute(0, 2, 3, 1))
        for encoder, trend in zip(self.encoder_layer, self.trend_layer):
            od_feat = encoder(od_feat)
            prev_od_feat = encoder(prev_od_feat)
            prev_od_feat, od_feat = trend(prev_od_feat, od_feat)
        od_out = self.output_layer(od_feat).permute(0, 3, 1, 2)
        prev_out = self.output_layer(prev_od_feat).permute(0, 3, 1, 2)
        return od_out, prev_out
