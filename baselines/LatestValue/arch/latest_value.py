from __future__ import annotations

import torch
from torch import nn


class LatestValue(nn.Module):
    def __init__(self, output_len: int, num_modes: int = 6) -> None:
        super().__init__()
        self.output_len = output_len
        self.num_modes = num_modes

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        last = history_data[:, -1:, :, :self.num_modes]
        pred = last.expand(-1, self.output_len, -1, -1).contiguous()
        return pred
