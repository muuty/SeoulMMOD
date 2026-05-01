"""Latest Value (LV / persistence) baseline as a BasicTS-compatible nn.Module.

The prediction at every horizon step is the last observed input frame
(``history_data[:, -1, ...]`` over the target channels). Param-free.

Designed to run **without a scaler** so predictions are returned in raw
flow units, matching the convention used by HistoricalAverage. The
framework then computes metrics directly on the raw scale.
"""
from __future__ import annotations

import torch
from torch import nn


class LatestValue(nn.Module):
    """Persistence forecast: repeat the last input frame across the horizon."""

    def __init__(self, output_len: int, num_modes: int = 6) -> None:
        super().__init__()
        self.output_len = output_len
        self.num_modes = num_modes
        # Trivial trainable parameter keeps the optimizer happy. No effect on forward.
        self.register_parameter('_unused', nn.Parameter(torch.zeros(1)))

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Return [B, output_len, N, num_modes] copies of the last input frame.

        Shapes:
            history_data: [B, input_len, N, F_total]  (raw values)
            returns:      [B, output_len, N, num_modes]
        """
        last = history_data[:, -1:, :, :self.num_modes]               # [B, 1, N, M]
        pred = last.expand(-1, self.output_len, -1, -1).contiguous()  # [B, T, N, M]
        # Add zero contribution from the dummy parameter so the graph has a
        # grad_fn (BasicTS calls loss.backward() during the dummy training pass).
        return pred + 0.0 * self._unused.sum()
