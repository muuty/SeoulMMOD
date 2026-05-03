from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import torch
from torch import nn


WEEK_HOURS = 168


class HistoricalAverage(nn.Module):
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        desc_path: str,
        num_nodes: int,
        input_len: int,
        output_len: int,
        train_ratio: float = 0.7,
        num_modes: int = 6,
        target_channel: Sequence[int] = (0, 1, 2, 3, 4, 5),
        tod_index: int = 6,
        dow_index: int = 7,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_modes = num_modes
        self.output_len = output_len
        self.tod_index = tod_index
        self.dow_index = dow_index

        with open(desc_path) as f:
            desc = json.load(f)
        T_total, N_pairs, F_total = tuple(desc['shape'])
        if N_pairs != num_nodes:
            raise ValueError(f'num_nodes mismatch: dataset N={N_pairs}, given {num_nodes}')

        flow = np.memmap(data_path, dtype='float32', mode='r',
                         shape=(T_total, N_pairs, F_total))

        train_size_for_scaler = int(T_total * train_ratio)
        chans = list(target_channel)
        means: list[float] = []
        stds: list[float] = []
        CHUNK = 512
        for ch in chans:
            running_sum = np.float64(0.0)
            running_sq = np.float64(0.0)
            count = 0
            for s in range(0, train_size_for_scaler, CHUNK):
                e = min(s + CHUNK, train_size_for_scaler)
                arr = flow[s:e, :, ch]
                running_sum += arr.sum()
                running_sq += (arr.astype(np.float64) ** 2).sum()
                count += arr.size
            m = running_sum / count
            v = running_sq / count - m * m
            s_dev = float(np.sqrt(max(v, 0.0))) or 1.0
            means.append(float(m))
            stds.append(float(s_dev))
        means_arr = np.asarray(means, dtype=np.float32)
        stds_arr = np.asarray(stds, dtype=np.float32)

        total_seq = T_total - input_len - output_len + 1
        train_end_seq = int(total_seq * train_ratio)
        train_t_end = train_end_seq + input_len
        n_full_weeks = max(train_t_end // WEEK_HOURS, 1)

        acc = np.zeros((WEEK_HOURS, N_pairs, num_modes), dtype=np.float64)
        for w in range(n_full_weeks):
            chunk = np.array(flow[w*WEEK_HOURS:(w+1)*WEEK_HOURS, :, :num_modes],
                             dtype=np.float32)
            acc += chunk
        ha_raw = (acc / n_full_weeks).astype(np.float32)

        ha_norm = (ha_raw - means_arr) / stds_arr
        self.register_buffer('table', torch.from_numpy(ha_norm))

        self.register_parameter('_unused', nn.Parameter(torch.zeros(1)))

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        tod = future_data[:, :, 0, self.tod_index]
        dow = future_data[:, :, 0, self.dow_index]
        tod_int = (tod * 24).round().long().clamp(0, 23)
        dow_int = (dow * 7).round().long().clamp(0, 6)
        how = (dow_int * 24 + tod_int) % WEEK_HOURS

        pred = self.table[how]
        return pred + 0.0 * self._unused.sum()
