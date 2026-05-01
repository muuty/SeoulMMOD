"""Historical Average (HA) baseline as a BasicTS-compatible nn.Module.

The lookup table is built once at ``__init__`` from the dataset's training
portion (matching the sequence-based train/val/test split used by other
SeoulMMOD baselines). Predictions in forward are looked up by hour-of-week
on the future tod/dow channels — no learnable parameters, no gradient flow.

The table is stored in normalized space (matching BasicTS's ZScoreScaler
with ``target_channel=[0..5]`` and ``norm_each_channel=False``) so that
the framework's standard ``inverse_transform`` brings predictions back to
raw flow units at metric time. This keeps HA on the same eval pipeline
and metrics as the other baselines.

Channel layout assumed (matches SeoulMMOD_Subdistrict_2024 / SeoulMMOD_District_2024):
    channels [0:6]: mode flows (target_channel)
    channel 6:      time-of-day, normalized to [0, 1]
    channel 7:      day-of-week, normalized to [0, 1]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


WEEK_HOURS = 168


class HistoricalAverage(nn.Module):
    """Weekly-cycle historical average lookup, predictions in normalized space."""

    def __init__(
        self,
        dataset_name: str,
        num_nodes: int,
        input_len: int,
        output_len: int,
        train_ratio: float = 0.7,
        num_modes: int = 6,
        target_channel: Sequence[int] = (0, 1, 2, 3, 4, 5),
        tod_index: int = 6,
        dow_index: int = 7,
        repo_root: str | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_modes = num_modes
        self.output_len = output_len
        self.tod_index = tod_index
        self.dow_index = dow_index

        repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        data_path = repo / 'datasets' / dataset_name / 'data.dat'
        desc_path = repo / 'datasets' / dataset_name / 'desc.json'
        with open(desc_path) as f:
            desc = json.load(f)
        T_total, N_pairs, F_total = tuple(desc['shape'])
        if N_pairs != num_nodes:
            raise ValueError(f'num_nodes mismatch: dataset N={N_pairs}, given {num_nodes}')

        flow = np.memmap(data_path, dtype='float32', mode='r',
                         shape=(T_total, N_pairs, F_total))

        # Per-channel mean/std on training portion — matches ZScoreScaler
        # (norm_each_channel=False, target_channel=list).
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

        # Sequence-based train_t_end (last frame referenced by the latest
        # training-set sample) — same convention as eval_naive_baselines.py.
        total_seq = T_total - input_len - output_len + 1
        train_end_seq = int(total_seq * train_ratio)
        train_t_end = train_end_seq + input_len
        n_full_weeks = max(train_t_end // WEEK_HOURS, 1)

        # Build raw weekly-cycle lookup using float64 accumulator.
        acc = np.zeros((WEEK_HOURS, N_pairs, num_modes), dtype=np.float64)
        for w in range(n_full_weeks):
            chunk = np.array(flow[w*WEEK_HOURS:(w+1)*WEEK_HOURS, :, :num_modes],
                             dtype=np.float32)
            acc += chunk
        ha_raw = (acc / n_full_weeks).astype(np.float32)

        # Normalize so model output lives in same space as scaler-transformed targets.
        ha_norm = (ha_raw - means_arr) / stds_arr
        self.register_buffer('table', torch.from_numpy(ha_norm))

        # Trivial trainable parameter so the dummy-training loss has a grad_fn.
        self.register_parameter('_unused', nn.Parameter(torch.zeros(1)))

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Look up HA prediction for each (b, t) using future tod/dow.

        Shapes:
            future_data: [B, output_len, N, F_total]  (target channels normalized)
            returns:     [B, output_len, N, num_modes]  (normalized space)
        """
        # Hour-of-week is identical across nodes within a (b, t) slice; take node 0.
        tod = future_data[:, :, 0, self.tod_index]
        dow = future_data[:, :, 0, self.dow_index]
        # tod/dow are stored as normalized [0,1]; recover discrete categories.
        tod_int = (tod * 24).round().long().clamp(0, 23)
        dow_int = (dow * 7).round().long().clamp(0, 6)
        how = (dow_int * 24 + tod_int) % WEEK_HOURS                   # [B, output_len]

        pred = self.table[how]                                        # [B, T, N, num_modes]
        # Add zero contribution from the dummy parameter so the graph has a grad_fn.
        return pred + 0.0 * self._unused.sum()
