import json
from typing import Union, List

import numpy as np
import torch

from .base_scaler import BaseScaler


class ZScoreScaler(BaseScaler):
    """
    ZScoreScaler performs Z-score normalization on the dataset.

    Supports single channel (int) or multiple channels (list[int]) for target_channel.
    When target_channel is a list, per-channel mean/std are computed and stored as 1-D tensors.
    """

    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool, rescale: bool,
                 target_channel: Union[int, List[int]] = 0):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)

        # Normalize target_channel to always be a list internally
        if isinstance(target_channel, int):
            self._multi = False
            self._channels = [target_channel]
        else:
            self._multi = True
            self._channels = list(target_channel)
        self.target_channel = target_channel

        # load dataset
        description_file_path = f'datasets/{dataset_name}/desc.json'
        with open(description_file_path, 'r') as f:
            description = json.load(f)
        data_file_path = f'datasets/{dataset_name}/data.dat'
        data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))
        train_size = int(len(data) * train_ratio)

        CHUNK = 512
        if norm_each_channel:
            # per-node normalization (on first target channel only, for backward compat)
            ch = self._channels[0]
            n_nodes = data.shape[1]
            running_sum = np.zeros(n_nodes, dtype=np.float64)
            running_sq = np.zeros(n_nodes, dtype=np.float64)
            count = 0
            for start in range(0, train_size, CHUNK):
                end = min(start + CHUNK, train_size)
                chunk = data[start:end, :, ch]
                running_sum += chunk.sum(axis=0).astype(np.float64)
                running_sq += (chunk.astype(np.float64) ** 2).sum(axis=0)
                count += (end - start)
            self.mean = torch.tensor((running_sum / count).astype(np.float32).reshape(1, -1))
            std = np.sqrt(running_sq / count - (running_sum / count) ** 2).astype(np.float32).reshape(1, -1)
            std[std == 0] = 1.0
            self.std = torch.tensor(std)
        else:
            # Per-channel global mean/std
            means, stds = [], []
            for ch in self._channels:
                running_sum = np.float64(0)
                running_sq = np.float64(0)
                count = 0
                for start in range(0, train_size, CHUNK):
                    end = min(start + CHUNK, train_size)
                    chunk = data[start:end, :, ch]
                    running_sum += chunk.sum().astype(np.float64)
                    running_sq += (chunk.astype(np.float64) ** 2).sum()
                    count += chunk.size
                m = np.float32(running_sum / count)
                s = np.float32(np.sqrt(running_sq / count - (running_sum / count) ** 2))
                if s == 0:
                    s = np.float32(1.0)
                means.append(m)
                stds.append(s)

            if self._multi:
                self.mean = torch.tensor(means)  # shape: [n_channels]
                self.std = torch.tensor(stds)
            else:
                self.mean = torch.tensor(means[0])  # scalar
                self.std = torch.tensor(stds[0])

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        if self._multi:
            for i, ch in enumerate(self._channels):
                input_data[..., ch] = (input_data[..., ch] - mean[i]) / std[i]
        else:
            input_data[..., self._channels[0]] = (input_data[..., self._channels[0]] - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        n_ch = input_data.shape[-1]

        if self._multi:
            # After feature selection, channels may be reindexed 0..k
            # If data has exactly len(channels) dims, apply in order
            if n_ch == len(self._channels):
                for i in range(n_ch):
                    input_data[..., i] = input_data[..., i] * std[i] + mean[i]
            else:
                # Apply to original channel indices if they exist
                for i, ch in enumerate(self._channels):
                    if ch < n_ch:
                        input_data[..., ch] = input_data[..., ch] * std[i] + mean[i]
        else:
            ch = self._channels[0] if n_ch > self._channels[0] else 0
            input_data[..., ch] = input_data[..., ch] * std + mean
        return input_data
