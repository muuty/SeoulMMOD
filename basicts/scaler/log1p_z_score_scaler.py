import json
from typing import List, Union

import numpy as np
import torch

from .base_scaler import BaseScaler


class Log1pZScoreScaler(BaseScaler):
    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        target_channel: Union[int, List[int]] = 0,
        input_len: int | None = None,
        output_len: int | None = None,
    ):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self._channels = [target_channel] if isinstance(target_channel, int) else list(target_channel)
        self._multi = not isinstance(target_channel, int)
        self.target_channel = target_channel

        with open(f"datasets/{dataset_name}/desc.json", "r") as f:
            shape = tuple(json.load(f)["shape"])
        data = np.memmap(f"datasets/{dataset_name}/data.dat", dtype="float32", mode="r", shape=shape)
        if input_len is not None and output_len is not None:
            train_end = int((shape[0] - input_len - output_len + 1) * train_ratio) + input_len
        else:
            train_end = int(shape[0] * train_ratio)

        means = []
        stds = []
        for ch in self._channels:
            running_sum = np.float64(0.0)
            running_sq = np.float64(0.0)
            count = 0
            for start in range(0, train_end, 512):
                end = min(start + 512, train_end)
                arr = np.log1p(np.asarray(data[start:end, :, ch], dtype=np.float32))
                running_sum += arr.sum(dtype=np.float64)
                running_sq += (arr.astype(np.float64) ** 2).sum()
                count += arr.size
            mean = running_sum / count
            var = max(running_sq / count - mean * mean, 0.0)
            std = np.sqrt(var)
            means.append(np.float32(mean))
            stds.append(np.float32(std if std > 0 else 1.0))

        self.mean = torch.tensor(means if self._multi else means[0])
        self.std = torch.tensor(stds if self._multi else stds[0])

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        if self._multi:
            for i, ch in enumerate(self._channels):
                input_data[..., ch] = (torch.log1p(input_data[..., ch]) - mean[i]) / std[i]
        else:
            ch = self._channels[0]
            input_data[..., ch] = (torch.log1p(input_data[..., ch]) - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        n_ch = input_data.shape[-1]
        if self._multi and n_ch == len(self._channels):
            for i in range(n_ch):
                input_data[..., i] = torch.expm1(input_data[..., i] * std[i] + mean[i])
        elif self._multi:
            for i, ch in enumerate(self._channels):
                if ch < n_ch:
                    input_data[..., ch] = torch.expm1(input_data[..., ch] * std[i] + mean[i])
        else:
            ch = self._channels[0] if n_ch > self._channels[0] else 0
            input_data[..., ch] = torch.expm1(input_data[..., ch] * std + mean)
        return input_data
