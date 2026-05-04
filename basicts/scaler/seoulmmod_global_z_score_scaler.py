import json

import numpy as np
import torch

from .base_scaler import BaseScaler


class SeoulMMODGlobalZScoreScaler(BaseScaler):
    """Global Z-score scaler over SeoulMMOD mode-flow columns."""

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool = False,
        rescale: bool = True,
        n_modes: int = 6,
        mode_index: int = None,
    ):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.n_modes = n_modes
        self.mode_index = mode_index

        description_file_path = f"datasets/{dataset_name}/desc.json"
        with open(description_file_path, "r") as f:
            description = json.load(f)

        data_file_path = f"datasets/{dataset_name}/data.dat"
        data = np.memmap(data_file_path, dtype="float32", mode="r", shape=tuple(description["shape"]))

        train_size = int(len(data) * train_ratio)
        chunk_size = 256

        running_sum = np.float64(0.0)
        running_sq = np.float64(0.0)
        count = 0

        for start in range(0, train_size, chunk_size):
            end = min(start + chunk_size, train_size)
            if self.mode_index is not None:
                chunk = data[start:end, :, self.mode_index:self.mode_index + 1]
            else:
                chunk = data[start:end, :, :self.n_modes]
            chunk64 = chunk.astype(np.float64)
            running_sum += chunk64.sum()
            running_sq += np.square(chunk64).sum()
            count += chunk64.size

        mean = np.float32(running_sum / count)
        std = np.float32(np.sqrt(running_sq / count - (running_sum / count) ** 2))
        if std == 0:
            std = np.float32(1.0)

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data[..., 0] = (input_data[..., 0] - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        input_data[..., 0] = input_data[..., 0] * std + mean
        return input_data
