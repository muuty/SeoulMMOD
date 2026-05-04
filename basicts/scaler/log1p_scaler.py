from typing import List, Union

import torch

from .base_scaler import BaseScaler


class Log1pScaler(BaseScaler):
    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool,
                 rescale: bool, target_channel: Union[int, List[int]] = 0):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self._channels = [target_channel] if isinstance(target_channel, int) else list(target_channel)
        self._multi = not isinstance(target_channel, int)
        self.target_channel = target_channel

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        for ch in self._channels:
            input_data[..., ch] = torch.log1p(input_data[..., ch])
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        input_data = input_data.clone()
        n_ch = input_data.shape[-1]
        if self._multi and n_ch == len(self._channels):
            for i in range(n_ch):
                input_data[..., i] = torch.expm1(input_data[..., i])
        else:
            for ch in self._channels:
                if ch < n_ch:
                    input_data[..., ch] = torch.expm1(input_data[..., ch])
        return input_data
