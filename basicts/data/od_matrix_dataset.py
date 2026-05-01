"""
OD matrix dataset for FactorizedOD-style models.

data.dat shape: [T, N, N, M] where N=regions, M=modes.
Each sample returns the full OD matrix at a time step + history.

Returns:
    inputs: [input_len, N, N, M] - history OD matrices
    target: [output_len, N, N, M] - future OD matrices
    tod: [output_len] - hour of day for each target step
    dow: [output_len] - day of week for each target step
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np

from .base_dataset import BaseDataset


class ODMatrixDataset(BaseDataset):
    """Dataset for full OD matrix forecasting."""

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float],
                 mode: str, input_len: int, output_len: int,
                 start_date: str = "2024-01-01",
                 memmap: bool = True, logger: logging.Logger = None) -> None:
        assert mode in ['train', 'valid', 'test']
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap=True)
        self.input_len = input_len
        self.output_len = output_len
        self.logger = logger

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'

        with open(self.description_file_path, 'r') as f:
            self.description = json.load(f)

        shape = tuple(self.description['shape'])  # [T, N, N, M]
        self.data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=shape)

        self.total_time = shape[0]
        self.start_datetime = datetime.strptime(start_date, "%Y-%m-%d")

        # Time split
        valid_len = int(self.total_time * train_val_test_ratio[1])
        test_len = int(self.total_time * train_val_test_ratio[2])
        train_len = self.total_time - valid_len - test_len

        if mode == 'train':
            self.t_start = 0
            self.t_end = train_len
        elif mode == 'valid':
            self.t_start = train_len - input_len
            self.t_end = train_len + valid_len
        else:
            self.t_start = train_len + valid_len - input_len
            self.t_end = self.total_time

        self.n_samples = self.t_end - self.t_start - input_len - output_len + 1

        if logger:
            logger.info(f'ODMatrixDataset [{mode}]: samples={self.n_samples}, '
                        f'shape={shape}')

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        t = self.t_start + index

        # History and future OD matrices
        history = np.array(self.data[t:t + self.input_len], dtype=np.float32)
        future = np.array(self.data[t + self.input_len:t + self.input_len + self.output_len], dtype=np.float32)

        # Compute tod/dow from time index
        tod = np.zeros(self.output_len, dtype=np.int64)
        dow = np.zeros(self.output_len, dtype=np.int64)
        for i in range(self.output_len):
            dt = self.start_datetime + timedelta(hours=int(t + self.input_len + i))
            tod[i] = dt.hour
            dow[i] = dt.weekday()

        # Also provide history tod/dow for models that need it
        hist_tod = np.zeros(self.input_len, dtype=np.int64)
        hist_dow = np.zeros(self.input_len, dtype=np.int64)
        for i in range(self.input_len):
            dt = self.start_datetime + timedelta(hours=int(t + i))
            hist_tod[i] = dt.hour
            hist_dow[i] = dt.weekday()

        return {
            'inputs': history,          # [input_len, N, N, M]
            'target': future,           # [output_len, N, N, M]
            'tod': tod,                 # [output_len]
            'dow': dow,                 # [output_len]
            'hist_tod': hist_tod,       # [input_len]
            'hist_dow': hist_dow,       # [input_len]
        }
