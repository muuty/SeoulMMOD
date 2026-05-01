"""
OD-pair-level dataset for large-scale OD flow forecasting.

Instead of loading all N OD pairs at once ([L, N, C]), this dataset
samples individual OD pairs and returns [L, N_modes, 1] per sample.
The DataLoader batches across OD pairs: [B, L, N_modes, 1].

This allows standard MTS models (DLinear, PatchTST, TimesNet, etc.)
to run on 181K+ OD pairs without OOM.

data.dat shape: [T, N_od_pairs, N_features]
  - N_features includes mode flows + temporal features
  - Mode flows are the first M columns (target)
  - Temporal features (tod, dow) are shared across OD pairs

The dataset yields samples indexed by (time_idx, od_pair_idx).
"""

import json
import logging
from typing import List

import numpy as np

from .base_dataset import BaseDataset


class ODPairDataset(BaseDataset):
    """Dataset that samples (time, od_pair) for per-pair forecasting."""

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float],
                 mode: str, input_len: int, output_len: int,
                 n_modes: int = 5, memmap: bool = True,
                 overlap: bool = False, logger: logging.Logger = None,
                 node_sample_size: int = 0,
                 mode_index: int = None) -> None:
        """
        Args:
            dataset_name: Name of dataset (for path resolution).
            train_val_test_ratio: [train, val, test] ratios.
            mode: 'train', 'valid', or 'test'.
            input_len: Number of historical time steps.
            output_len: Number of future time steps to predict.
            n_modes: Number of mode features (first n_modes columns of data).
            memmap: Whether to keep data as memmap (recommended for large data).
            overlap: Whether splits can overlap.
            node_sample_size: If >0, randomly sample this many OD pairs per epoch
                              (for faster training). 0 = use all pairs.
            mode_index: If set (int 0..n_modes-1), return only this single
                        mode per sample. Overrides n_modes to 1.
                        in output shape.
        """
        assert mode in ['train', 'valid', 'test']
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap=True)
        self.input_len = input_len
        self.output_len = output_len
        self.n_modes = n_modes
        self.node_sample_size = node_sample_size
        self.mode_index = mode_index
        self.logger = logger
        if mode_index is not None:
            assert 0 <= mode_index < n_modes, f"mode_index {mode_index} out of range [0, {n_modes})"

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'

        with open(self.description_file_path, 'r') as f:
            self.description = json.load(f)

        shape = tuple(self.description['shape'])  # [T, N, F]
        self.data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=shape)

        self.total_time = shape[0]
        self.n_pairs = shape[1]
        self.n_features = shape[2]

        # Time split
        valid_len = int(self.total_time * train_val_test_ratio[1])
        test_len = int(self.total_time * train_val_test_ratio[2])
        train_len = self.total_time - valid_len - test_len

        if mode == 'train':
            self.t_start = 0
            self.t_end = train_len
        elif mode == 'valid':
            self.t_start = train_len - input_len  # need history for first valid sample
            self.t_end = train_len + valid_len
        else:
            self.t_start = train_len + valid_len - input_len
            self.t_end = self.total_time

        self.n_time_samples = self.t_end - self.t_start - input_len - output_len + 1

        # OD pair indices
        if node_sample_size > 0 and node_sample_size < self.n_pairs:
            self._sampled_pairs = np.random.choice(self.n_pairs, node_sample_size, replace=False)
            self._sampled_times = np.random.choice(self.n_time_samples, node_sample_size, replace=True)
            self._n_active_pairs = node_sample_size
        else:
            self._sampled_pairs = None
            self._sampled_times = None
            self._n_active_pairs = self.n_pairs

        if logger:
            logger.info(f'ODPairDataset [{mode}]: time={self.n_time_samples}, '
                        f'pairs={self._n_active_pairs}, '
                        f'total_samples={len(self)}')

    def __len__(self) -> int:
        if self._sampled_pairs is not None:
            return self._n_active_pairs
        return self.n_time_samples * self._n_active_pairs

    def __getitem__(self, index: int) -> dict:
        if self._sampled_pairs is not None:
            pair_idx = self._sampled_pairs[index]
            time_local = self._sampled_times[index]
        else:
            # Decompose flat index → (time_idx, pair_idx)
            time_local = index // self._n_active_pairs
            pair_local = index % self._n_active_pairs
            pair_idx = pair_local
        t = self.t_start + time_local

        # Extract [input_len + output_len, n_modes_out] for this OD pair
        if self.mode_index is not None:
            # Optional single-mode view.
            seq = self.data[t:t + self.input_len + self.output_len, pair_idx,
                            self.mode_index:self.mode_index + 1]
        else:
            seq = self.data[t:t + self.input_len + self.output_len, pair_idx, :self.n_modes]
        seq = np.array(seq, dtype=np.float32)  # copy from memmap

        # Reshape to [L, N, 1] to match BasicTS [L, N, C] convention
        # N = 1 if mode_index is set, else n_modes
        seq = seq[:, :, np.newaxis]

        history = seq[:self.input_len]       # [input_len, n_modes, 1]
        future = seq[self.input_len:]        # [output_len, n_modes, 1]

        return {
            'inputs': history,
            'target': future,
            'time_index': np.int64(t + self.input_len - 1),
        }
