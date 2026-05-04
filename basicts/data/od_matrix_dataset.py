import json
import logging
from typing import List, Optional

import numpy as np

from .simple_tsf_dataset import TimeSeriesForecastingDataset


class ODMatrixDataset(TimeSeriesForecastingDataset):
    def __init__(self, dataset_name: str, train_val_test_ratio: List[float],
                 mode: str, input_len: int, output_len: int,
                 memmap: bool = True, overlap: bool = False,
                 prev_period: Optional[int] = None,
                 logger: logging.Logger = None) -> None:
        super().__init__(
            dataset_name=dataset_name,
            train_val_test_ratio=train_val_test_ratio,
            mode=mode,
            input_len=input_len,
            output_len=output_len,
            memmap=memmap,
            overlap=overlap,
            logger=logger,
        )
        num_pairs = tuple(self.description["shape"])[1]
        num_od_nodes = int(np.sqrt(num_pairs))
        if num_od_nodes * num_od_nodes != num_pairs:
            raise ValueError(f"ODMatrixDataset requires square OD pairs, got {num_pairs}")
        self.num_od_nodes = num_od_nodes
        self.time_offset = self._time_offset()
        self.prev_period = prev_period
        self.sample_offset = 0
        self.full_data = None
        if self.prev_period is not None:
            self.sample_offset = max(self.prev_period - self.time_offset, 0)
            self.full_data = np.memmap(
                self.data_file_path,
                dtype="float32",
                mode="r",
                shape=tuple(self.description["shape"]),
            )

    def _time_offset(self) -> int:
        total_len = tuple(self.description["shape"])[0]
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len
        if self.mode == "train":
            return 0
        if self.mode == "valid":
            offset_left = self.input_len - 1 if self.overlap else 0
            return train_len - offset_left
        offset_left = self.input_len - 1 if self.overlap else 0
        return train_len + valid_len - offset_left

    def __getitem__(self, index: int) -> dict:
        sample_index = index + self.sample_offset
        item = super().__getitem__(sample_index)
        item["time_index"] = np.int64(self.time_offset + sample_index + self.input_len - 1)
        if self.prev_period is not None:
            history_start = self.time_offset + sample_index - self.prev_period
            target_start = history_start + self.input_len
            item["prev_inputs"] = self._full_slice(history_start, self.input_len)
            item["prev_target"] = self._full_slice(target_start, self.output_len)
        return item

    def __len__(self) -> int:
        return super().__len__() - self.sample_offset

    def _full_slice(self, start: int, length: int) -> np.ndarray:
        return np.asarray(self.full_data[start:start + length], dtype=np.float32).copy()
