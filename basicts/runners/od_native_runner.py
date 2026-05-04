import json
import math
import os
from typing import Dict, Optional

import torch

from .runner_zoo.simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class ODNativeRunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        dataset_name = cfg["DATASET"]["NAME"]
        desc_path = os.path.join("datasets", dataset_name, "desc.json")
        with open(desc_path, "r") as f:
            desc = json.load(f)
        self.num_pairs = tuple(desc["shape"])[1]
        self.num_od_nodes = int(math.sqrt(self.num_pairs))
        if self.num_od_nodes * self.num_od_nodes != self.num_pairs:
            raise ValueError(f"OD-native models require square OD pairs, got {self.num_pairs}")

    def _to_matrix(self, data: torch.Tensor) -> torch.Tensor:
        batch, length, num_pairs, channels = data.shape
        if num_pairs != self.num_pairs:
            raise ValueError(f"Expected {self.num_pairs} OD pairs, got {num_pairs}")
        return data.reshape(batch, length, self.num_od_nodes, self.num_od_nodes, channels)

    def _to_pairs(self, data: torch.Tensor) -> torch.Tensor:
        batch, length, n_origin, n_dest, channels = data.shape
        if n_origin != self.num_od_nodes or n_dest != self.num_od_nodes:
            raise ValueError(f"Expected OD matrix {self.num_od_nodes}x{self.num_od_nodes}, got {n_origin}x{n_dest}")
        return data.reshape(batch, length, self.num_pairs, channels)

    def forward(self, data: Dict, epoch: Optional[int] = None,
                iter_num: Optional[int] = None, train: bool = True,
                **kwargs) -> Dict:
        data = self.preprocessing(data)

        future_data, history_data = data["target"], data["inputs"]
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        time_index = data.get("time_index")
        if time_index is not None:
            time_index = self.to_running_device(time_index)

        batch_size, length, num_pairs, _ = future_data.shape

        history_target = self.select_target_features(history_data)
        future_target = self.select_target_features(future_data)
        history_model = self.select_input_features(history_data)
        future_model = self.select_input_features(future_data)

        history_od = self._to_matrix(history_model)
        future_od = self._to_matrix(future_model)
        model_return = self.model(
            history_data=history_od,
            future_data=future_od,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
            time_index=time_index,
        )
        if isinstance(model_return, torch.Tensor):
            model_return = {"prediction": model_return}

        prediction = model_return["prediction"]
        if prediction.ndim == 5:
            prediction = self._to_pairs(prediction)
        elif prediction.ndim != 4:
            raise ValueError(f"Unexpected OD-native prediction shape: {prediction.shape}")

        model_return["prediction"] = prediction
        model_return["inputs"] = history_target
        model_return["target"] = future_target

        assert list(prediction.shape)[:3] == [batch_size, length, num_pairs], \
            "OD-native prediction must flatten back to [B, L, N_pairs, C]."

        return self.postprocessing(model_return)
