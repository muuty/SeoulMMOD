import json
import math
import os
from typing import Dict, Optional

import torch
from easytorch.core.checkpoint import save_ckpt
from torch.nn.parallel import DistributedDataParallel as DDP

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
        self.clip_prediction = cfg["MODEL"].get("CLIP_PREDICTION", False)

    def preprocessing(self, input_data: Dict) -> Dict:
        input_data = super().preprocessing(input_data)
        if self.scaler is not None:
            for key in ("prev_inputs", "prev_target"):
                if key in input_data:
                    input_data[key] = self.scaler.transform(input_data[key])
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        input_data = super().postprocessing(input_data)
        if self.scaler is not None and self.scaler.rescale:
            for key in ("prev_prediction", "prev_target"):
                if key in input_data:
                    input_data[key] = self.scaler.inverse_transform(input_data[key])
        return input_data

    def _to_matrix(self, data: torch.Tensor) -> torch.Tensor:
        batch, length, _, channels = data.shape
        return data.reshape(batch, length, self.num_od_nodes, self.num_od_nodes, channels)

    def _to_pairs(self, data: torch.Tensor) -> torch.Tensor:
        batch, length, _, _, channels = data.shape
        return data.reshape(batch, length, self.num_pairs, channels)

    def forward(self, data: Dict, epoch: Optional[int] = None,
                iter_num: Optional[int] = None, train: bool = True,
                **kwargs) -> Dict:
        data = self.preprocessing(data)

        future_data, history_data = data["target"], data["inputs"]
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        prev_history_data = data.get("prev_inputs")
        prev_future_data = data.get("prev_target")
        if prev_history_data is not None:
            prev_history_data = self.to_running_device(prev_history_data)
        if prev_future_data is not None:
            prev_future_data = self.to_running_device(prev_future_data)
        time_index = data.get("time_index")
        if time_index is not None:
            time_index = self.to_running_device(time_index)

        history_target = self.select_target_features(history_data)
        future_target = self.select_target_features(future_data)
        history_model = self.select_input_features(history_data)
        future_model = self.select_input_features(future_data)
        prev_future_target = None

        history_od = self._to_matrix(history_model)
        future_od = self._to_matrix(future_model)
        prev_history_od = None
        prev_future_od = None
        if prev_history_data is not None:
            prev_history_od = self._to_matrix(self.select_input_features(prev_history_data))
        if prev_future_data is not None:
            prev_future_od = self._to_matrix(self.select_input_features(prev_future_data))
            prev_future_target = self.select_target_features(prev_future_data)
        model_return = self.model(
            history_data=history_od,
            future_data=future_od,
            prev_history_data=prev_history_od,
            prev_future_data=prev_future_od,
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

        if "prev_prediction" in model_return:
            prev_prediction = model_return["prev_prediction"]
            if prev_prediction.ndim == 5:
                prev_prediction = self._to_pairs(prev_prediction)
            model_return["prev_prediction"] = prev_prediction
            if prev_future_target is not None:
                model_return["prev_target"] = prev_future_target

        model_return["prediction"] = prediction
        model_return["inputs"] = history_target
        model_return["target"] = future_target

        model_return = self.postprocessing(model_return)
        if self.clip_prediction and not train and "prediction" in model_return:
            model_return["prediction"] = model_return["prediction"].clamp_min(0)
        return model_return

    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        metric = self.meter_pool.get_value(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": self.optim.state_dict(),
                "best_metrics": self.best_metrics,
            }
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                "{}_best_{}.pt".format(self.model_name, metric_name.replace("/", "_")),
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
            self.current_patience = self.early_stopping_patience
        elif self.early_stopping_patience is not None:
            self.current_patience -= 1
