import os

import torch
from easytorch.core.checkpoint import save_ckpt
from torch.nn.parallel import DistributedDataParallel as DDP

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class _NoOpOptimizer:
    param_groups = []

    def zero_grad(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def __repr__(self):
        return "NoOpOptimizer()"


class NoBPRunner(SimpleTimeSeriesForecastingRunner):
    def build_optim(self, optim_cfg, model):
        return _NoOpOptimizer()

    def backward(self, loss: torch.Tensor):
        pass

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
