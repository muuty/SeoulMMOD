import numpy as np
import torch

from .metric_meter import AvgMeter


def _wmape_sums(prediction, target):
    return ((prediction - target).abs().sum(dtype=torch.float64),
            target.abs().sum(dtype=torch.float64))


class WMAPEMeter(AvgMeter):
    """Aggregate wMAPE across batches."""

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        error_sum, target_sum = _wmape_sums(prediction, target)
        self._sum += error_sum.item()
        self._count += target_sum.item()

    @property
    def value(self) -> float:
        return self._sum / self._count if self._count else float('nan')


def unmasked_wmape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Return sum(|prediction - target|) / sum(|target|) as a fraction."""
    numerator, denominator = _wmape_sums(prediction, target)
    return numerator / denominator.masked_fill(denominator == 0, float('nan'))
