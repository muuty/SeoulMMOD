import numpy as np
import torch


def unmasked_wape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Per-batch global WAPE = sum(|prediction - target|) / sum(|target|).

    Differs from `masked_wape` in two ways:
      1. No null masking. All elements (including zeros) contribute.
      2. Sums over the entire batch tensor at once instead of computing a
         per-row WAPE and then averaging. This matches the WAPE definition
         used in `eval_per_mode.py` and `analyze_poi_error_gu.py`.

    `null_val` is accepted for API parity with `masked_wape` but is ignored.

    When aggregated across batches via AvgMeter (the default), the reported
    value is the batch_size-weighted mean of per-batch WAPEs, which equals
    the true global WAPE only if all batches share the same sum-of-targets.
    For exact cross-batch global WAPE, use a standalone evaluation script
    that accumulates the numerator and denominator separately.
    """
    numerator = torch.sum(torch.abs(prediction - target))
    denominator = torch.sum(torch.abs(target)) + 5e-5
    return numerator / denominator
