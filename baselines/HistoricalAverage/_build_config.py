"""Shared CFG factory for HistoricalAverage configs across (dataset, horizon)."""
from __future__ import annotations

import os

import numpy as np
from easydict import EasyDict

from basicts.metrics import masked_mae, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler

from .arch import HistoricalAverage


NUM_EPOCHS = 1  # one dummy training pass; HA table is built at __init__


def build_config(dataset_name: str, num_nodes: int,
                 input_len: int, output_len: int,
                 num_modes: int = 6) -> EasyDict:
    """Return a BasicTS CFG for HistoricalAverage on the given dataset and horizon.

    Skips the scaler (HA operates in raw flow units). Optimizer is SGD with
    lr=0 — there are no learnable parameters, the trivial dummy parameter
    produces a zero gradient, and the optimizer step is a no-op.
    """
    CFG = EasyDict()
    CFG.DESCRIPTION = (f'HistoricalAverage on {dataset_name} '
                       f'(input={input_len}, output={output_len})')
    CFG.GPU_NUM = 1
    CFG.RUNNER = SimpleTimeSeriesForecastingRunner

    CFG.DATASET = EasyDict()
    CFG.DATASET.NAME = dataset_name
    CFG.DATASET.TYPE = TimeSeriesForecastingDataset
    CFG.DATASET.PARAM = EasyDict({
        'dataset_name': dataset_name,
        'train_val_test_ratio': [0.7, 0.1, 0.2],
        'input_len': input_len,
        'output_len': output_len,
    })

    # ZScoreScaler matches the scaler used by other SeoulMOD baselines (STID,
    # iTransformer, SOFTS, etc.). HA's lookup table is stored in normalized
    # space inside the arch, so framework transform/inverse_transform pipeline
    # produces raw-scale metrics equivalent to ``eval_naive_baselines_gpu.py``.
    CFG.SCALER = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    target_channel = list(range(num_modes))
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': dataset_name,
        'train_ratio': 0.7,
        'norm_each_channel': False,
        'rescale': True,
        'target_channel': target_channel,
    })

    CFG.MODEL = EasyDict()
    CFG.MODEL.NAME = 'HistoricalAverage'
    CFG.MODEL.ARCH = HistoricalAverage
    CFG.MODEL.PARAM = EasyDict({
        'dataset_name': dataset_name,
        'num_nodes': num_nodes,
        'input_len': input_len,
        'output_len': output_len,
        'train_ratio': 0.7,
        'num_modes': num_modes,
        'target_channel': target_channel,
        'tod_index': num_modes,
        'dow_index': num_modes + 1,
    })
    CFG.MODEL.FORWARD_FEATURES = list(range(num_modes + 2))
    CFG.MODEL.TARGET_FEATURES = target_channel

    CFG.METRICS = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'RMSE': masked_rmse})
    CFG.METRICS.TARGET = 'MAE'
    CFG.METRICS.NULL_VAL = np.nan

    CFG.TRAIN = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'checkpoints', 'HistoricalAverage',
        f'{dataset_name}_{NUM_EPOCHS}_{input_len}_{output_len}',
    )
    CFG.TRAIN.LOSS = masked_mae
    CFG.TRAIN.OPTIM = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = 'SGD'
    CFG.TRAIN.OPTIM.PARAM = {'lr': 0.0}
    CFG.TRAIN.DATA = EasyDict()
    CFG.TRAIN.DATA.BATCH_SIZE = 16
    CFG.TRAIN.DATA.SHUFFLE = False

    CFG.VAL = EasyDict()
    CFG.VAL.INTERVAL = 1
    CFG.VAL.DATA = EasyDict()
    CFG.VAL.DATA.BATCH_SIZE = 16

    CFG.TEST = EasyDict()
    CFG.TEST.INTERVAL = 1
    CFG.TEST.DATA = EasyDict()
    CFG.TEST.DATA.BATCH_SIZE = 16

    CFG.EVAL = EasyDict()
    CFG.EVAL.USE_GPU = True

    return CFG
