"""Shared CFG factory for LatestValue configs across (dataset, horizon)."""
from __future__ import annotations

import os

import numpy as np
from easydict import EasyDict

from basicts.metrics import masked_mae, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler

from .arch import LatestValue


NUM_EPOCHS = 1  # one dummy pass; LV is parameter-free


def build_config(dataset_name: str, num_nodes: int,
                 input_len: int, output_len: int,
                 num_modes: int = 6) -> EasyDict:
    """Return a BasicTS CFG for LatestValue on the given dataset and horizon."""
    CFG = EasyDict()
    CFG.DESCRIPTION = (f'LatestValue on {dataset_name} '
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

    # ZScoreScaler matches other SeoulMOD baselines. LV repeats the last input
    # frame, which (under the scaler) is in normalized space — framework
    # inverse_transform restores raw scale for metric computation.
    CFG.SCALER = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': dataset_name,
        'train_ratio': 0.7,
        'norm_each_channel': False,
        'rescale': True,
        'target_channel': list(range(num_modes)),
    })

    CFG.MODEL = EasyDict()
    CFG.MODEL.NAME = 'LatestValue'
    CFG.MODEL.ARCH = LatestValue
    CFG.MODEL.PARAM = EasyDict({
        'output_len': output_len,
        'num_modes': num_modes,
    })
    CFG.MODEL.FORWARD_FEATURES = list(range(num_modes + 2))
    CFG.MODEL.TARGET_FEATURES = list(range(num_modes))

    CFG.METRICS = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'RMSE': masked_rmse})
    CFG.METRICS.TARGET = 'MAE'
    CFG.METRICS.NULL_VAL = np.nan

    CFG.TRAIN = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'checkpoints', 'LatestValue',
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
