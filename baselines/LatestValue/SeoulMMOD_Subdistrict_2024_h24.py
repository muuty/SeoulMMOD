import os
import sys

import numpy as np
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_rmse, unmasked_wmape
from basicts.runners import NoBPRunner
from basicts.scaler import ZScoreScaler

from .arch import LatestValue

DATA_NAME = 'SeoulMMOD_Subdistrict_2024'
INPUT_LEN = 24
OUTPUT_LEN = 24
TRAIN_VAL_TEST_RATIO = [0.7, 0.1, 0.2]
NUM_MODES = 6
TARGET_CHANNEL = list(range(NUM_MODES))
NUM_EPOCHS = 1

CFG = EasyDict()
CFG.DESCRIPTION = f'LatestValue on {DATA_NAME} (input={INPUT_LEN}, output={OUTPUT_LEN})'
CFG.GPU_NUM = 1
CFG.RUNNER = NoBPRunner

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': False,
    'rescale': True,
    'target_channel': TARGET_CHANNEL,
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'LatestValue'
CFG.MODEL.ARCH = LatestValue
CFG.MODEL.PARAM = EasyDict({
    'output_len': OUTPUT_LEN,
    'num_modes': NUM_MODES,
})
CFG.MODEL.FORWARD_FEATURES = list(range(NUM_MODES + 2))
CFG.MODEL.TARGET_FEATURES = TARGET_CHANNEL

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'RMSE': masked_rmse, 'wMAPE': unmasked_wmape})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = np.nan

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints', 'LatestValue',
    f'{DATA_NAME}_{NUM_EPOCHS}_{INPUT_LEN}_{OUTPUT_LEN}',
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
