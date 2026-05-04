import os
import sys

import numpy as np
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_rmse, unmasked_wmape
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import AGCRN

############################## Hot Parameters ##############################
# AGCRN does not use tod/dow embeddings, so we only forward the 6 flow channels.
DATA_NAME = 'SeoulMMOD_District_2024'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']

MODEL_ARCH = AGCRN
MODEL_PARAM = {
    'num_nodes': 625,
    'input_dim': 6,           # 6 mode flows as input
    'rnn_units': 64,
    'output_dim': 6,          # predict all 6 modes
    'horizon': OUTPUT_LEN,
    'num_layers': 2,
    'default_graph': True,
    'embed_dim': 10,
    'cheb_k': 2,
}
NUM_EPOCHS = 100
BATCH_SIZE = 32

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'AGCRN on SeoulMMOD_District_2024 (all-mode, input_dim=6, output_dim=6, h=24)'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': False,
    'rescale': True,
    'target_channel': [0, 1, 2, 3, 4, 5],  # normalize all 6 flow channels
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4, 5]  # AGCRN doesn't use tod/dow
CFG.MODEL.TARGET_FEATURES = [0, 1, 2, 3, 4, 5]    # predict all 6 modes

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'RMSE': masked_rmse,
    'wMAPE': unmasked_wmape,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = np.nan

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__ + '_allmode_2024',
    '_'.join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'Adam'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.003,
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = NUM_EPOCHS + 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True
