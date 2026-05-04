import os
import sys
import numpy as np
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_rmse, unmasked_wmape
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import STID

############################## Hot Parameters ##############################
DATA_NAME = 'SeoulMMOD_District_2024'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']

MODEL_ARCH = STID
MODEL_PARAM = {
    "num_nodes": 625,
    "input_len": INPUT_LEN,
    "input_dim": 6,           # 6 mode flows
    "output_dim": 6,          # predict all 6 modes
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 3,
    "if_node": True,
    "node_dim": 32,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": 24,
    "day_of_week_size": 7,
    "tod_index": 6,           # tod at channel 6 (after 6 flows)
    "dow_index": 7,           # dow at channel 7
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'STID on SeoulMMOD_District_2024 all-mode (input_dim=6, output_dim=6, tod+dow, h=12)'
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
    'target_channel': [0, 1, 2, 3, 4, 5],
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]
CFG.MODEL.TARGET_FEATURES = [0, 1, 2, 3, 4, 5]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'RMSE': masked_rmse, 'wMAPE': unmasked_wmape})
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
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.002, "weight_decay": 0.0001}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 50, 80], "gamma": 0.5}
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 16

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = NUM_EPOCHS + 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 16

CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True
