import os
import sys

import numpy as np
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import MTGNNMulti

DATA_NAME = 'SeoulMOD_2024'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 6
OUTPUT_LEN = 6
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']

MODEL_ARCH = MTGNNMulti
MODEL_PARAM = {
    'num_nodes': 181476,
    'output_len': OUTPUT_LEN,
    'num_modes': 6,
    'gcn_true': True,
    'buildA_true': True,             # self-learned graph (no predefined adjacency)
    'gcn_depth': 2,
    'predefined_A': None,
    'static_feat': None,
    'dropout': 0.3,
    'subgraph_size': 20,             # top-k pruning
    'node_dim': 40,
    'dilation_exponential': 1,
    'conv_channels': 32,
    'residual_channels': 32,
    'skip_channels': 64,
    'end_channels': 128,
    'seq_length': INPUT_LEN,
    'in_dim': 6,
    'layers': 3,
    'propalpha': 0.05,
    'tanhalpha': 3,
    'layer_norm_affline': True,
}
NUM_EPOCHS = 100
BATCH_SIZE = 4

CFG = EasyDict()
CFG.DESCRIPTION = 'MTGNN (multi-mode) on SeoulMOD_2024, h=6'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

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
    'target_channel': [0, 1, 2, 3, 4, 5],
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4, 5]
CFG.MODEL.TARGET_FEATURES = [0, 1, 2, 3, 4, 5]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'RMSE': masked_rmse,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = np.nan

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    'MTGNNMulti_allmode_2024',
    '_'.join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'Adam'
CFG.TRAIN.OPTIM.PARAM = {'lr': 0.001, 'weight_decay': 0.0001}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
CFG.TRAIN.LR_SCHEDULER.PARAM = {'milestones': [50], 'gamma': 0.5}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE

CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True
