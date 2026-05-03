"""STAEformer on subdistrict-scale SeoulMMOD_Subdistrict_2024 (all-mode joint, 181476 nodes, h=6).

OOM test: ST models are typically designed for small N. This config attempts
the full subdistrict-scale all-mode setting to probe the OOM boundary.
"""
import os
import sys
import numpy as np
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_rmse, unmasked_wape
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import STAEformer

DATA_NAME = 'SeoulMMOD_Subdistrict_2024'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 6
OUTPUT_LEN = 6
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']

MODEL_ARCH = STAEformer
MODEL_PARAM = {
    "num_nodes": 181476,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": 24,
    "input_dim": 6,
    "output_dim": 6,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "tod_index": 6,
    "dow_index": 7,
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 80,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "use_mixed_proj": True,
}
NUM_EPOCHS = 100

CFG = EasyDict()
CFG.DESCRIPTION = 'STAEformer on SeoulMMOD_Subdistrict_2024 subdistrict-scale (OOM probe, 181476 nodes, h=6)'
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
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]
CFG.MODEL.TARGET_FEATURES = [0, 1, 2, 3, 4, 5]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'RMSE': masked_rmse,
    'WAPE': unmasked_wape,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = np.nan

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
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [25, 50, 75],
    "gamma": 0.5,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0,
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 1
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 1

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 1

CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True
