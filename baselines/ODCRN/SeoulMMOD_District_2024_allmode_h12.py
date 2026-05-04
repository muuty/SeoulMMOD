import os
import sys

import numpy as np
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import ODMatrixDataset
from basicts.metrics import masked_mae, masked_rmse, unmasked_wmape
from basicts.runners import ODNativeRunner
from basicts.scaler import ZScoreScaler

from .arch import ODCRNAdapter


DATA_NAME = "SeoulMMOD_District_2024"
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.7, 0.1, 0.2]
NUM_EPOCHS = 100
BATCH_SIZE = 16

MODEL_ARCH = ODCRNAdapter
MODEL_PARAM = {
    "num_nodes": 25,
    "input_dim": 6,
    "output_dim": 6,
    "output_len": OUTPUT_LEN,
    "hidden_dim": 32,
    "K_cheby": 2,
    "num_layers": 2,
    "dgc": True,
}

CFG = EasyDict()
CFG.DESCRIPTION = "ODCRN OD-native adapter on SeoulMMOD_District_2024, h=12"
CFG.GPU_NUM = 1
CFG.RUNNER = ODNativeRunner

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = ODMatrixDataset
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "memmap": True,
})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_ratio": TRAIN_VAL_TEST_RATIO[0],
    "norm_each_channel": False,
    "rescale": True,
    "target_channel": [0, 1, 2, 3, 4, 5],
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4, 5]
CFG.MODEL.TARGET_FEATURES = [0, 1, 2, 3, 4, 5]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({"MAE": masked_mae, "RMSE": masked_rmse, "wMAPE": unmasked_wmape})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = np.nan

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 10
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "ODCRN_allmode_2024",
    "_".join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001}
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 3.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True

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
