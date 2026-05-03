import os
import sys
import numpy as np
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.scaler import SeoulMMODGlobalZScoreScaler
from basicts.data import ODPairDataset
from basicts.metrics import masked_mae, masked_rmse, unmasked_wape
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import get_regular_settings

from .arch import MTSMixer

############################## Hot Parameters ##############################
DATA_NAME = "SeoulMMOD_District_2024"
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]

MODEL_ARCH = MTSMixer
MODEL_PARAM = {
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "enc_in": 6,
    "dec_in": 6,
    "d_model": 64,
    "d_ff": 4,
    "use_norm": True,
    "e_layers": 4,
    "fac_T": False,
    "fac_C": True,
    "sampling": 1,
    "individual": False,
    "rev": True,
}
NUM_EPOCHS = 100
BATCH_SIZE = 256

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = "MTSMixer on SeoulMMOD_District_2024 (district-scale, per-OD-pair, 6 interacting modes as variates, all 625 pairs, h=12)"
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = ODPairDataset
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "n_modes": 6,
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = SeoulMMODGlobalZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_ratio": TRAIN_VAL_TEST_RATIO[0],
    "norm_each_channel": False,
    "rescale": True,
    "n_modes": 6,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    "MAE": masked_mae,
    "RMSE": masked_rmse,
    "WAPE": unmasked_wape,
})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = np.nan

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join([DATA_NAME, "allmode_2024", str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
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
    "milestones": [1, 25, 50],
    "gamma": 0.5,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0,
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
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True
