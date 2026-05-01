import os, sys, numpy as np
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from .arch import GraphWaveNet

DATA_NAME = "SeoulMOD_GU_2024_m0"
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings["INPUT_LEN"]
OUTPUT_LEN = regular_settings["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]

MODEL_ARCH = GraphWaveNet
MODEL_PARAM = {
    "num_nodes": 625, "dropout": 0.3,
    "supports": None, "gcn_bool": True, "addaptadj": True, "aptinit": None,
    "in_dim": 1, "out_dim": OUTPUT_LEN,
    "residual_channels": 32, "dilation_channels": 32,
    "skip_channels": 256, "end_channels": 512,
    "kernel_size": 2, "blocks": 4, "layers": 2,
}
NUM_EPOCHS = 100
CFG = EasyDict()
CFG.DESCRIPTION = "GWNet mode-sep m0"
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({"dataset_name": DATA_NAME, "train_val_test_ratio": TRAIN_VAL_TEST_RATIO, "input_len": INPUT_LEN, "output_len": OUTPUT_LEN})
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({"dataset_name": DATA_NAME, "train_ratio": TRAIN_VAL_TEST_RATIO[0], "norm_each_channel": False, "rescale": True})
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({"MAE": masked_mae, "RMSE": masked_rmse})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = np.nan
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "GWNet_m0_modesep", "_".join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]))
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [50], "gamma": 0.5}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True
