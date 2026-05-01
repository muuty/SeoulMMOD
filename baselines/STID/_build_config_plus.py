"""CFG factory for STIDPlus on SeoulMMOD district - holiday / POI / dist+time ablations."""
from __future__ import annotations

import os

import numpy as np
from easydict import EasyDict

from basicts.metrics import masked_mae, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler

from .arch import STIDPlus


NUM_EPOCHS = 100
DISTRICT_POI_CATEGORIES = [
    'poi_food_dining', 'poi_cafe_bar', 'poi_retail_shopping',
    'poi_health_medical', 'poi_beauty_wellness', 'poi_education',
    'poi_office_professional', 'poi_accommodation', 'poi_culture_leisure',
    'poi_transport', 'poi_religious', 'poi_entertainment', 'poi_other',
]
DEFAULT_POI_PATH = os.environ.get(
    'SEOULMMOD_POI_PATH',
    os.path.join('datasets', 'district_poi.csv'),
)


def build_config(*, dataset_name: str, num_nodes: int,
                 input_len: int, output_len: int,
                 input_dim: int, num_modes: int,
                 tod_index: int, dow_index: int,
                 # ablation flags
                 if_holiday: bool = False, holiday_index: int | None = None,
                 if_holiday_future: bool = False,
                 holiday_use_input_start: bool = False,
                 if_poi: bool = False,
                 # extra dynamic input channels (each gets its own Conv2d projection).
                 # specs are in dataset-channel-index space; will be remapped to
                 # forward-features-index space below.
                 extra_input_specs: list[dict] | None = None,
                 forward_features: list[int] | None = None,
                 ckpt_tag: str = 'stidplus') -> EasyDict:
    """Build a CFG for STIDPlus.

    ``input_dim`` controls how many channels feed the time_series_emb_layer
    (typically num_modes, but for the dist/time ablation it widens to num_modes+2).
    ``forward_features`` is the explicit list of channels passed to the model;
    when None, defaults to first ``input_dim`` mode channels + tod + dow [+ holiday].
    """
    if forward_features is None:
        ff = list(range(num_modes)) + [tod_index, dow_index]
        if (if_holiday or if_holiday_future) and holiday_index is not None:
            ff.append(holiday_index)
        forward_features = ff
    target_features = list(range(num_modes))

    CFG = EasyDict()
    CFG.DESCRIPTION = (f'STIDPlus({ckpt_tag}) on {dataset_name} '
                       f'(input_dim={input_dim}, h={input_len})')
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

    CFG.SCALER = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': dataset_name,
        'train_ratio': 0.7,
        'norm_each_channel': False,
        'rescale': True,
        'target_channel': target_features,
    })

    model_param = {
        'num_nodes': num_nodes,
        'input_len': input_len,
        'input_dim': input_dim,
        'output_dim': num_modes,
        'embed_dim': 32,
        'output_len': output_len,
        'num_layer': 3,
        'if_node': True,
        'node_dim': 32,
        'if_T_i_D': True,
        'if_D_i_W': True,
        'temp_dim_tid': 32,
        'temp_dim_diw': 32,
        'time_of_day_size': 24,
        'day_of_week_size': 7,
        # NOTE: indices below are positions in ``forward_features``, not the
        # original dataset channel positions (BasicTS slices FORWARD_FEATURES
        # before the model sees it).
        'tod_index': forward_features.index(tod_index),
        'dow_index': forward_features.index(dow_index),
    }
    if if_holiday or if_holiday_future:
        model_param['holiday_index'] = forward_features.index(holiday_index)
        model_param['holiday_size'] = 3
        model_param['holiday_use_input_start'] = holiday_use_input_start
    if if_holiday:
        model_param['if_holiday'] = True
        model_param['temp_dim_holiday'] = 32
    if if_holiday_future:
        model_param['if_holiday_future'] = True
        model_param['temp_dim_holiday_future'] = 32
    if if_poi:
        model_param['if_poi'] = True
        model_param['poi_features_path'] = DEFAULT_POI_PATH
        model_param['poi_categories'] = DISTRICT_POI_CATEGORIES
        model_param['poi_dim'] = 32
        model_param['poi_hidden'] = 64
        model_param['dataset_name'] = dataset_name
    if extra_input_specs:
        # Remap channel indices to positions inside forward_features.
        remapped = []
        for spec in extra_input_specs:
            remapped.append({
                'name': spec['name'],
                'ch': forward_features.index(spec['ch']),
                'emb_dim': spec['emb_dim'],
            })
        model_param['extra_input_specs'] = remapped

    CFG.MODEL = EasyDict()
    CFG.MODEL.NAME = 'STIDPlus'
    CFG.MODEL.ARCH = STIDPlus
    CFG.MODEL.PARAM = model_param
    CFG.MODEL.FORWARD_FEATURES = forward_features
    CFG.MODEL.TARGET_FEATURES = target_features

    CFG.METRICS = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'RMSE': masked_rmse})
    CFG.METRICS.TARGET = 'MAE'
    CFG.METRICS.NULL_VAL = np.nan

    CFG.TRAIN = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
    CFG.TRAIN.EARLY_STOPPING_PATIENCE = 10
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'checkpoints',
        f'STIDPlus_{ckpt_tag}_2024',
        f'{dataset_name}_{NUM_EPOCHS}_{input_len}_{output_len}',
    )
    CFG.TRAIN.LOSS = masked_mae
    CFG.TRAIN.OPTIM = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = 'Adam'
    CFG.TRAIN.OPTIM.PARAM = {'lr': 0.002, 'weight_decay': 0.0001}
    CFG.TRAIN.LR_SCHEDULER = EasyDict()
    CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
    CFG.TRAIN.LR_SCHEDULER.PARAM = {'milestones': [1, 50, 80], 'gamma': 0.5}
    CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
    CFG.TRAIN.DATA = EasyDict()
    CFG.TRAIN.DATA.BATCH_SIZE = 16
    CFG.TRAIN.DATA.SHUFFLE = True

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
