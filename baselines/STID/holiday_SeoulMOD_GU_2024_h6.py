from ._build_config_plus import build_config
# Dataset SeoulMOD_GU_2024_ctx has 9 channels: 6 modes, tod=6, dow=7, holiday=8.
CFG = build_config(
    dataset_name='SeoulMOD_GU_2024_ctx',
    num_nodes=625, num_modes=6, input_dim=6,
    input_len=6, output_len=6,
    tod_index=6, dow_index=7,
    if_holiday=True, holiday_index=8,
    ckpt_tag='holiday',
)
