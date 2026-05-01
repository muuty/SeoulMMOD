from ._build_config_plus import build_config
CFG = build_config(
    dataset_name='SeoulMOD_GU_2024_rain',
    num_nodes=625, num_modes=6, input_dim=6,
    input_len=12, output_len=12,
    tod_index=8, dow_index=9,
    forward_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    extra_input_specs=[
        {'name': 'rain_o', 'ch': 6, 'emb_dim': 16},
        {'name': 'rain_d', 'ch': 7, 'emb_dim': 16},
    ],
    ckpt_tag='rain',
)
