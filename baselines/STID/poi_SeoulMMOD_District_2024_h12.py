from ._build_config_plus import build_config
CFG = build_config(
    dataset_name='SeoulMMOD_District_2024',
    num_nodes=625, num_modes=6, input_dim=6,
    input_len=12, output_len=12,
    tod_index=6, dow_index=7,
    if_poi=True,
    ckpt_tag='poi',
)
