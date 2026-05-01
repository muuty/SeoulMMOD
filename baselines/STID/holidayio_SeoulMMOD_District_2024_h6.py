from ._build_config_plus import build_config
# Holiday IO: encode both input window's start state AND prediction window's
# start state (= input end + 1). Tells the model "where you came from" and
# "where you're going" - useful for transitions like Sunday -> Monday or
# workday -> holiday eve.
CFG = build_config(
    dataset_name='SeoulMMOD_District_2024_ctx',
    num_nodes=625, num_modes=6, input_dim=6,
    input_len=6, output_len=6,
    tod_index=6, dow_index=7,
    if_holiday=True, if_holiday_future=True,
    holiday_index=8, holiday_use_input_start=True,
    ckpt_tag='holidayio',
)
