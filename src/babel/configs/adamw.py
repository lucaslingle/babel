from babel.configs.base import get_base_config


def get_config():
    config = get_base_config()

    # mesh
    config.n_mesh_rows = 32
    config.n_mesh_cols = 4

    # width
    config.d_model = 3584
    config.n_layer = 28

    # optimization
    config.tokens_per_global_batch = 2 ** 21
    config.optim_name = "adamw"
    config.optim_dtype = "float32"

    # periodic action settings
    config.n_pretrain_step = 50_000
    config.n_warmup_step = 1_000

    return config
