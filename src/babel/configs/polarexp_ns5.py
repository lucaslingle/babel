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
    config.optim_name = "muon"
    config.optim_dtype = "float32"
    config.optim_ns_steps = 5

    # periodic action settings
    config.n_pretrain_step = 43_000
    config.n_warmup_step = 850

    return config
