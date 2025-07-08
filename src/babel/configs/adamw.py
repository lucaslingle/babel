from babel.configs.base import get_base_config


def get_config():
    config = get_base_config()

    # optimization
    config.scaling_lock = "aspect"
    config.optim_name = "adamw"
    config.dataset_name = "fineweb"
    config.model_size = 13 * (128 ** 2) * (28 ** 3)
    config.token_budget = (2 ** 19) * 200_000
    config.tokens_per_global_batch = 2 ** 19
    config.wd_lam = 0.00001
    config.wd_indep = True
    config.qk_norm = True

    return config
