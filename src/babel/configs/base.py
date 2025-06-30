from ml_collections import config_dict


def get_base_config():
    config = config_dict.ConfigDict()

    config.scaling_lock = "aspect"  # one of "aspect", "depth", "width"
    config.dataset_name = "fineweb"  # one of "fineweb", "commonpile", "cci3hq"
    config.model_size = 1_200_000_000  # used with scaling_lock to derive d_model and n_layer
    config.token_budget = 26_000_000_000  # used with tokens_per_global_batch to derive n_training_steps

    # architecture
    config.param_dtype = "float32"  # master copy of weights in fp32
    config.dtype = "bfloat16"  # weights and activations are in bfloat16 on fwd/bwd
    config.d_head = 128
    config.n_heads_per_group = 1  # gqa equiv to mha if 1
    config.ff_multiple = 3.0  # swiglu width multiple
    config.rotary_base = 10_000  # rope theta
    config.rmsnorm_params = True  # rmsnorm scale params
    config.rmsnorm_eps = 1e-6  # rmsnorm epsilon
    config.qk_norm = False  # normalize queries and keys?

    # optimization
    config.tokens_per_global_batch = 2**18  # batch size * sequence len
    config.grad_clip = 1.0  # grad clip max l2 norm
    config.optim_name = "adamw"  # one of "adamw", "lion", "muon"
    config.optim_dtype = "float32"
    config.lr_eta = 0.0003  # learning rate
    config.lr_schedule_name = "cosine"
    config.lr_schedule_warmup_frac = 0.02
    config.lr_schedule_end_value_frac = 0.1
    config.wd_lam = 0.1  # weight decay
    config.wd_indep = False  # use independent weight decay?

    # periodic action settings
    config.n_log_step = 100  # log every
    config.n_checkpoint_step = 1000  # checkpoint every
    config.n_fast_eval_step = 100

    return config


def get_config():
    return get_base_config()
