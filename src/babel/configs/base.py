from ml_collections import config_dict


def get_base_config():
    config = config_dict.ConfigDict()
    config.n_mesh_rows = 128
    config.n_mesh_cols = 1

    # architecture
    config.param_dtype = "float32"  # master copy of weights in fp32
    config.dtype = "bfloat16"  # weights and activations are in bfloat16 on fwd/bwd
    config.n_ctx = 256  # sequence length
    config.n_layer = 16  # depth
    config.d_model = 2048  # width
    config.d_head = 128
    config.n_heads_per_group = 1  # gqa equiv to mha if 1
    config.ff_multiple = 3.0  # swiglu width multiple
    config.rotary_base = 10_000  # rope theta
    config.rmsnorm_params = True  # rmsnorm scale params
    config.rmsnorm_eps = 1e-6  # rmsnorm epsilon

    # optimization
    config.tokens_per_global_batch = 2**18  # batch size * sequence len
    config.grad_clip = 1.0  # grad clip max l2 norm
    config.lr_eta = 0.0003  # base learning rate
    config.lr_schedule_name = "cosine"
    config.lr_schedule_end_frac = 0.1
    config.optim_name = "adamw"
    config.optim_dtype = "bfloat16"
    config.optim_beta1 = 0.9
    config.optim_beta2 = 0.95
    config.optim_eps = 1e-8
    config.wd_lam = 0.1  # weight decay lambda
    config.wd_indep = False  # use independent weight decay?

    # periodic action settings
    config.n_log_step = 100  # log every
    config.n_checkpoint_step = 500  # checkpoint every
    config.n_eval_step = 100  # number of eval steps
    config.n_warmup_step = 2_000  # warmup steps during pretraining
    config.n_pretrain_step = 100_000  # pretraining steps

    return config


def get_config():
    return get_base_config()
