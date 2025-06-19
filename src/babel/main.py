import functools
import posixpath
import time
import math

import flax.linen as nn
import jax
import jax.experimental.mesh_utils as jmu
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
import wandb
from absl import app
from absl import flags
from absl import logging
from flax.training import train_state as train_utils
from ml_collections import config_flags

from babel.data import get_tokenizer
from babel.data import get_dataset
from babel.data import get_batch
from babel.data import count_batches
from babel.data import DATASET_CONFIGS
from babel.model import MESH_AXES
from babel.model import TransformerConfig
from babel.model import Transformer
from babel.optim import muon
from babel.sharding import get_namedsharding
from babel.sharding import sharding_constraint
from babel.sharding import to_global_array

FLAGS = flags.FLAGS
PROJECT_NAME = "babel_sweeps_paper"
config_flags.DEFINE_config_file("config", None, "Config file", lock_config=False)
flags.DEFINE_string("workdir", None, "Working directory (GCS or local)")
flags.DEFINE_string("group", None, "Group name for experiment")
flags.DEFINE_integer("seed", 0, "Experiment rng seed")
flags.DEFINE_string("hf_token", None, "API key for HuggingFace")
flags.DEFINE_string("wb_token", None, "API key for Weights and Biases")
flags.DEFINE_string("wb_runid", None, "Run ID for Weights and Biases")
flags.DEFINE_bool("tpu", True, "Use TPU cluster?")
flags.mark_flags_as_required(["config", "workdir", "group", "wb_token", "hf_token"])


@functools.lru_cache(maxsize=1)
def get_depth_and_width():
    assert FLAGS.config.ff_multiple == 3.0

    n = FLAGS.config.model_size
    if FLAGS.scaling_lock == "aspect":
        n_layer = int(math.ceil((n / (13 * 128 * 128)) ** 0.33))
        d_model = 128 * n_layer
        return (n_layer, d_model)
    elif FLAGS.scaling_lock == "depth":
        n_layer = 6
        d_model = (int(math.ceil((n / (13 * n_layer)) ** 0.5)) // 128) * 128
        return (n_layer, d_model)
    elif FLAGS.scaling_lock == "width":
        d_model = 768;
        n_layer = int(math.ceil(n / (13 * d_model ** 2)))
        return (n_layer, d_model)
    else:
        raise NotImplementedError


def get_n_layer():
    return get_depth_and_width()[0]


def get_d_model():
    return get_depth_and_width()[1]


def get_n_pretrain_step():
    return FLAGS.config.token_budget // FLAGS.config.tokens_per_global_batch


@functools.lru_cache(maxsize=1)
def get_dataset_config():
    return DATASET_CONFIGS[FLAGS.config.dataset_name]


@functools.lru_cache(maxsize=1)
def get_transformer_config():
    return TransformerConfig.create(
        **vars(FLAGS.config)["_fields"],
        d_model=get_d_model(),
        n_layer=get_n_layer(),
        n_ctx=get_dataset_config().sequence_len,
        n_vocab=len(get_tokenizer(dataset_config=get_dataset_config(), hf_token=FLAGS.hf_token)),
    )


@functools.lru_cache(maxsize=1)
def get_global_mesh():
    # mesh settings were optimized via trial and error for tpu v3
    assert jax.device_count() == 128
    n_mesh_rows = 128 if FLAGS.config.model_size < 10 ** 9 else 32
    n_mesh_cols = 1 if FLAGS.config.model_size < 10 ** 9 else 4
    return jax.sharding.Mesh(
        devices=jmu.create_device_mesh(
            mesh_shape=(n_mesh_rows, n_mesh_cols),
            devices=jax.devices(),
        ),
        axis_names=("X", "Y"),  # we will be using 2D-finalized from GSPMD paper
    )


def get_init_rng():
    return jax.random.PRNGKey(FLAGS.seed)


def get_params(rng, global_mesh):
    config = get_transformer_config()
    inputs = jnp.zeros(dtype=jnp.uint32, shape=[1, config.n_ctx])
    params = Transformer(config, global_mesh).init({"params": rng}, inputs)["params"]
    return params


def get_schedule():
    warmup_steps = int(FLAGS.config.lr_schedule_warmup_frac * get_n_pretrain_step())
    annealing_steps = get_n_pretrain_step() - warmup_steps
    end = FLAGS.config.lr_schedule_end_value_frac
    warmup = optax.linear_schedule(0.0, 1.0, transition_steps=warmup_steps)
    if FLAGS.config.lr_schedule_name == "linear":
        annealing = optax.linear_schedule(1.0, end, transition_steps=annealing_steps)
    elif FLAGS.config.lr_schedule_name == "cosine":
        annealing = optax.cosine_decay_schedule(1.0, alpha=end, decay_steps=annealing_steps)
    else:
        raise NotImplementedError
    return optax.join_schedules([warmup, annealing], boundaries=[warmup_steps])


def get_optimizer():
    if FLAGS.config.optim_name == "adamw":
        kwargs = dict(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            mu_dtype=FLAGS.config.optim_dtype,
            weight_decay=0.0 if FLAGS.config.wd_indep else FLAGS.config.wd_lam,
        )
        return optax.adamw(FLAGS.config.lr_eta, **common_kwargs)
    elif FLAGS.config.optim_name == "lion":
        kwargs = dict(
            b1=0.95,
            b2=0.98,
            mu_dtype=FLAGS.config.optim_dtype,
            weight_decay=0.0 if FLAGS.config.wd_indep else FLAGS.config.wd_lam,
        )
        return optax.lion(FLAGS.config.lr_eta, **common_kwargs)
    elif FLAGS.config.optim_name == "muon":
        kwargs = dict(
            b1=0.9,  # used by adam subset of muon
            b2=0.95,  # used by adam subset of muon
            eps=1e-8,  # used by adam subset of muon
            mu_dtype=FLAGS.config.optim_dtype,
            weight_decay=0.0 if FLAGS.config.wd_indep else FLAGS.config.wd_lam,
        )
        return muon(FLAGS.config.lr_eta, **common_kwargs)
    else:
        raise NotImplementedError


def grad_transform_factory():
    chain = []
    if FLAGS.config.grad_clip > 0.0:
        chain.append(optax.clip_by_global_norm(FLAGS.config.grad_clip))
    chain.append(get_optimizer())
    if FLAGS.config.wd_indep:
        chain.append(optax.add_decayed_weights(-FLAGS.config.wd_lam))
        # Normally we use scale_by_learning_rate
        # as the final step in the chained optimizer definition e.g., in adamw.
        # This scales by the negative of the lr,
        # so that the weight decay and the update are subtracted from the weights
        # by optax.apply_updates(p, u) := (p + u).
        #
        # When using weight decay outside an optimizer, as in here,
        # we have no following scale_by_learning_rate,
        # so we flip the sign of the weight decay manually.
    chain.append(optax.scale_by_schedule(get_schedule()))
    return optax.chain(*chain)


def init_fn(rng):
    return train_utils.TrainState.create(
        apply_fn=None,
        params=get_params(rng=rng, global_mesh=get_global_mesh()),
        tx=get_optimizer(),
    )


def get_train_state(rng_init):
    # ref: https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html#the-output-s-sharding  # noqa
    global_mesh = get_global_mesh()
    prng_sharding = get_namedsharding(axis_names=(None,), device_mesh=global_mesh)
    abstract_variables = jax.eval_shape(init_fn, rng_init)
    state_sharding = nn.get_sharding(abstract_variables, global_mesh)
    jit_init_fn = jax.jit(
        init_fn,
        static_argnums=(),
        in_shardings=(prng_sharding,),
        out_shardings=state_sharding,
    )
    initialized_state = jit_init_fn(rng_init)
    return initialized_state


def get_ndbe():
    # returns (param ct, token ct, global bsz in tokens, lr)
    nl = FLAGS.config.n_layer
    dm = FLAGS.config.d_model
    dff = int(FLAGS.config.d_model * FLAGS.config.ff_multiple)
    ff_proj_ct = 3
    ns = get_n_pretrain_step()
    bsz = FLAGS.config.tokens_per_global_batch

    n = nl * (4 * dm ** 2 + ff_proj_ct * dm * dff)
    d = ns * bsz
    b = bsz
    e = FLAGS.config.lr_eta
    return n, d, b, e


def get_modelname():
    n, d, b, e = get_ndbe()
    return f"{FLAGS.group}_{b}_{e}"


def get_checkpoint_manager():
    return ocp.CheckpointManager(
        directory=posixpath.join(FLAGS.workdir, "checkpoints", get_modelname()),
        options=ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=1,
            save_interval_steps=1,
            step_prefix="state",
            enable_async_checkpointing=False,
        ),
    )


def do_restore(mgr, state):
    if mgr.latest_step() is not None:
        abstract = jtu.tree_map(ocp.utils.to_shape_dtype_struct, state)
        state = mgr.restore(
            step=mgr.latest_step(),
            args=ocp.args.StandardRestore(abstract),
        )
    start_step = mgr.latest_step() or 0
    return state, start_step


def do_save(mgr, state, step):
    mgr.save(step, args=ocp.args.StandardSave(state))
    mgr.wait_until_finished()


def get_loss_mask(tgt, pad_id):
    return jnp.logical_not(jnp.equal(tgt, pad_id))


def extract_input_and_target(batch):
    dataset_config = get_dataset_config()
    if dataset_config.tokenizer_adds_bos:
        # in this case, the array shape is (bsz, seqlen+1), and we slice only,
        # obtaining inp, tgt shapes of (bsz, seqlen).
        inp, tgt = batch[:, 0:-1], batch[:, 1:]
    else:
        # in this case, the array shape is (bsz, seqlen), and we slice and pad
        # obtaining inp, tgt shapes of (bsz, seqlen).
        bos_token_id = get_tokenizer(
            dataset_config=dataset_config, 
            hf_token=FLAGS.hf_token,
        ).bos_token_id
        inp = jnp.pad(
            batch[:, 0:-1],
            pad_width=((0, 0), (1, 0)),
            constant_values=bos_token_id,
        )
        tgt = batch
    return inp, tgt


def loss_fn(params, batch):
    config = get_transformer_config()
    global_mesh = get_global_mesh()
    inp, tgt = extract_input_and_target(batch)
    logits = Transformer(config, global_mesh).apply({"params": params}, inp)
    terms = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt)
    terms = sharding_constraint(terms, MESH_AXES["XN"], global_mesh)
    pad_id = get_tokenizer(dataset_config=get_dataset_config(), hf_token=FLAGS.hf_token).pad_token_id
    mask = get_loss_mask(tgt=tgt, pad_id=pad_id)
    metrics = dict(
        loss_term_avg=jnp.mean(mask * terms),
        loss_mask_avg=jnp.mean(mask),
        loss_avg=jnp.mean(mask * terms) / jnp.mean(mask),
    )
    return metrics["loss_term_avg"], metrics


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step_op(state, batch):
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def train_logging_op(metrics, step, t0):
    metrics = jax.block_until_ready(metrics)
    tf = time.perf_counter()
    metrics["secs_per_step"] = (tf - t0) / FLAGS.config.n_log_step
    metrics["step"] = step
    logging.info(metrics)
    if jax.process_index() == 0:
        wandb.log(data=metrics, step=step)
    return tf


@jax.jit
def eval_step_op(state, batch):
    (_, metrics) = loss_fn(state.params, batch)
    return metrics


def eval_step_logging_op(metrics, eval_step):
    jax.block_until_ready(metrics)
    logging.info(f"Eval step {eval_step}")


def eval_loss_logging_op(eval_loss, train_step):
    jax.block_until_ready(eval_loss)
    if jax.process_index() == 0:
        wandb.log(data=dict(eval_loss=eval_loss), step=train_step)
    logging.info(f"Eval loss {eval_loss}")


def train_loop():
    global_mesh = get_global_mesh()
    rng_init = get_init_rng()
    mgr = get_checkpoint_manager()
    state = get_train_state(rng_init)
    state, start_step = do_restore(mgr, state)

    n_host = jax.process_count()
    n_ctx = get_dataset_config().sequence_len
    local_batch_size = FLAGS.config.tokens_per_global_batch // (n_host * n_ctx)
    ds_train_shard = get_dataset(
        local_batch_size=local_batch_size,
        dataset_config=get_dataset_config(),
        split="train",
        workdir=FLAGS.workdir,
        hf_token=FLAGS.hf_token,
    )

    eval_loss = None
    t0 = time.perf_counter()
    for train_step in range(start_step, get_n_pretrain_step()):
        batch = get_batch(
            shard=ds_train_shard,
            local_batch_size=local_batch_size,
            dataset_config=get_dataset_config(),
            step=train_step,
        )
        batch = to_global_array(batch, global_mesh)
        state, metrics = train_step_op(state, batch)

        if (step + 1) % FLAGS.config.n_log_step == 0:
            t0 = train_logging_op(metrics, step, t0)

        if (step + 1) % FLAGS.config.n_checkpoint_step == 0:
            eval_loss = eval_loop(state)
            eval_loss_logging_op(eval_loss, step)
            do_save(mgr, state, step)

    eval_loss = eval_loop(state)
    eval_loss_logging_op(eval_loss, step)
    do_save(mgr, state, step)

    return eval_loss


def eval_loop(state):
    global_mesh = get_global_mesh()

    n_host = jax.process_count()
    n_ctx = get_dataset_config().sequence_len
    local_batch_size = FLAGS.config.tokens_per_global_batch // (n_host * n_ctx)
    ds_test_shard = get_dataset(
        local_batch_size=local_batch_size,
        dataset_config=get_dataset_config(),
        split="test",
        workdir=FLAGS.workdir,
        hf_token=FLAGS.hf_token,
    )
    batch_count = count_batches(
        shard=shard, 
        local_batch_size=local_batch_size,
        dataset_config=get_dataset_config(),
    )

    loss_terms = []
    mask_terms = []
    for eval_step in range(0, batch_count):
        batch = get_batch(
            shard=ds_test_shard,
            local_batch_size=local_batch_size,
            dataset_config=get_dataset_config(),
            step=eval_step,
        )
        batch = to_global_array(batch, global_mesh)
        metrics = eval_step_op(state, batch)
        eval_step_logging_op(metrics, step)
        loss_terms.append(metrics["loss_term_avg"].astype(jnp.float32))
        mask_terms.append(metrics["loss_mask_avg"].astype(jnp.float32))

    loss_terms_avg = sum(loss_terms) / len(loss_terms)
    mask_terms_avg = sum(mask_terms) / len(mask_terms)
    eval_loss_avg = loss_terms_avg / mask_terms_avg
    return eval_loss_avg


def main(argv):
    del argv
    if FLAGS.tpu:
        jax.distributed.initialize()

    if jax.process_index() == 0:
        wandb.login(anonymous="never", key=FLAGS.wb_token, verify=True)
        wandb.init(
            project=PROJECT_NAME,
            group=FLAGS.group,
            config={**vars(FLAGS.config)["_fields"], "seed": FLAGS.seed},
            mode="online" if FLAGS.wb_token else "disabled",
            resume="must" if FLAGS.wb_runid else "never",
            id=FLAGS.wb_runid,
        )

    eval_loss = train_loop()

    if jax.process_index() == 0:
        n, d, b, e = get_ndbe()
        table = wandb.Table(
            columns=["Group", "N", "D", "B", "E", "Loss"],
            data=[[FLAGS.group, n, d, b, e, eval_loss]],
        )
        wandb.log({"sweep_table": table})


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
