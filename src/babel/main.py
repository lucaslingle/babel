import functools
import posixpath
import time

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
from babel.data import get_train_batch
from babel.data import get_eval_batch
from babel.data import TOKENIZER_BOS
from babel.model import MESH_AXES
from babel.model import TransformerConfig
from babel.model import Transformer
from babel.optim import muon
from babel.optim import gradpower
from babel.sharding import get_namedsharding
from babel.sharding import sharding_constraint
from babel.sharding import to_global_array

FLAGS = flags.FLAGS
PROJECT_NAME = "babel_gradpower_ablation_part1"
config_flags.DEFINE_config_file("config", None, "Config file", lock_config=False)
flags.DEFINE_string("workdir", None, "Working directory (GCS or local)")
flags.DEFINE_string("group", None, "Group name for experiment")
flags.DEFINE_integer("seed", 0, "Experiment rng seed")
flags.DEFINE_string("hf_token", None, "API key for HuggingFace")
flags.DEFINE_string("wb_token", None, "API key for Weights and Biases")
flags.DEFINE_string("wb_runid", None, "Run ID for Weights and Biases")
flags.DEFINE_bool("tpu", True, "Use TPU cluster?")
flags.mark_flags_as_required(["config", "workdir", "group", "hf_token", "wb_token"])


@functools.lru_cache(maxsize=1)
def get_transformer_config():
    return TransformerConfig.create(
        **vars(FLAGS.config)["_fields"],
        n_vocab=len(get_tokenizer(FLAGS.hf_token)),
    )


@functools.lru_cache(maxsize=1)
def get_global_mesh():
    return jax.sharding.Mesh(
        devices=jmu.create_device_mesh(
            mesh_shape=(FLAGS.config.n_mesh_rows, FLAGS.config.n_mesh_cols),
            devices=jax.devices(),
        ),
        axis_names=("X", "Y"),  # using 2D-finalized from GSPMD paper
    )


def get_init_rng():
    return jax.random.PRNGKey(FLAGS.seed)


def get_params(rng, global_mesh):
    config = get_transformer_config()
    inputs = jnp.zeros(dtype=jnp.uint32, shape=[1, config.n_ctx])
    params = Transformer(config, global_mesh).init({"params": rng}, inputs)["params"]
    return params


def get_schedule():
    warmup_steps = FLAGS.config.n_warmup_step
    annealing_steps = FLAGS.config.n_pretrain_step - FLAGS.config.n_warmup_step
    end = FLAGS.config.lr_schedule_end_frac
    warmup = optax.linear_schedule(0.0, 1.0, transition_steps=warmup_steps)
    if FLAGS.config.lr_schedule_name == "linear":
        annealing = optax.linear_schedule(1.0, end, transition_steps=annealing_steps)
    elif FLAGS.config.lr_schedule_name == "cosine":
        annealing = optax.cosine_decay_schedule(1.0, alpha=end, decay_steps=annealing_steps)
    else:
        raise NotImplementedError
    return optax.join_schedules([warmup, annealing], boundaries=[warmup_steps])


def get_optimizer():
    common_kwargs = dict(
        b1=FLAGS.config.optim_beta1,
        b2=FLAGS.config.optim_beta2,
        eps=FLAGS.config.optim_eps,
        mu_dtype=FLAGS.config.dtype,
        weight_decay=0.0 if FLAGS.config.wd_indep else FLAGS.config.wd_lam,
    )
    if FLAGS.config.optim_name == "adamw":
        return optax.adamw(FLAGS.config.lr_eta, **common_kwargs)
    elif FLAGS.config.optim_name == "lion":
        del common_kwargs["eps"]
        return optax.lion(FLAGS.config.lr_eta, **common_kwargs)
    elif FLAGS.config.optim_name == "muon":
        return muon(FLAGS.config.lr_eta, **common_kwargs)
    else:
        raise NotImplementedError


def grad_transform_factory():
    chain = []
    if FLAGS.config.grad_clip > 0.0:
        chain.append(optax.clip_by_global_norm(FLAGS.config.grad_clip))
    if FLAGS.config.grad_power != 1.0:
        chain.append(gradpower(FLAGS.config.grad_power))
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
    ns = FLAGS.config.n_pretrain_step
    bsz = FLAGS.config.tokens_per_global_batch

    n = nl * (4 * dm ** 2 + ff_proj_ct * dm * dff)
    d = ns * bsz
    b = bsz
    e = FLAGS.config.lr_eta
    return n, d, b, e


def get_modelname():
    n, d, b, e = get_ndbe()
    return f"{FLAGS.group}_{n}_{d}_{b}_{e}"


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
    if TOKENIZER_BOS:
        # in this case, the array shape is (bsz, seqlen+1), and we slice only,
        # obtaining inp, tgt shapes of (bsz, seqlen).
        inp, tgt = batch[:, 0:-1], batch[:, 1:]
    else:
        # in this case, the array shape is (bsz, seqlen), and we slice and pad
        # obtaining inp, tgt shapes of (bsz, seqlen).
        bos_token_id = get_tokenizer(FLAGS.hf_token).bos_token_id
        if bos_token_id is None:
            bos_token_id = get_tokenizer(FLAGS.hf_token).pad_token_id
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
    mask = get_loss_mask(tgt, pad_id=get_tokenizer(FLAGS.hf_token).pad_token_id)
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
    n_ctx = FLAGS.config.n_ctx
    local_batch_size = FLAGS.config.tokens_per_global_batch // (n_host * n_ctx)
    ds_shard = get_dataset(
        hf_token=FLAGS.hf_token,
        local_batch_size=local_batch_size,
        sequence_len=n_ctx,
        workdir=FLAGS.workdir,
    )

    eval_loss = None
    t0 = time.perf_counter()
    for step in range(start_step, FLAGS.config.n_pretrain_step):
        batch = get_train_batch(
            shard=ds_shard,
            local_batch_size=local_batch_size,
            sequence_len=n_ctx,
            train_step=step,
            n_eval_step=FLAGS.config.n_eval_step,
        )
        batch = to_global_array(batch, global_mesh)
        state, metrics = train_step_op(state, batch)

        if (step + 1) % FLAGS.config.n_log_step == 0:
            t0 = train_logging_op(metrics, step, t0)

        if (step + 1) % FLAGS.config.n_checkpoint_step == 0:
            eval_loss = eval_loop(state)
            eval_loss_logging_op(eval_loss, step)
            do_save(mgr, state, step)

    return eval_loss


def eval_loop(state):
    global_mesh = get_global_mesh()

    n_host = jax.process_count()
    n_ctx = FLAGS.config.n_ctx
    local_batch_size = FLAGS.config.tokens_per_global_batch // (n_host * n_ctx)
    ds_shard = get_dataset(
        hf_token=FLAGS.hf_token,
        local_batch_size=local_batch_size,
        sequence_len=n_ctx,
        workdir=FLAGS.workdir,
    )

    loss_terms = []
    mask_terms = []
    for step in range(0, FLAGS.config.n_eval_step):
        batch = get_eval_batch(
            shard=ds_shard,
            local_batch_size=local_batch_size,
            sequence_len=n_ctx,
            eval_step=step,
            n_eval_step=FLAGS.config.n_eval_step,
        )
        batch = to_global_array(batch, global_mesh)
        metrics = eval_step_op(state, batch)
        eval_step_logging_op(metrics, step)
        loss_terms.append(metrics["loss_term_avg"])
        mask_terms.append(metrics["loss_mask_avg"])

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
