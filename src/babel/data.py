import functools
import math
import os
import posixpath
import time

import blobfile
import datasets
import huggingface_hub
import numpy as np
import jax
import jax.numpy as jnp
import tqdm
import transformers
from absl import logging


HF_PATH = "HuggingFaceFW/fineweb"
HF_NAME = "sample-350BT"
HF_SPLIT = "train"
HF_DATACOL = "text"
SEQUENCES_IN_DATASET = 517_000_000  # a conservative estimate, will drop rest
ARRAY_DTYPE = np.uint16
WRITE_BUFFER_SIZE = 512
TOKENIZER = "meta-llama/Llama-2-7b-hf"
TOKENIZER_BOS = True


@functools.lru_cache(maxsize=1)
def get_tokenizer(hf_token):
    huggingface_hub.login(token=hf_token)
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    if TOKENIZER == "meta-llama/Llama-2-7b-hf":
        tokenizer.pad_token = tokenizer.bos_token
    return tokenizer


def _get_shard_fps(workdir, pcount, pindex):
    remote_fp = posixpath.join(
        workdir, "datasets", HF_PATH, HF_NAME, TOKENIZER, f"{pcount}", f"{HF_SPLIT}-{pindex}.bin"
    )
    local_fp = posixpath.join(
        "/tmp/", f"{HF_SPLIT}-{pindex}.bin"
    )
    return remote_fp, local_fp


def write_dataset(hf_token, local_batch_size, sequence_len, workdir):
    pcount = jax.process_count()
    pindex = jax.process_index()
    remote_fp, local_fp = _get_shard_fps(workdir=workdir, pcount=pcount, pindex=pindex)

    if blobfile.exists(remote_fp):
        logging.info(f"Mem-mapped file exists at {remote_fp}, skipping write...")
        return
    if os.path.exists(local_fp):
        os.remove(local_fp)

    huggingface_hub.login(token=hf_token)
    ds = datasets.load_dataset(
        path=HF_PATH,
        name=HF_NAME,
        split=HF_SPLIT,
        streaming=True,
    )
    tokenizer = get_tokenizer(hf_token)

    def processing_func(examples):
        es = examples[HF_DATACOL]
        es = [e for i, e in enumerate(es) if i % pcount == pindex]
        kws = dict(
            padding="max_length",
            truncation=True,
            max_length=sequence_len + int(TOKENIZER_BOS),
        )
        es = tokenizer(es, **kws)["input_ids"]   # assumes bos prepend by tokenizer
        return dict(token_ids=es)

    processing_bsz = WRITE_BUFFER_SIZE * pcount
    ds = ds.map(
        processing_func,
        batched=True,
        batch_size=processing_bsz,
        remove_columns=list(ds.column_names),
    )

    sequences_per_shard = SEQUENCES_IN_DATASET // pcount
    lcm = math.lcm(WRITE_BUFFER_SIZE, local_batch_size)
    writable_sequences_per_shard = (sequences_per_shard // lcm) * lcm
    ds = ds.take(writable_sequences_per_shard)  # drop rest via lazy op
    ds = ds.iter(batch_size=WRITE_BUFFER_SIZE, drop_last_batch=True)

    n_write_iters = writable_sequences_per_shard // WRITE_BUFFER_SIZE
    array = np.memmap(
        local_fp,
        dtype=ARRAY_DTYPE,
        mode="w+",
        shape=(writable_sequences_per_shard * (sequence_len+int(TOKENIZER_BOS)),)
    )
    offset = 0
    increment = WRITE_BUFFER_SIZE * (sequence_len+int(TOKENIZER_BOS))
    for _ in tqdm.tqdm(range(n_write_iters), desc=f"Writing to {local_fp} with memmap"):
        batch = None
        while batch is None:
            try:
                batch = next(ds)["token_ids"]
            except BaseException as e:
                time.sleep(1)
        array_batch = np.array(batch, dtype=ARRAY_DTYPE).reshape(-1)
        array[offset: offset + increment] = array_batch
        offset += increment
    array.flush()

    logging.info(f"Copying {local_fp} to {remote_fp}")
    blobfile.copy(local_fp, remote_fp, overwrite=True)


def read_dataset(workdir):
    remote_fp, local_fp = _get_shard_fps(
        workdir=workdir,
        pcount=jax.process_count(),
        pindex=jax.process_index(),
    )
    if not blobfile.exists(local_fp):
        logging.info(f"Copying {remote_fp} to {local_fp}")
        blobfile.copy(remote_fp, local_fp, overwrite=True)
    logging.info(f"Reading with np.memmap...")
    array = np.memmap(local_fp, dtype=ARRAY_DTYPE, mode="r")
    return array


def get_dataset(hf_token, local_batch_size, sequence_len, workdir):
    write_dataset(hf_token, local_batch_size, sequence_len, workdir)
    return read_dataset(workdir)


def get_train_batch(shard, local_batch_size, sequence_len, train_step, n_eval_step):
    """
    worked example

    tokens_per_shard = 48
    local_batch_size = 4
    sequence_len = 2
    TOKENIZER_BOS = true
    n_eval_step = 2

    tokens_per_read = 4 * (2 + 1) = 12
    train_tokens_per_shard = 48 - 2 * 12 = 24

    train_step = 0
    offset = 0, so covers first tokens_per_read = 12 tokens.

    train_step = 1
    offset = (1 * 12) % 24 = 12, so covers next tokens_per_read = 12 tokens.

    train_step = 2
    offset = (2 * 12) % 24 = 0, so covers first tokens_per_read = 12 tokens.

    the pattern is that the last
        n_eval_step * tokens_per_read = 2 * 12 = 24
    tokens in shard remain unread by get_train_batch
    """
    tokens_per_shard = shard.shape[0]
    tokens_per_read = local_batch_size * (sequence_len + int(TOKENIZER_BOS))
    train_tokens_per_shard = tokens_per_shard - n_eval_step * tokens_per_read
    assert train_tokens_per_shard > 0

    offset = (train_step * tokens_per_read) % train_tokens_per_shard
    shape = (local_batch_size, sequence_len + int(TOKENIZER_BOS))
    batch = shard[offset:offset+tokens_per_read].reshape(*shape)
    return batch


def get_eval_batch(shard, local_batch_size, sequence_len, eval_step, n_eval_step):
    """
    worked example

    tokens_per_shard = 48
    local_batch_size = 4
    sequence_len = 2
    TOKENIZER_BOS = true
    n_eval_step = 2

    tokens_per_read = 4 * (2 + 1) = 12
    train_tokens_per_shard = 48 - 2 * 12 = 24

    eval_step = 0
    offset = 24 + (0 * 12) = 24

    eval_step = 1
    offset = 24 + (1 * 12) = 36

    eval_step = 2 -> assertion error
    """
    assert eval_step < n_eval_step
    tokens_per_shard = shard.shape[0]
    tokens_per_read = local_batch_size * (sequence_len + int(TOKENIZER_BOS))
    train_tokens_per_shard = tokens_per_shard - n_eval_step * tokens_per_read
    assert train_tokens_per_shard > 0

    offset = train_tokens_per_shard + (eval_step * tokens_per_read)
    shape = (local_batch_size, sequence_len + int(TOKENIZER_BOS))
    batch = shard[offset:offset+tokens_per_read].reshape(*shape)
    return batch


def get_debug_batch(local_batch_size, sequence_len):
    return 42 * jnp.ones(
        shape=(local_batch_size, sequence_len + int(TOKENIZER_BOS)),
        dtype=ARRAY_DTYPE,
    )
