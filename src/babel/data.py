import functools
import math
import os
import posixpath
import time
import collections

import blobfile
import datasets
import huggingface_hub
import numpy as np
import jax
import tqdm
import transformers
from absl import logging


DatasetConfig = collections.namedtuple("DatasetConfig", [
    "path",
    "name",
    "original_split",
    "datacol",
    "sequences_in_dataset",
    "array_dtype",
    "write_buffer_size",
    "tokenizer",
    "tokenizer_adds_bos",
    "sequence_len",
])


DATASET_CONFIGS = dict(
    fineweb=DatasetConfig(
        path="HuggingFaceFW/fineweb",
        name="sample-10BT", #"sample-350BT",
        original_split="train",
        datacol="text",
        sequences_in_dataset=13_000_000, #517_000_000,
        array_dtype=np.uint16,
        write_buffer_size=512,
        tokenizer="meta-llama/Llama-2-7b-hf",
        tokenizer_adds_bos=True,
        sequence_len=256,
    ),
    commonpile=DatasetConfig(
        path="common-pile/comma_v0.1_training_dataset",
        name=None,
        original_split="train",
        datacol="text",
        sequences_in_dataset=783_000_000,
        array_dtype=np.uint16,
        write_buffer_size=512,
        tokenizer="meta-llama/Llama-2-7b-hf",
        tokenizer_adds_bos=True,
        sequence_len=256,
    ),
    cci3hq=DatasetConfig(
        path="BAAI/CCI3-HQ",
        name=None,
        original_split="train",
        datacol="text",
        sequences_in_dataset=53_000_000,
        array_dtype=np.uint32,
        write_buffer_size=512,
        tokenizer="google/byt5-small",
        tokenizer_adds_bos=False,
        sequence_len=256,  # this leaves 13 billion toks and we arent using that many for sweeps
    )
)


@functools.lru_cache(maxsize=1)
def get_tokenizer(*, dataset_config, hf_token):
    huggingface_hub.login(token=hf_token)
    tokenizer = transformers.AutoTokenizer.from_pretrained(dataset_config.tokenizer)
    if dataset_config.tokenizer == "meta-llama/Llama-2-7b-hf":
        tokenizer.pad_token = tokenizer.bos_token
    if dataset_config.tokenizer == "google/byt5-small":
        tokenizer.bos_token = tokenizer.pad_token
    return tokenizer


def _get_shard_fps(*, dataset_config, split, pcount, pindex, workdir):
    remote_fp = posixpath.join(
        workdir, 
        "datasets", 
        dataset_config.path, 
        dataset_config.name, 
        dataset_config.tokenizer, 
        f"{pcount}", 
        f"{split}-{pindex}.bin",
    )
    local_fp = posixpath.join(
        "/tmp/", 
        f"{split}-{pindex}.bin",
    )
    return remote_fp, local_fp


def write_dataset(*, local_batch_size, dataset_config, split, workdir, hf_token):
    dc = dataset_config
    pcount = jax.process_count()
    pindex = jax.process_index()
    remote_fp, local_fp = _get_shard_fps(
        dataset_config=dataset_config,
        split=split,
        pcount=pcount, 
        pindex=pindex, 
        workdir=workdir
    )

    if blobfile.exists(remote_fp):
        logging.info(f"Mem-mapped file exists at {remote_fp}, skipping write...")
        return
    if os.path.exists(local_fp):
        os.remove(local_fp)

    huggingface_hub.login(token=hf_token)
    ds = datasets.load_dataset(
        path=dc.path,
        name=dc.name,
        split=dc.original_split,
        streaming=True,
    )
    tokenizer = get_tokenizer(dataset_config=dc, hf_token=hf_token)

    if split == "train":
        def filter_func(examples):
            es = examples[dc.datacol]
            es = [e for i, e in enumerate(es) if (i % 100) < 99]
            return dict(filtered_text=es)
    else:
        def filter_func(examples):
            es = examples[dc.datacol]
            es = [e for i, e in enumerate(es) if (i % 100) == 99]
            return dict(filtered_text=es)

    ds = ds.map(
        filter_func,
        batched=True,
        batch_size=1000,
        remove_columns=list(ds.column_names),
    )

    def processing_func(examples):
        es = examples["filtered_text"]
        # es = examples[dc.datacol]
        es = [e for i, e in enumerate(es) if i % pcount == pindex]
        kws = dict(
            padding="max_length",
            truncation=True,
            max_length=dc.sequence_len + int(dc.tokenizer_adds_bos),
        )
        es = tokenizer(es, **kws)["input_ids"]
        return dict(token_ids=es)

    processing_bsz = dc.write_buffer_size * pcount
    ds = ds.map(
        processing_func,
        batched=True,
        batch_size=processing_bsz,
        remove_columns=list(ds.column_names), 
    )

    sequences_in_dataset = dc.sequences_in_dataset
    if split == "train":
        sequences_in_dataset *= 0.99
    else:
        sequences_in_dataset *= 0.01
    sequences_in_dataset = int(sequences_in_dataset)
    logging.info(f"sequences_in_dataset: {sequences_in_dataset}")
    
    sequences_per_shard = sequences_in_dataset // pcount
    lcm = math.lcm(dc.write_buffer_size, local_batch_size)
    writable_sequences_per_shard = (sequences_per_shard // lcm) * lcm
    logging.info(f"writable_sequences_per_shard: {writable_sequences_per_shard}")
    ds = ds.take(writable_sequences_per_shard)  # drop rest via lazy op
    ds = ds.iter(batch_size=dc.write_buffer_size, drop_last_batch=True)

    n_write_iters = writable_sequences_per_shard // dc.write_buffer_size
    logging.info(f"n_write_iters: {n_write_iters}")
    array = np.memmap(
        local_fp,
        dtype=dc.array_dtype,
        mode="w+",
        shape=(writable_sequences_per_shard * (dc.sequence_len+int(dc.tokenizer_adds_bos)),)
    )
    offset = 0
    increment = dc.write_buffer_size * (dc.sequence_len+int(dc.tokenizer_adds_bos))
    for _ in tqdm.tqdm(range(n_write_iters), desc=f"Writing to {local_fp} with memmap"):
        batch = None
        while batch is None:
            try:
                batch = next(ds)["token_ids"]
            except BaseException as e:
                # time.sleep(1)
                logging.error(e)
                raise
        array_batch = np.array(batch, dtype=dc.array_dtype).reshape(-1)
        array[offset: offset + increment] = array_batch
        offset += increment
    array.flush()

    logging.info(f"Copying {local_fp} to {remote_fp}")
    blobfile.copy(local_fp, remote_fp, overwrite=True)


def read_dataset(*, dataset_config, split, workdir):
    remote_fp, local_fp = _get_shard_fps(
        dataset_config=dataset_config,
        split=split,
        pcount=jax.process_count(), 
        pindex=jax.process_index(), 
        workdir=workdir
    )
    if not blobfile.exists(local_fp):
        logging.info(f"Copying {remote_fp} to {local_fp}")
        blobfile.copy(remote_fp, local_fp, overwrite=True)
    logging.info(f"Reading with np.memmap...")
    array = np.memmap(local_fp, dtype=dataset_config.array_dtype, mode="r")
    return array


def get_dataset(*, local_batch_size, dataset_config, split, workdir, hf_token):
    write_dataset(
        local_batch_size=local_batch_size, 
        dataset_config=dataset_config, 
        split=split, 
        workdir=workdir, 
        hf_token=hf_token,
    )
    return read_dataset(
        dataset_config=dataset_config, 
        split=split, 
        workdir=workdir,
    )


def get_batch(*, shard, local_batch_size, dataset_config, step):
    dc = dataset_config
    tokens_per_shard = shard.shape[0]
    tokens_per_read = local_batch_size * (dc.sequence_len + int(dc.tokenizer_adds_bos))

    offset = (step * tokens_per_read) % tokens_per_shard
    shape = (local_batch_size, (dc.sequence_len + int(dc.tokenizer_adds_bos)))
    batch = shard[offset:offset+tokens_per_read].reshape(*shape)
    return batch


def count_batches(*, shard, local_batch_size, dataset_config):
    dc = dataset_config
    tokens_per_shard = shard.shape[0]
    tokens_per_read = local_batch_size * (dc.sequence_len + int(dc.tokenizer_adds_bos))
    return tokens_per_shard // tokens_per_read
