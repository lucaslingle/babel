from dataclasses import fields
from typing import Any

import chex
import einops
import flax.linen as nn
from flax.linen.transforms import C
import jax
import jax.numpy as jnp
from flax import struct
from flax.linen import partitioning as nnp

from babel.sharding import sharding_constraint
from babel.dims import Dims


MESH_AXES = Dims(X="X", Y="Y", N=None)


@struct.dataclass
class TransformerConfig:
    param_dtype: Any
    dtype: Any
    n_vocab: int
    n_ctx: int
    n_layer: int
    d_model: int
    d_head: int
    n_heads_per_group: int
    ff_multiple: int
    rotary_base: int
    rmsnorm_params: bool
    rmsnorm_eps: float
    qk_norm: bool

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(TransformerConfig)}
        flt = {k: v for k, v in kwargs.items() if k in signature}
        flt.update({k: jnp.dtype(v) for k, v in flt.items() if k.endswith("dtype")})
        return cls(**flt)


def get_initializer(fan_in):
    return jax.nn.initializers.truncated_normal(fan_in**-0.5, lower=-3.0, upper=3.0)


def get_einsum_kwargs(dtype):
    return dict(
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=dtype,
        optimize="optimal",
    )


class RMSNorm(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    suffix: str

    @nn.compact
    def __call__(self, x):
        eps = jnp.array([self.cfg.rmsnorm_eps], dtype=x.dtype)
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1) + eps)
        output = x / rms[..., None]
        if self.cfg.rmsnorm_params:
            output *= self.param(
                "g_" + self.suffix,
                nn.with_partitioning(jax.nn.initializers.ones, MESH_AXES["Y"], self.global_mesh),
                [self.cfg.d_model],
                self.cfg.param_dtype,
            ).astype(self.cfg.dtype)[None, None, ...]
        return output


class QNorm(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    suffix: str = "aq"

    @nn.compact
    def __call__(self, x):
        n_groups = self.cfg.d_model // (self.cfg.n_heads_per_group * self.cfg.d_head)
        n_heads_per_group = self.cfg.n_heads_per_group
        eps = jnp.array([self.cfg.rmsnorm_eps], dtype=x.dtype)
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1) + eps)
        output = x / rms[..., None]
        if self.cfg.rmsnorm_params:
            gains = self.param(
                "g_" + self.suffix,
                nn.with_partitioning(jax.nn.initializers.ones, MESH_AXES["NNN"], self.global_mesh),
                [n_groups, n_heads_per_group, self.cfg.d_head],
                self.cfg.param_dtype,
            ).astype(self.cfg.dtype)
            output *= jnp.expand_dims(jnp.expand_dims(gains, 0), -2)
        return output


class KNorm(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    suffix: str = "ak"

    @nn.compact
    def __call__(self, x):
        n_groups = self.cfg.d_model // (self.cfg.n_heads_per_group * self.cfg.d_head)
        eps = jnp.array([self.cfg.rmsnorm_eps], dtype=x.dtype)
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1) + eps)
        output = x / rms[..., None]
        if self.cfg.rmsnorm_params:
            gains = self.param(
                "g_" + self.suffix,
                nn.with_partitioning(jax.nn.initializers.ones, MESH_AXES["NN"], self.global_mesh),
                [n_groups, self.cfg.d_head],
                self.cfg.param_dtype,
            ).astype(self.cfg.dtype)
            output *= jnp.expand_dims(jnp.expand_dims(gains, 0), -2)
        return output


class RotaryEncoding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    is_queries: bool

    @nn.compact
    def __call__(self, x):
        *_, length, width = x.shape

        positions = jnp.arange(length)
        positions = positions[..., None]  # expand along width axis

        dimensions = jnp.arange(width // 2)  # half each for sin and cos
        ang_freqs = jnp.power(self.cfg.rotary_base, -dimensions / (width // 2))
        ang_freqs = ang_freqs[None, ...]  # expand along length axis

        radians = positions * ang_freqs
        radians = radians[None, None, ...]

        if self.is_queries:
            radians = radians[None, ...]

        cos = jnp.cos(radians).astype(x.dtype)
        sin = jnp.sin(radians).astype(x.dtype)

        even, odd = jnp.split(x, 2, axis=-1)
        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos

        mesh_axes = MESH_AXES["XYNNN"] if self.is_queries else MESH_AXES["XYNN"]
        r = jnp.concatenate([r_even, r_odd], axis=-1)
        r = sharding_constraint(r, mesh_axes, self.global_mesh)
        chex.assert_shape(r, x.shape)
        return r


class CausalMask(nn.Module):
    n_ctx: int
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        i = jnp.arange(self.n_ctx)[..., None]
        j = jnp.arange(self.n_ctx)[None, ...]
        mask = jnp.less(i, j)  # i.e., j > i, indicator masks out non-causal connections
        mask = mask[None, None, None, ...]
        x = x - jnp.array([1e30], dtype=x.dtype) * mask
        x = sharding_constraint(x, MESH_AXES["XYNNN"], self.global_mesh)
        return x


class GroupedQueryAttention(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        n_groups = self.cfg.d_model // (self.cfg.n_heads_per_group * self.cfg.d_head)
        n_heads_per_group = self.cfg.n_heads_per_group
        initializer = get_initializer(fan_in=self.cfg.d_model)
        einsum_kwargs = get_einsum_kwargs(dtype=self.cfg.dtype)

        wq = self.param(
            "w_aq",
            nn.with_partitioning(initializer, MESH_AXES["XY"], self.global_mesh),
            (self.cfg.d_model, self.cfg.d_model),
            self.cfg.param_dtype,
        )
        wk = self.param(
            "w_ak",
            nn.with_partitioning(initializer, MESH_AXES["XY"], self.global_mesh),
            (self.cfg.d_model, self.cfg.d_model // self.cfg.n_heads_per_group),
            self.cfg.param_dtype,
        )
        wv = self.param(
            "w_av",
            nn.with_partitioning(initializer, MESH_AXES["XY"], self.global_mesh),
            (self.cfg.d_model, self.cfg.d_model // self.cfg.n_heads_per_group),
            self.cfg.param_dtype,
        )
        wo = self.param(
            "w_ao",
            nn.with_partitioning(initializer, MESH_AXES["YX"], self.global_mesh),
            (self.cfg.d_model, self.cfg.d_model),
            self.cfg.param_dtype,
        )

        q = jnp.einsum("bti,io->bto", x, wq, **einsum_kwargs)
        k = jnp.einsum("bti,io->bto", x, wk, **einsum_kwargs)
        v = jnp.einsum("bti,io->bto", x, wv, **einsum_kwargs)
        q = einops.rearrange(
            q, "b t (g h d) -> b g h t d",
            g=n_groups,
            h=n_heads_per_group
        )
        k = einops.rearrange(k, "b t (g d) -> b g t d", g=n_groups)
        v = einops.rearrange(v, "b t (g d) -> b g t d", g=n_groups)
        q = sharding_constraint(q, MESH_AXES["XYNNN"], self.global_mesh)
        k = sharding_constraint(k, MESH_AXES["XYNN"], self.global_mesh)
        v = sharding_constraint(v, MESH_AXES["XYNN"], self.global_mesh)

        if self.cfg.qk_norm:
            q = QNorm(self.cfg, self.global_mesh)(q)
            k = KNorm(self.cfg, self.global_mesh)(k)

        rope_kws = dict(cfg=self.cfg, global_mesh=self.global_mesh)
        q = RotaryEncoding(**rope_kws, is_queries=True)(q)
        k = RotaryEncoding(**rope_kws, is_queries=False)(k)

        mult = jnp.array([self.cfg.d_head**-0.25], dtype=self.cfg.dtype)
        s = jnp.einsum("bghid,bgjd->bghij", q * mult, k * mult, **einsum_kwargs)
        s = CausalMask(self.cfg.n_ctx, self.global_mesh)(s)
        p = jax.nn.softmax(s, axis=-1)
        o = jnp.einsum("bghij,bgjd->bghid", p, v, **einsum_kwargs)
        o = einops.rearrange(o, "b g h t d -> b t (g h d)")

        r = jnp.einsum("bti,io->bto", o, wo, **einsum_kwargs)
        r = sharding_constraint(r, MESH_AXES["XNY"], self.global_mesh)
        return r


class PositionwiseFeedforward(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        d_ff = int(self.cfg.ff_multiple * self.cfg.d_model)
        i_initializer = get_initializer(fan_in=self.cfg.d_model)
        o_initializer = get_initializer(fan_in=d_ff)
        einsum_kwargs = get_einsum_kwargs(dtype=self.cfg.dtype)

        wi = self.param(
            "w_fi",
            nn.with_partitioning(i_initializer, MESH_AXES["XY"], self.global_mesh),
            (self.cfg.d_model, d_ff),
            self.cfg.param_dtype,
        )
        wg = self.param(
            "w_fg",
            nn.with_partitioning(i_initializer, MESH_AXES["XY"], self.global_mesh),
            (self.cfg.d_model, d_ff),
            self.cfg.param_dtype,
        )
        wo = self.param(
            "w_fo",
            nn.with_partitioning(o_initializer, MESH_AXES["YX"], self.global_mesh),
            (d_ff, self.cfg.d_model),
            self.cfg.param_dtype,
        )

        h = jnp.einsum("btm,mf->btf", x, wi, **einsum_kwargs)
        g = jnp.einsum("btm,mf->btf", x, wg, **einsum_kwargs)
        h *= jax.nn.silu(g)
        r = jnp.einsum("btf,fm->btm", h, wo, **einsum_kwargs)
        r = sharding_constraint(r, MESH_AXES["XNY"], self.global_mesh)
        return r


class TransformerBlock(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, _):
        kws = dict(cfg=self.cfg, global_mesh=self.global_mesh)

        y = RMSNorm(**kws, suffix="ain")(x)
        y = sharding_constraint(y, MESH_AXES["XNY"], self.global_mesh)
        r = GroupedQueryAttention(**kws)(y)
        r = RMSNorm(**kws, suffix="aout")(r)
        x += r
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        y = RMSNorm(**kws, suffix="fin")(x)
        y = sharding_constraint(y, MESH_AXES["XNY"], self.global_mesh)
        r = PositionwiseFeedforward(**kws)(y)
        r = RMSNorm(**kws, suffix="fout")(r)
        x += r
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        return x, None


class Embedding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        initializer = get_initializer(fan_in=1.0)
        we = self.param(
            "w_e",
            nn.with_partitioning(initializer, MESH_AXES["NY"], self.global_mesh),
            [self.cfg.n_vocab, self.cfg.d_model],
            self.cfg.param_dtype,
        )
        e = jnp.take_along_axis(
            we.astype(self.cfg.dtype)[None, ...],  # 1VM
            x[..., None],  # BT1
            axis=1,
        )
        e = RMSNorm(cfg=self.cfg, global_mesh=self.global_mesh, suffix="eout")(e)
        e = sharding_constraint(e, MESH_AXES["XNY"], self.global_mesh)
        return e


class Unembedding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        initializer = get_initializer(fan_in=self.cfg.d_model)
        einsum_kwargs = get_einsum_kwargs(dtype=self.cfg.dtype)
        wu = self.param(
            "w_u",
            nn.with_partitioning(initializer, MESH_AXES["YN"], self.global_mesh),
            [self.cfg.d_model, self.cfg.n_vocab],
            self.cfg.param_dtype,
        )
        y = RMSNorm(self.cfg, self.global_mesh, "uin")(x)
        u = jnp.einsum("btm,mv->btv", y, wu, **einsum_kwargs)
        u = sharding_constraint(u, MESH_AXES["XNN"], self.global_mesh)
        return u


class Transformer(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, token_ids):
        x = nnp.remat(Embedding)(self.cfg, self.global_mesh)(token_ids)
        x, _ = nn.scan(
            nnp.remat(TransformerBlock),
            length=self.cfg.n_layer,
            variable_axes=dict(params=0, intermediates=0),  # use axis 0 for params,sown
            variable_broadcast=False,  # no variable sharing across layers
            split_rngs=dict(params=True),  # each layer's init shall use a distinct rng
            in_axes=0,  # use n_layer first for inputted kv cache
            out_axes=0,  # use n_layer first for outputted kv cache
            metadata_params={nn.PARTITION_NAME: None},  # no pipeline parallel
        )(self.cfg, self.global_mesh)(x, None)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        x = nnp.remat(Unembedding)(self.cfg, self.global_mesh)(x)
        return x
