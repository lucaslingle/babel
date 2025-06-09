import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from absl import logging
from optax import tree_utils as otu
from optax._src import transform
from optax._src import base
from optax._src import utils, combine
from collections.abc import Callable
from typing import Any, Optional, Union, NamedTuple

MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


def gradpower(
    power: float = 1.2
) -> base.GradientTransformation:
    """
      Reference:
        Wang et al., `GradPower: Powering Gradients for Faster Language Model Pre-Training
        <https://arxiv.org/abs/2505.24275>`_, 2025
    """

    def update_fn(updates, state, params):
        sign_ = jax.tree.map(jnp.sign, updates)
        abs_ = jax.tree.map(jnp.abs, updates)
        pow_ = jax.tree.map(lambda x: jnp.power(x, power), abs_)
        updates = jax.tree.map(lambda x, y: x * y, sign_, pow_)
        return updates, state

    return base.GradientTransformation(base.init_empty_state, update_fn)



def orthogonalize_matrix(
    x: jax.Array,  # should have shape (n_layer, in_dim, out_dim)
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> jax.Array:

    a, b, c = (3.4445, -4.7750, 2.0315)
    transposed = False
    if x.shape[1] > x.shape[2]:
        x = x.transpose(0, 2, 1)
        transposed = True

    def newton_schulz_iterator(X: jax.Array) -> jax.Array:
        A = X @ X.transpose(0, 2, 1)
        B = b * A + c * A @ A
        return a * X + B @ X

    x /= jnp.linalg.norm(x, axis=(1, 2), keepdims=True) + jnp.array([eps], dtype=x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: newton_schulz_iterator(x), x)
    if transposed:
        x = x.transpose(0, 2, 1)
    return x


def apply_orthogonalization_logic(x, ns_steps, eps, base_scale):
    names, mesh = x.names, x.mesh
    x = x.value
    nl, d1, d2 = x.shape
    x_orth = orthogonalize_matrix(x, ns_steps, eps)
    scale = base_scale * jnp.sqrt(jnp.maximum(d1, d2)) if base_scale > 0.0 else 1.0
    output = x_orth * scale
    return nn.Partitioned(value=output, names=names, mesh=mesh)


def orthogonalize_tree(
    pytree: Any,
    ns_steps,
    eps,
    base_scale,
) -> Any:
    if isinstance(pytree, optax.MaskedNode):
        return pytree
    elif isinstance(pytree, dict):
        return {k: orthogonalize_tree(v, ns_steps, eps, base_scale) for k, v in pytree.items()}
    else:
        return None if pytree is None else apply_orthogonalization_logic(pytree, ns_steps, eps, base_scale)


class MuonState(NamedTuple):
    mu: base.Updates
    

def scale_by_muon(
    momentum: float = 0.95,
    *,
    mu_dtype: Optional[chex.ArrayDType] = None,
    nesterov: bool = True,
    eps: float = 1e-7,
    ns_steps: int = 5,
    base_scale: float = 0.2,
) -> base.GradientTransformation:
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype) # First moment
        return MuonState(mu=mu)

    def update_fn(updates, state, params=None):
        del params

        def momentum_grad(grad, buf):
            return grad + momentum * buf if grad is not None else None

        new_mu = jax.tree.map(momentum_grad, updates, state.mu, is_leaf=lambda x: x is None)
        mu = jax.tree.map(momentum_grad, updates, new_mu, is_leaf=lambda x: x is None) if nesterov else new_mu
        mu_orth = orthogonalize_tree(mu, ns_steps, eps, base_scale)
        return mu_orth, MuonState(mu=otu.tree_cast(new_mu, mu_dtype))

    return base.GradientTransformation(init_fn, update_fn)


def param_labels_func(params):
    def recurse(subtree, path=()):
        if isinstance(subtree, dict):
            return {k: recurse(v, path + (k,)) for k, v in subtree.items()}
        else:
            if path[-1].startswith("g_") or path[-1] in {"w_e", "w_u"}:
                logging.info(f"{path} -> adam")
                return "adam"
            else:
                logging.info(f"{path} -> muon")
                return "muon"
    return recurse(params)


def muon(
    learning_rate: base.ScalarOrSchedule,
    momentum: float = 0.95,
    *,
    mu_dtype: Optional[chex.ArrayDType] = None,
    nesterov: bool = True,
    ns_steps: int = 5,
    base_scale: float = 0.2,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    mask: MaskOrFn = None,
) -> base.GradientTransformationExtraArgs:
    return combine.chain(
        combine.multi_transform(
            transforms={
                "muon": scale_by_muon(
                    momentum=momentum,
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                    base_scale=base_scale,
                ),
                "adam": transform.scale_by_adam(
                    b1=b1,
                    b2=b2,
                    eps=eps,
                    eps_root=0,
                    mu_dtype=mu_dtype,
                )
            },
            param_labels=param_labels_func,
        ),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),  # scales by -lr.
    )
