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


NS_COEFS = [
    [8.28721201814563, -23.595886519098837, 17.300387312530933],
    [4.107059111542203, -2.9478499167379106, 0.5448431082926601],
    [3.9486908534822946, -2.908902115962949, 0.5518191394370137],
    [3.3184196573706015, -2.488488024314874, 0.51004894012372],
    [2.300652019954817, -1.6689039845747493, 0.4188073119525673],
    [1.891301407787398, -1.2679958271945868, 0.37680408948524835],
    [1.8750014808534479, -1.2500016453999487, 0.3750001645474248],
    [1.875, -1.25, 0.375], # subsequent coeffs equal this numerically
    [1.875, -1.25, 0.375], # subsequent coeffs equal this numerically
    [1.875, -1.25, 0.375], # subsequent coeffs equal this numerically
]
NS_COEFS = [[a / 1.01, b / 1.01**3, c / 1.01**5] for (a, b, c) in NS_COEFS[:-1]] + [NS_COEFS[-1]]


def orthogonalize_matrix(
    x: jax.Array,  # should have shape (n_layer, in_dim, out_dim)
    ns_steps: int,
    eps: float = 1e-7,
) -> jax.Array:

    a, b, c = (3.4445, -4.7750, 2.0315)
    transposed = False
    if x.shape[1] > x.shape[2]:
        x = x.transpose(0, 2, 1)
        transposed = True

    def newton_schulz_iterator(X: jax.Array, abc: jax.Array) -> jax.Array:
        A = X @ X.transpose(0, 2, 1)
        B = abc[1] * A + abc[2] * A @ A
        return abc[0] * X + B @ X

    x /= jnp.linalg.norm(x, axis=(1, 2), keepdims=True) + jnp.array([eps], dtype=x.dtype)
    ns_coefs = jnp.array(NS_COEFS[0:ns_steps])
    x, _ = jax.lax.scan(lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coefs)
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
    ns_steps: int,
    momentum: float = 0.95,
    *,
    mu_dtype: Optional[chex.ArrayDType] = None,
    nesterov: bool = True,
    eps: float = 1e-7,
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
    ns_steps: int,
    momentum: float = 0.95,
    *,
    mu_dtype: Optional[chex.ArrayDType] = None,
    nesterov: bool = True,
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
