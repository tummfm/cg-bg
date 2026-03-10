from typing import Any, Callable, Optional
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
from functools import partial
from diffrax import diffeqsolve, ODETerm, Tsit5, Dopri5, PIDController, SaveAt
import equinox as eqx

class ODEState(eqx.Module):
    x: jnp.ndarray
    logp: jnp.ndarray

def gaussian_log_density(x: jnp.ndarray, mu: float = 0.0, sigma: float = 1.0) -> jnp.ndarray:
    dim = x.shape[-1]
    sq = jnp.sum((x - mu) ** 2, axis=-1)
    return -0.5 * (dim * jnp.log(2 * jnp.pi * sigma ** 2) + sq / sigma ** 2)

def divergence(
    f: Callable[..., jnp.ndarray], 
    n: int, 
    mean: bool = True
) -> Callable[..., jnp.ndarray]:
    """Return a function that computes divergence of `f` w.r.t. x.

    - If n == 0: compute exact trace via Jacobian.
    - If n > 0: use Hutchinson estimator with `n` random probes.
    The returned function has signature (params, x, features, t, key) when n>0
    and (params, x, features, t, _) when n==0 (the last arg is unused).
    """

    if n == 0:

        @partial(jax.vmap, in_axes=(None, 0, None, None))
        def div(params, x, input_args, _):
            def f_val(x_local):
                out = f(params, x_local, **input_args)
                out = jnp.atleast_1d(out)
                return out

            jac = jax.jacobian(f_val)(x)
            jac = jnp.squeeze(jac)
            if jac.ndim >= 2:
                return jnp.trace(jac)
            return jac

        return div

    if n > 0:

        @partial(jax.vmap, in_axes=(None, 0, None, None))
        def div(params, x, input_args, key):
            def f_val(x_local):
                out = f(params, x_local, **input_args)
                return jnp.atleast_1d(jnp.squeeze(out))

            _, vjp = jax.vjp(f_val, x)

            def vJv(subkey):
                v = jax.random.rademacher(subkey, x.shape, dtype=x.dtype)
                return jnp.vdot(vjp(v)[0], v)

            probes = jax.random.split(key, n)
            ret = jax.vmap(vJv)(probes)
            return ret.mean() if mean else ret

        return div



def get_dynamics(apply_fn: Callable, n: int, mean: bool = True):
    div_fn = divergence(apply_fn, n=n, mean=mean)

    def dynamics(t, state: ODEState, args):
        params, inputs, key = args
        x = state.x
        # t_batch = jnp.broadcast_to(t, (x.shape[0], 1))
        # input_args = {"t": t_batch, **inputs}
        inputs["t"] = t
        v = jax.vmap(lambda p, x: apply_fn(p, x=x, **inputs), in_axes=(None, 0))(params, x)
        div = div_fn(params, x, inputs, key)

        return ODEState(
            x=v.reshape(state.x.shape),
            logp=-div.reshape(state.logp.shape)
        )

    return dynamics

def make_diffrax_solver(method: str):
    if method.lower() == "tsit5":
        return Tsit5()
    elif method.lower() == "dopri5":
        return Dopri5()
    else:
        raise ValueError(f"Unknown diffrax solver: {method}")

def batched_sampler(
    dt0: float,
    params: Any,
    apply_fn: Callable,
    method: str,
    num_batches: int,
    batch_size: int,
    n_nodes: int,
    n_dim: int,
    features: Optional[jnp.ndarray],
    num_z: int,
    mean: bool,
    rng: jax.random.PRNGKey,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    dynamics_fn = get_dynamics(apply_fn, n=num_z, mean=mean)
    solver = make_diffrax_solver(method)
    term = ODETerm(dynamics_fn)
    saveat = SaveAt(t1=True)
    controller = PIDController(rtol=rtol, atol=atol)
    params = eqx.tree_inference(params)
    inputs = {}
    if features is not None:
        inputs["features"] = features
        inputs["training"] = False

    @scan_tqdm(num_batches, desc="Sampling")
    def run_one_batch(rng, _):
        rng, rng_x0, rng_div = jax.random.split(rng, 3)

        x0 = jax.random.normal(rng_x0, (batch_size, n_nodes*n_dim))
        logp0 = gaussian_log_density(x0)

        state0 = ODEState(x=x0, logp=logp0)

        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=dt0,
            y0=state0,
            args=(params, inputs, rng_div),
            saveat=saveat,
            stepsize_controller=controller,
        )

        return rng, sol.ys

    rng, states = jax.lax.scan(
        run_one_batch,
        rng,
        jnp.arange(num_batches)
    )

    x_all = jnp.concatenate(states.x, axis=0).reshape(-1, n_nodes, n_dim)
    x_all = jnp.squeeze(x_all)
    logp_all = jnp.concatenate(states.logp, axis=0).reshape(-1)

    return x_all, logp_all