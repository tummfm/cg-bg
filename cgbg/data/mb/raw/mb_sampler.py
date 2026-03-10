import jax
from jax import numpy as jnp
from jax import grad, random, lax, vmap
import numpy as onp
import os
import argparse
from tqdm import trange
from functools import partial

# Argument parser for device selection
parser = argparse.ArgumentParser()
parser.add_argument(
    '--device',
    type=str,
    help='GPU or MIG UUID to use for training'
)
args, _ = parser.parse_known_args()
print(f"Running on device: {args.device}")

# Set environment variables based on parsed arguments
if args.device:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

class MullerBrownLangevinMD:
    def __init__(self, dt=0.1, steps=100_000, mass=1.0, gamma=0.1, temperature=1.0, seed=0, biased=True):
        self.dt = dt
        self.steps = steps
        self.mass = mass
        self.gamma = gamma
        self.kT = temperature
        self.key = random.PRNGKey(seed)
        self.biased = biased
        self.noise_scale = jnp.sqrt(2 * gamma * temperature / mass)

    def force(self, xy, biased=True):
        return -grad(lambda xy_: potential(xy_, biased))(xy)

    @partial(jax.jit, static_argnums=0)
    def simulate_single_traj(self, xy0, v0, key):
        def step_fn(carry, _):
            xy, v, key = carry
            key, subkey = random.split(key)
            f_biased = self.force(xy, biased=self.biased)

            noise = self.noise_scale * random.normal(subkey, shape=(2,))
            v_new = v + (self.dt * f_biased / self.mass) - self.gamma * v * self.dt + noise * jnp.sqrt(self.dt)
            xy_new = xy + v_new * self.dt

            e_unbiased = potential(xy_new, biased=False)
            f_unbiased = self.force(xy_new, biased=False)
            
            return (xy_new, v_new, key), (xy_new, v_new, f_unbiased, e_unbiased)

        init_state = (jnp.asarray(xy0), jnp.asarray(v0), key)
        _, (positions, velocities, force, energies) = lax.scan(step_fn, init_state, None, length=self.steps)
        positions = positions[-1, :]
        velocities = velocities[-1, :]
        force = force[-1, :]
        energies = energies[-1]
        return positions, velocities, force, energies

    def run_multiple(self, n_traj, n_samples=100_000):
        self.key, key1, key2, key3 = random.split(self.key, 4)
        # xy0s = jnp.asarray([-0.55828035, 1.44169]).reshape(n_traj, 2)  # Initial positions
        keys = random.split(key1, (n_samples, n_traj))
        xy0s = random.uniform(key2, shape=(n_traj,2), minval=10.0, maxval=50.0)
        v0s = random.normal(key3, shape=(n_traj, 2)) * jnp.sqrt(self.kT / self.mass)
        batched_sim = vmap(self.simulate_single_traj, in_axes=(0, 0, 0))

        positions = []
        velocities = []
        forces = []
        energies = []
        for i in trange(n_samples):
            key = keys[i]
            pos, vel, f, ene = batched_sim(xy0s, v0s, key)
            positions.append(pos)
            velocities.append(vel)
            forces.append(f)
            energies.append(ene)
            xy0s, v0s = pos, vel  # Update initial conditions for the next trajectory
        self.positions = jnp.array(positions)
        self.velocities = jnp.array(velocities)
        self.forces = jnp.array(forces)
        self.energies = jnp.array(energies)
        return self.positions, self.velocities, self.forces, self.energies

    def save_multiple(self, filename, **kwargs):
        onp.savez(filename, **{k: onp.asarray(v) for k, v in kwargs.items()})
    
def potential(xy, biased=True):
        x, y = xy
        term1 = -17.3 * jnp.exp(-0.0039 * (x - 48)**2 - 0.0391 * (y - 8)**2)
        term2 = -8.7 * jnp.exp(-0.0039 * (x - 32)**2 - 0.0391 * (y - 16)**2)
        term3 = -14.7 * jnp.exp(-0.0254 * (x - 24)**2 + 0.043 * (x - 24) * (y - 32) - 0.0254 * (y - 32)**2)
        term4 = 1.3 * jnp.exp(0.00273 * (x - 16)**2 + 0.0023 * (x - 16) * (y - 24) + 0.00273 * (y - 24)**2)
        
        if biased:
            x0 = 32.0
            width = 5.0
            height = -4.0
            bias_potential = height * jnp.exp(-(x - x0)**2 / (2 * width**2))
            return term1 + term2 + term3 + term4 + bias_potential

        return term1 + term2 + term3 + term4


if __name__ == "__main__":
    n_traj = 10
    steps = 100
    n_samples = 50_000
    biased = False
    kT0 = 1.0

    sim = MullerBrownLangevinMD(steps=steps, biased=biased, temperature=kT0)
    positions, velocities, forces, energies = sim.run_multiple(n_traj=n_traj, n_samples=n_samples)

    file_prefix = "mb"
    if biased:
        file_prefix += "_biased"
    else:
        file_prefix += "_unbiased"
    filename = f"{file_prefix}.npz"

    sim.save_multiple(
        filename=filename,
        positions=positions,
        forces = forces,
        energies=energies,
    )
    print(f"Langevin dynamic simulation trajectories saved to {filename}")