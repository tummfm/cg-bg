import os
import jax
import jax.numpy as jnp
from typing import Callable
from flax.training.train_state import TrainState
import pickle
import orbax.checkpoint as ocp
from tqdm import tqdm
import matplotlib.pyplot as plt
import optax

@jax.jit
def train_step(
    state: TrainState,
    batch: dict
) -> tuple[TrainState, float]:
    
    def loss_wrap(params):
        return loss_fn(params, state.apply_fn, batch)
    
    loss, grads = jax.value_and_grad(loss_wrap)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss

def loss_fn(
    params: dict, 
    apply_fn: Callable, 
    batch: dict
) -> jnp.ndarray:
    
    def energy_fn(x):
        return apply_fn(params, x)
    
    x = batch["x"]
    fx_true = batch["force"]
    _, g_vjp = jax.vjp(energy_fn, x)
    fx_pred = -g_vjp(jax.numpy.ones_like(x))[0]
    loss = jnp.mean((fx_pred - fx_true) ** 2)

    return loss

class MBForceMatching:
    def __init__(self, dataloader, model, learning_rate, kT, rng):

        self.dataloader = dataloader
        self.model = model
        self.learning_rate = learning_rate
        self.rng = rng
        self.kT = kT

    def create_train_state(self):
        dummy_batch = next(iter(self.dataloader))["x"]
        params = self.model.init({"params": self.rng}, dummy_batch)
        tx = optax.adam(self.learning_rate)
        init_state = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx,)
        return init_state

    def train(self, epochs):
        state = self.create_train_state()
        epoch_losses = []
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        for _ in pbar:
            batch_losses = []

            for batch in self.dataloader:
                batch = jax.tree.map(jnp.asarray, batch)
                state, loss = train_step(state, batch)
                batch_losses.append(loss.item())

            avg_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(avg_loss)
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

        self.state = state
        self.epoch_losses = epoch_losses
        self.final_params = state.params

        save_path = os.path.abspath("pmf_params.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.final_params, f)
        print(f"Checkpoint saved to {save_path}")

    @staticmethod
    def V(x, y):
        term1 = -17.3 * jnp.exp(-0.0039 * (x - 48)**2 - 0.0391 * (y - 8)**2)
        term2 = -8.7 * jnp.exp(-0.0039 * (x - 32)**2 - 0.0391 * (y - 16)**2)
        term3 = -14.7 * jnp.exp(-0.0254 * (x - 24)**2 + 0.043 * (x - 24) * (y - 32) - 0.0254 * (y - 32)**2)
        term4 = 1.3 * jnp.exp(0.00273 * (x - 16)**2 + 0.0023 * (x - 16) * (y - 24) + 0.00273 * (y - 24)**2)
        return term1 + term2 + term3 + term4

    def evaluate(self):
        
        x_vals = jnp.linspace(0, 50, 300)
        y_vals = jnp.linspace(0, 50, 300)
        
        @jax.jit
        def calculate_true_pmf(xi_vals, yi_vals):
            def V_cg(xi):
                V_y = jax.vmap(lambda y: self.V(xi, y))(yi_vals)
                return -self.kT * jnp.log(jnp.trapezoid(jnp.exp(- V_y / self.kT), yi_vals))
            return jax.vmap(V_cg)(xi_vals)
        
        V_eff = calculate_true_pmf(x_vals, y_vals)
        V_eff -= jnp.min(V_eff)

        x_input = x_vals.reshape(-1, 1)
        energy = self.predict(x_input)
        energy -= jnp.min(energy)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ax1.plot(x_vals, energy, lw=2, color="blue", label="Learned PMF")
        ax1.plot(x_vals, V_eff, lw=2, color="red", linestyle="--", label="True PMF")
        ax1.set_xlabel("x")
        ax1.set_ylabel("PMF")
        ax1.set_title("Learned vs True PMF")
        ax1.legend()
        ax2.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss per Epoch")
        plt.tight_layout()
        save_file = os.path.abspath("pmf_evaluation.png")
        plt.savefig(save_file)
        plt.close()
        print(f"Evaluation plot saved to {save_file}")

    def predict(self, dataset, params=None, batch_size=None):
        dataset = jax.tree.map(jnp.asarray, dataset)
        if params is None:
            params = self.final_params
        R = dataset["R"]
        n_samples = R.shape[0]
        if batch_size is None:
            batch_size = n_samples

        def batch_fn(batch_data):
            return self.model.apply(params, batch_data)

        remainder = n_samples % batch_size
        if remainder != 0:
            pad_len = batch_size - remainder
            padding = jnp.zeros((pad_len, *R.shape[1:]), dtype=R.dtype)
            R_padded = jnp.concatenate([R, padding], axis=0)
        else:
            pad_len = 0
            R_padded = R
        num_batches = R_padded.shape[0] // batch_size
        reshaped_inputs = R_padded.reshape(num_batches, batch_size, *R.shape[1:])
        energies_batched = jax.lax.map(batch_fn, reshaped_inputs)
        energies = energies_batched.reshape(-1)
        if pad_len > 0:
            energies = energies[:n_samples]

        return {"U": energies}

