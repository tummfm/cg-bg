import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple
from torch.utils.data import Dataset, DataLoader
from hydra_zen.typing import Partial
from tqdm import tqdm
from cgbg.flow.ema import EMATrainState
from cgbg.flow.train_utils import train_step, plot_train_loss
from cgbg.flow.plot import aldp_plots, mb_plots
from chemtrain.quantity import kb

def train(
    epochs: int,
    dataloader: DataLoader,
    state: EMATrainState,
) -> Tuple[EMATrainState, jnp.ndarray]:

    epoch_losses = []
    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for _ in pbar:
        batch_losses = []

        for batch in dataloader:
            batch = jax.tree.map(jnp.asarray, batch)
            state, loss = train_step(state, batch)
            batch_losses.append(loss)

        avg_loss = jnp.mean(jnp.array(batch_losses))
        pbar.set_postfix({
            "loss": f"{avg_loss:.6f}"
        })
        epoch_losses.append(avg_loss)
    epoch_losses = jnp.array(epoch_losses)
    plot_train_loss(epoch_losses)
    
    return state

def sample(
    dataset: Dataset, 
    psampler: Partial[Callable],
    state: EMATrainState, 
    rng: jax.random.PRNGKey
) -> dict:

    features = dataset[0].get("features", None)
    species = getattr(dataset, "species", None)
    if species is not None:
        species = species[0]
    box = getattr(dataset, "box", None)
    if box is not None:
        box = box[0]

    x, logp = psampler(
        params=state.params,
        apply_fn=state.apply_fn,
        features=features,
        rng=rng,
    )
    samples = {"R": x, "logp": logp, "kT": jnp.full((x.shape[0],), dataset.kT)}

    if species is not None:
        species = jnp.broadcast_to(species, (x.shape[0], *species.shape))
        samples["species"] = species

    if box is not None:
        box = jnp.broadcast_to(box, (x.shape[0], *box.shape))
        samples["box"] = box

    return samples

def reweight(
    raw: dict,
    trainer: Any,
    params: dict,
) -> dict:
    
    samples = raw.copy()
    if "box" in raw:
        samples["R"] = samples["R"] / samples["box"][0, 0, 0]
    kT = kb * 300
    logp = samples["logp"]
    samples["U"] = jnp.zeros((samples["R"].shape[0],))

    predictions = trainer.predict(dataset=samples, params=params, batch_size=1000)
    energies = predictions["U"]
    
    log_weights = -energies / kT - logp
    mask = log_weights < jnp.percentile(log_weights, 99)
    log_weights = jnp.where(mask, log_weights, -jnp.inf)
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights)
    
    samples_and_weights = {
        **raw,
        "U": energies,
        "w": weights,
    }

    return samples_and_weights

def plots(samples: dict, task: str, work_dir: str):
    if "aldp" in task:
        aldp_plots(samples, task, work_dir)
    elif "mb" in task:
        mb_plots(samples, work_dir)