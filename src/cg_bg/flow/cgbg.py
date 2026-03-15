from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from hydra_zen.typing import Partial
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cg_bg.flow.ema import EMATrainState
from cg_bg.flow.plot import aldp_plots, mb_plots
from cg_bg.flow.train_utils import plot_train_loss, train_step


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
        pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
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

    x, logp = psampler(
        params=state.params,
        apply_fn=state.apply_fn,
        features=features,
        rng=rng,
    )
    samples = {"R": x, "logp": logp, "kT": dataset.kT}

    species = getattr(dataset, "species", None)
    if species is not None:
        species = species[0]
        species = jnp.broadcast_to(species, (x.shape[0], *species.shape))
        samples["species"] = species

    box = getattr(dataset, "box", None)
    if box is not None:
        box = box[0]
        box = jnp.broadcast_to(box, (x.shape[0], *box.shape))
        samples["box"] = box

    return samples


def reweight(
    raw: dict,
    trainer: Any,
    params: FrozenDict,
) -> dict:

    samples = raw.copy()
    kT = samples["kT"]
    logp = samples["logp"]
    samples["U"] = jnp.zeros((samples["R"].shape[0],))
    if "box" in raw:
        samples["R"] = samples["R"] / samples["box"][0, 0, 0]

    predictions = trainer.predict(dataset=samples, params=params, batch_size=1000)
    energies = predictions["U"]

    log_weights = -energies / kT - logp
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)

    samples_and_weights = {
        **raw,
        "U": energies,
        "log_w": log_weights,
    }

    return samples_and_weights


def plots(samples: dict, task: str, ref_dir: str):
    if "aldp" in task:
        aldp_plots(samples, task, ref_dir)
    elif "mb" in task:
        mb_plots(samples, ref_dir)
