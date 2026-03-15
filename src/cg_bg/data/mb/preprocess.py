import jax
import numpy as np


def collate_fn(batch_list: list, rng: np.random.Generator) -> dict[str, np.ndarray]:
    
    batch = jax.tree.map(lambda *leaves: np.stack(leaves), *batch_list)
    
    x1 = batch["x"]
    batch_size = x1.shape[0]

    x0 = rng.standard_normal(size=x1.shape)
    t = rng.uniform(low=0.0, high=1.0, size=(batch_size, 1))

    xt = (1.0 - t) * x0 + t * x1
    vt = x1 - x0

    batch["vt"] = vt
    batch["input"] = {
        "x": xt, 
        "t": t
    }

    return batch