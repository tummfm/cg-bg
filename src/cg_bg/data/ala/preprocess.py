import jax
import numpy as np


def batch_random_rotation_matrices(batch_size: int, rng: np.random.Generator) -> np.ndarray:

    q = rng.standard_normal(size=(batch_size, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.empty((batch_size, 3, 3))
    R[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[:, 0, 1] = 2 * x * y - 2 * w * z
    R[:, 0, 2] = 2 * x * z + 2 * w * y
    R[:, 1, 0] = 2 * x * y + 2 * w * z
    R[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[:, 1, 2] = 2 * y * z - 2 * w * x
    R[:, 2, 0] = 2 * x * z - 2 * w * y
    R[:, 2, 1] = 2 * y * z + 2 * w * x
    R[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2

    return R


def apply_random_rotations(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:

    bs = batch.shape[0]
    R_batch = batch_random_rotation_matrices(bs, rng)  # (BS, 3, 3)

    batch_offset = np.mean(batch, axis=1, keepdims=True)  # (BS, 1, 3)
    batch_centered = batch - batch_offset  # (BS, num_atoms, 3)

    rotated = np.einsum("bij,bnj->bni", R_batch, batch_centered) + batch_offset

    return rotated


def apply_com_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    num_atoms = x.shape[1]
    std = 1.0 / np.sqrt(num_atoms)
    noise = rng.standard_normal(size=(x.shape[0], 1, x.shape[2])) * std
    return x + noise


def collate_fn_com(
    batch_list: list,
    rng: np.random.Generator,
    sigma: float,
    mu: float,
    com_center: bool = True,
) -> dict[str, np.ndarray]:

    batch = jax.tree.map(lambda *leaves: np.stack(leaves), *batch_list)

    x1 = batch["x"]
    if com_center:
        x1 = apply_com_noise(x1, rng=rng)

    x1 = apply_random_rotations(x1, rng=rng)

    batch_size = x1.shape[0]
    x0 = rng.standard_normal(size=x1.shape) * sigma + mu

    t = rng.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1))

    xt = (1.0 - t) * x0 + t * x1
    vt = x1 - x0

    xt_flat = xt.reshape(batch_size, -1)
    vt_flat = vt.reshape(batch_size, -1)

    batch["vt"] = vt_flat
    batch["input"] = {
        "x": xt_flat,
        "features": batch["features"],
        "t": t.squeeze(axis=-1),
    }

    return batch
