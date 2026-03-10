import jax
import numpy as np

def batch_random_rotation_matrices(batch_size: int):

    q = np.random.normal(size=(batch_size, 4))
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

def apply_random_rotations(batch: np.ndarray):

    bs = batch.shape[0]
    R_batch = batch_random_rotation_matrices(bs)  # (BS, 3, 3)

    batch_offset = np.mean(batch, axis=1, keepdims=True)  # (BS, 1, 3)
    batch_centered = batch - batch_offset  # (BS, num_atoms, 3)

    rotated = np.einsum("bij,bnj->bni", R_batch, batch_centered) + batch_offset

    return rotated

def corebeta_mapping(heavy_beads):
    corebeta = {}
    mapping = np.array([1, 3, 4, 5, 6, 8])
    corebeta['R'] = heavy_beads["R"][:, mapping, :]
    corebeta['F'] = heavy_beads["F"][:, mapping, :]
    corebeta['species'] = heavy_beads["species"][:, mapping]
    corebeta['mask'] = heavy_beads["mask"][:, mapping]
    corebeta['box'] = heavy_beads["box"]
    return corebeta

def flatten_if_3d(arr):
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    return arr

# def collate_fn(batch):
#     batch = jax.tree.map(lambda *leaves: np.stack(leaves), *batch)
#     batch["input"]["x"] = apply_random_rotations(batch["input"]["x"])
#     batch = jax.tree.map(flatten_if_3d, batch)

#     return batch

def collate_fn(batch_list):

    batch = jax.tree.map(lambda *leaves: np.stack(leaves), *batch_list)

    x1 = batch["x"] 
    x1 = apply_random_rotations(x1)
    batch_size = x1.shape[0]
    x0 = np.random.normal(size=x1.shape)
    t = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1))
    xt = (1.0 - t) * x0 + t * x1
    vt = x1 - x0

    xt_flat = xt.reshape(batch_size, -1)
    vt_flat = vt.reshape(batch_size, -1)

    batch["vt"] = vt_flat
    batch["input"] = {"x": xt_flat, "features": batch["features"], "t": t.squeeze(axis=-1)}
    
    return batch