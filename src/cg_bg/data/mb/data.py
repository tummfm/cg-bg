from functools import partial

import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from cg_bg.data.mb.preprocess import collate_fn


class MBDataset(Dataset):
    def __init__(self, data_path: str):
        with np.load(data_path, allow_pickle=True) as data:
            data = dict(data)
            for key in data.keys():
                setattr(self, key, data[key])
        self.kT = 1.0

    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        item = {"x": self.R[idx]}
        if hasattr(self, "F"):
            item["force"] = self.F[idx]
        return item


def get_mb_dataloader(
    dataset: MBDataset,
    num_samples: int,
    batch_size: int,
    seed: int,
    flow: bool = True,
    shuffle: bool = False,
    drop_last: bool = False,
    mu: float = 25.0,
    sigma: float = 25.0,
) -> DataLoader:

    sampler_rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset))
    indices = sampler_rng.choice(all_indices, num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)

    if flow:
        collate_rng = np.random.default_rng(seed + 1)
        templated_collate = partial(collate_fn, rng=collate_rng, sigma=sigma, mu=mu)

    else:
        templated_collate = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=templated_collate,
    )

    return dataloader
