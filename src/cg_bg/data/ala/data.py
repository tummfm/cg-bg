from functools import partial
from typing import Any

import numpy as np
from chemtrain.data import preprocessing
from chemtrain.quantity import kb
from chemutils.datasets import pepsol
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from cg_bg.data.ala.preprocess import collate_fn_com
from cg_bg.utils.standarization import standardize


class ALADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        feat_type: str = "distinguish",
        com_center: bool = True,
    ):

        self.kT = kb * 300

        with np.load(data_path, allow_pickle=True) as data:
            data = dict(data)
            for key in data.keys():
                setattr(self, key, data[key])

        if feat_type == "distinguish":
            self.features = np.arange(self.R.shape[1]).reshape(-1, 1)
        elif feat_type == "none":
            self.features = None
        elif feat_type == "species":
            self.features = self.species[0].reshape(-1, 1)

        if com_center:
            self.R, self.std = standardize(self.R)

    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        return {
            "x": self.R[idx],
            "features": self.features,
        }


def get_ala_pmf_dataset(
    data_path: str,
    scale_R: float = 1.0,
    scale_U: float = 1.0,
    fractional: bool = True,
    train_frac: float = 1.0,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:

    raw = np.load(data_path, allow_pickle=True)
    data = dict(raw)

    train_data, val_data, test_data = preprocessing.train_val_test_split(
        data, train_ratio=0.9, val_ratio=0.1, shuffle=True, shuffle_seed=seed
    )
    splits = {"training": train_data, "validation": val_data}

    for key, subset in splits.items():
        splits[key] = pepsol.scale_dataset(subset, scale_R=scale_R, scale_U=scale_U, fractional=fractional)

    # Reduce training size
    n_train = splits["training"]["R"].shape[0]
    keep = int(train_frac * n_train)
    for field in splits["training"]:
        splits["training"][field] = splits["training"][field][:keep]

    return splits


def get_ala_dataloader_com(
    dataset: ALADataset,
    num_samples: int,
    batch_size: int,
    seed: int,
    shuffle: bool = False,
    drop_last: bool = False,
    mu: float = 0.0,
    sigma: float = 0.1,
    com_center: bool = True,
) -> DataLoader:

    sampler_rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset))
    indices = sampler_rng.choice(all_indices, num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)

    collate_rng = np.random.default_rng(seed + 1)
    templated_collate = partial(
        collate_fn_com,
        rng=collate_rng,
        sigma=sigma,
        mu=mu,
        com_center=com_center,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=templated_collate,
    )

    return dataloader
