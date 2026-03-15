from functools import partial

import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from cg_bg.data.mb.preprocess import collate_fn


class MBDataset(Dataset):
    def __init__(self, datapath: str):
        with np.load(datapath, allow_pickle=True) as data:
            data = dict(data)
            for key in data.keys():
                setattr(self, key, data[key])
        self.kT = 1.0

    def __len__(self):
        return len(self.R)

class MBFLOWDataset(MBDataset):
    def __init__(self, datapath: str):
        super().__init__(datapath)

    def __getitem__(self, idx):
        return {
            "x": self.R[idx]
        }

class MBPMFDataset(MBDataset):
    def __init__(self, datapath: str):
        super().__init__(datapath)

    def __getitem__(self, idx):
        return {
            "x": self.R[idx],
            "force": self.F[idx],
        }


def get_mb_dataloader(
    dataset: MBDataset,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> DataLoader:

    sampler_rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset))
    indices = sampler_rng.choice(all_indices, num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)

    collate_rng = np.random.default_rng(seed + 1)
    templated_collate = partial(collate_fn, rng=collate_rng)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        shuffle=False, 
        drop_last=True, 
        collate_fn=templated_collate
    )

    return dataloader

def get_mb_pmf_dataloader(
    dataset: Dataset,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> DataLoader:

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset))
    indices = rng.choice(all_indices, num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
    )

    return dataloader
