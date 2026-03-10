from typing import Any
from chemtrain.data import preprocessing
from chemutils.datasets import pepsol
import numpy as np
from chemtrain.quantity import kb
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from cgbg.data.aldp.preprocess import collate_fn, corebeta_mapping

class ALDPDataset(Dataset):
    def __init__(self, datapath: str, n_nodes: int):
        with np.load(datapath, allow_pickle=True) as data:
            data = dict(data)
            if n_nodes == 6:
                data = corebeta_mapping(data)
            for key in data.keys():
                setattr(self, key, data[key])
        self.kT = kb * 300

    def __len__(self):
        return len(self.R)

class ALDPFLOWDataset(ALDPDataset):
    def __init__(self, datapath: str, feat_type: str ="distinguish", n_nodes: int = 10):
        super().__init__(datapath, n_nodes=n_nodes)

        if feat_type == "distinguish":
            self.features = np.arange(self.R.shape[1]).reshape(-1, 1)  # shape: (10, 1)
        elif feat_type == "none":
            self.features = None
        elif feat_type == "species":
            self.features = self.species[0].reshape(-1, 1)
    
    def __getitem__(self, idx):
        return {
            "x": self.R[idx],
            "features": self.features,
        }
    
def get_aldp_pmf_dataset(
    data_path: str,
    scale_R: float,
    scale_U: float,
    fractional: bool,
    train_frac: float,
    n_nodes: int,
    seed: int
) -> dict[str, dict[str, Any]]:
    
    raw = np.load(data_path, allow_pickle=True)
    data = dict(raw)
    if n_nodes == 6:
        data = corebeta_mapping(data)

    train_data, val_data, test_data = preprocessing.train_val_test_split(
        data, train_ratio=0.9, val_ratio=0.1, shuffle=True, shuffle_seed=seed
    )
    splits = {'training': train_data, 'validation': val_data, 'testing': test_data}

    for key, subset in splits.items():
        splits[key] = pepsol.scale_dataset(
            subset, scale_R=scale_R, scale_U=scale_U, fractional=fractional
        )

    # Reduce training size
    n_train = splits['training']['R'].shape[0]
    keep = int(train_frac * n_train)
    for field in splits['training']:
        splits['training'][field] = splits['training'][field][:keep]

    return splits

def get_aldp_dataloader(
    dataset: ALDPDataset,
    num_samples: int, 
    batch_size: int,
) -> DataLoader:

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn
    )

    return dataloader