import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

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
        self.R0 = np.random.normal(size=self.R.shape)
        self.t = np.random.uniform(size=(self.R.shape[0],1), low=0.0, high=1.0)
        self.xt = (1.0 - self.t) * self.R0 + self.t * self.R
        self.vt = self.R - self.R0

    def __getitem__(self, idx):
        return {
            "x0": self.R0[idx],
            "x1": self.R[idx],
            "vt": self.vt[idx],
            "t": self.t[idx],
            "input": {
                "x": self.xt[idx],
                "t": self.t[idx],
            }
        }
    
class MBPMFDataset(MBDataset):
    def __init__(self, datapath: str):
        super().__init__(datapath)

    def __getitem__(self, idx):
        return {
            "x": self.R[idx],
            "forces": self.F[idx],
        }
    
def get_mb_dataloader(
    dataset: Dataset,
    num_samples: int, 
    batch_size: int,
) -> DataLoader:

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        shuffle=False, 
        drop_last=True,
    )

    return dataloader
