import os
from typing import Callable, Dict
import torch
from torch.functional import Tensor
from models.utils import TSPModelInput



def map_func(x):
    return TSPModelInput(x['coords'], x['ref_tour'], x['ref_length'], x['id'])
    


class TSPDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path: str,
        mapping_func: Callable[[Dict], TSPModelInput] = None,
    ):
        super().__init__()
        self.mapping_func = mapping_func if mapping_func is not None else map_func
        self.filenames = [x.path for x in os.scandir(path) if x.is_file()]

    
    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        sample = torch.load(self.filenames[idx])
        sample['id'] = self.filenames[idx].split(os.sep)[-1].split('.')[0]
        return self.mapping_func(sample)



class RandomTrainTSPDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            n_steps: int,
            bsz: int,
            n_features: int,
            sample_n_nodes: Callable[[], int],
            device: str,
            max_norm: bool = True
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.bsz = bsz
        self.n_features = n_features
        self.sample_n_nodes = sample_n_nodes
        self.max_norm = max_norm
        self.device = device


    def __len__(self):
        return self.n_steps
    
    
    def __getitem__(self, _):
        n_nodes = self.sample_n_nodes()
        batch = torch.rand((self.bsz, n_nodes, self.n_features), device=self.device)
        if self.max_norm:
            batch /= batch.max(1, keepdim=True)[0]
        return TSPModelInput(batch, None, None)
