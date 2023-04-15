from typing import Sequence
import torch
from models.utils import TSPModelInput
import random



def _identity(x=None):
    return x


def _nop_collate_fn(x: Sequence):
    return x[0]



def sample_max_n_nodes(min_nodes, max_nodes):
    return random.randint(min_nodes, max_nodes)



def custom_collate_fn(samples: Sequence[TSPModelInput]):
    return TSPModelInput(
            torch.stack([sample.coords for sample in samples]),
            torch.stack([sample.ref_tour for sample in samples]),
            torch.tensor([sample.ref_len for sample in samples], dtype=torch.float32),
            [sample.id for sample in samples]
    )
