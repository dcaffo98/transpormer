from typing import Union, Sequence
from dataclasses import dataclass
from torch import Tensor



@dataclass
class TSPModelInput:
    coords: Tensor
    ref_tour: Tensor
    ref_len: Union[float, Tensor]
    id: Union[str, Sequence[str]]= None

    def __len__(self):
        return 1 if len(self.coords.shape) < 3 else len(self.coords)



@dataclass
class TSPModelOutput:
    tour: Tensor
    sum_log_probs: Tensor
    attn_matrix: Tensor = None
