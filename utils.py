import numpy as np
import networkx as nx
from torch import Tensor
import torch
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(levelname)s: %(message)s"
)


def get_tour_coords(coords, tour):
    return coords[torch.arange(len(tour)).view(-1, 1), tour]



def get_tour_len(coords: Tensor, tour: Tensor = None) -> Tensor:
    """Compute the length of a batch of tours.

    Args:
        tour (Tensor): shape (N, L, D)

    Returns:
        Tensor: shape (N), contains the length of each tour in the batch.
    """   
    if tour is not None:
        coords = get_tour_coords(coords, tour)
    diff = torch.diff(coords, dim=1)
    return diff.square().sum(dim=-1).sqrt().sum(dim=-1)



def np2nx(x: np.ndarray):
    G = nx.Graph()
    for i, node in enumerate(x):
        G.add_node(i + 1, pos=node)
        for j, node2 in enumerate(x):
            if i != j:
                d = ((node - node2) ** 2).sum() ** 0.5
                G.add_edge(i + 1, j + 1, weight=d.item())
    return G
