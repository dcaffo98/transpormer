from random import random
from sys import stdout
from uuid import uuid4
import numpy as np
import networkx as nx
from torch import Tensor
from dataclasses import dataclass
import torch
import os
import logging
from typing import Union, Sequence
from multiprocessing.pool import Pool
from multiprocessing import Manager
import pathlib
from functools import partial
import ctypes

from models.utils import TSPModelInput


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(levelname)s: %(message)s"
)



def random_sample(n_nodes, n_features):
    x = torch.rand(n_nodes, n_features)
    x /= x.max(0)[0]
    G = np2nx(x)
    tsp = nx.approximation.traveling_salesman_problem
    ref_tour = torch.tensor(tsp(G, cycle=True)) - 1
    ref_len = get_tour_len(x.unsqueeze(0), ref_tour.unsqueeze(0))
    return TSPModelInput(x, ref_tour, ref_len.item())



def mp_save_random_sample(_, path, n_nodes, n_features, n_completed, n):
        sample = random_sample(n_nodes, n_features)
        try:
            assert len(sample.ref_tour) == n_nodes + 1
        except AssertionError:
            # TODO: fix this
            logger.warning(f"Error: tour len is {len(sample.ref_tour)} rather than {n_nodes + 1}. Skipping...")
            return
        torch.save(dict(
            coords=sample.coords,
            ref_tour=sample.ref_tour,
            ref_length=sample.ref_len
        ), os.path.join(path, f"{uuid4()}.pt"))
        n_completed.value += 1
        if random() > 0.99:
            i = n_completed.value
            stdout.write(f"[{i}/{n}] done ({(i / n * 100):.2f}%)\n")



def create_random_dataset(path, n, n_nodes, n_features, n_processes=4):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
    m = Manager()
    n_completed = m.Value(ctypes.c_ulong, 0)    

    func = partial(mp_save_random_sample, path=path, n_nodes=n_nodes, n_features=n_features, n_completed=n_completed, n=n)

    with Pool(n_processes) as P:
        P.map(func, range(n), chunksize=n // n_processes) 

    effective = len(os.listdir(path))
    while effective < n:
        logger.warning(f"Requested {n}\t\tCompleted:{effective}\nGenerating other {n - effective} samples in single process...")
        for _ in range(effective, n):
            mp_save_random_sample(None, path, n_nodes, n_features, n_completed, n)   
        effective = len(os.listdir(path))    

    logger.info('Done!')



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



if __name__ == '__main__':
    create_random_dataset('tsp_data/dummy', int(1e2), 50, 2, 6)
