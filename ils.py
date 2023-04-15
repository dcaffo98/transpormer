from math import ceil
from typing import Callable, Union
import numpy as np
import torch



def batch_perturb_constrained(tour: torch.Tensor, whitelist_idxs: torch.Tensor):
    out = tour.detach().clone().to(whitelist_idxs.device)
    permuted = whitelist_idxs[:, torch.randperm(whitelist_idxs.shape[1])]
    zero2bsz = torch.arange(tour.shape[0], device=whitelist_idxs.device).view(-1, 1)
    out[zero2bsz, whitelist_idxs] = out[zero2bsz, permuted]
    return out


    
def batch_perturb(tour: torch.Tensor, nodes: int):
    whitelist_idxs = torch.randperm(tour.shape[1] - 2)[:nodes] + 1
    whitelist_idxs = whitelist_idxs.expand(tour.shape[0], -1)
    return batch_perturb_constrained(tour, whitelist_idxs)



def batch_ils(
    objective: Callable, 
    coords,
    start, 
    n_restarts: int, 
    n_iterations: int, 
    whitelist_idxs=None,
    max_perturbs=None
) -> Union[np.array, float]:

    device = coords.device
    best = start.to(device)
    best_eval = objective(coords, start).to(device)
    bsz, nodes = start.shape
    zero2bsz = torch.arange(bsz, device=device).view(-1, 1)
    if whitelist_idxs is None:
        max_perturbs = max_perturbs if max_perturbs is not None else (nodes - 1)
        to_perturb = torch.randint(ceil((nodes - 1) * 0.1), max_perturbs, (1,)).item()
        whitelist_idxs = torch.randperm(nodes - 2)[:to_perturb] + 1
        whitelist_idxs = whitelist_idxs.to(device).expand(bsz, -1)
        
   
    for i in range(n_restarts):
        # init perturbation
        start = batch_perturb_constrained(best, whitelist_idxs)
        candidate_tour = start.clone()

        # perform n_iterations swaps
        for _ in range(n_iterations):
            idxs = torch.randperm(whitelist_idxs.shape[1])[:2].expand(bsz, -1)
            idxs = whitelist_idxs[zero2bsz, idxs]
            candidate_tour[zero2bsz, idxs] = candidate_tour[zero2bsz, idxs.flip(-1)]
            candidate_eval = objective(coords, candidate_tour)
            improve_mask = candidate_eval < best_eval
            best[improve_mask] = candidate_tour[improve_mask]
            best_eval[improve_mask] = candidate_eval[improve_mask]

    return best, best_eval







if __name__ == '__main__':
    bsz, nodes = 512, 50

    tour = [torch.randperm(nodes) for _ in range(bsz)]
    tour = torch.stack(tour)
    tour = torch.concat([tour, tour[:, 0:1]], dim=1)
    whitelist_idxs = torch.randint(1, nodes, (bsz, 10))
    perturbed = batch_perturb_constrained(tour, whitelist_idxs)
    assert all([len(set(x.tolist())) for x in perturbed])
    nodes_to_perturb = 5
    perturbed = batch_perturb(tour, nodes_to_perturb)
    assert all([len(set(x.tolist())) for x in perturbed])
    perturbed_nodes = (perturbed != tour).sum(-1)
    assert torch.all(perturbed[:, 0] == perturbed[:, -1])
    assert torch.all((perturbed_nodes > 0) & (perturbed_nodes <= nodes_to_perturb))