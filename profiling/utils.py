from functools import partial
import torch
from ils import batch_ils
from training.utils import tour_len_ils_batch
from utils import get_tour_len



def stats_tour_len(model_output, batch):
    return get_tour_len(batch.coords, model_output.tour).cpu().numpy()


def stats_len_to_ref_len_ratio(model_output, batch):
    out = get_tour_len(batch.coords, model_output.tour) / batch.ref_len.to(batch.coords.device)
    return out.cpu().numpy()


def stats_tour_len_ils_batch(model_output, batch, n_restarts=5, n_iterations=15, k=0, max_perturbs=None):
    _, best_len = tour_len_ils_batch(model_output, batch, n_restarts, n_iterations, k, max_perturbs)
    return best_len.cpu().numpy()


def get_stats_metrics(args):
    metrics = {}
    if args.metrics is not None:
        for metric in args.metrics:
            if metric == 'len_to_ref_len_ratio':
                metrics[metric] = stats_len_to_ref_len_ratio
            elif metric == 'tour_len':
                metrics[metric] = stats_tour_len
            elif metric == 'tour_len_ils':
                f = partial(
                    stats_tour_len_ils_batch,
                    n_restarts=args.ils_n_restarts,
                    n_iterations=args.ils_n_iterations,
                    k=args.ils_k,
                    max_perturbs=args.ils_max_perturbs
                )
                metrics[metric] = f
            else:
                # TODO: eventually add other metrics
                raise NotImplementedError()
    return metrics