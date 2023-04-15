from datasets.dataset_factory import get_eval_dataset, get_train_dataset
from ils import batch_ils
from models.custom_transformer import TSPCustomTransformer, TSPTransformer
from models.losses import CustomTSPLossReinforce, TSPLossReinforce
from models.wrapped_models import NetworkxWrapper, RLAgentWithBaseline
from torch.optim import Adam, SGD
import torch.nn  as nn
from torch.optim.lr_scheduler import LambdaLR
from models.wrapped_models import RLAgentWithBaseline
import torch
from utils import get_tour_len, logger
import numpy as np
from functools import partial



def _is_new_best_highest(old, new):
    return new > old
def _is_new_best_lowest(old, new):
    return new < old

def len_to_ref_len_ratio(model_output, batch):
    tours = model_output.tour
    tour_coords = batch.coords[torch.arange(len(tours)).view(-1, 1), tours]
    tour_len = get_tour_len(tour_coords)
    return (tour_len.cpu() / batch.ref_len).mean().item()




def avg_tour_len(model_output, batch):          
    tours = model_output.tour
    tour_coords = batch.coords[torch.arange(len(tours)).view(-1, 1), tours]
    tour_len = get_tour_len(tour_coords)
    return tour_len.mean().item()



def tour_len_ils_batch(model_output, batch, n_restarts=5, n_iterations=15, k=0, max_perturbs=None):
    if k > 0:
        A = model_output.attn_matrix
        H = (- torch.log(A) * A).sum(-1)
        whitelist_idxs = H[:, 1:].topk(k, dim=-1).indices + 1
    else:
        whitelist_idxs = None
    best_tour, best_len = batch_ils(
        get_tour_len,
        batch.coords,
        model_output.tour,
        n_restarts,
        n_iterations,
        whitelist_idxs,
        max_perturbs
    )
    return best_tour, best_len



def avg_tour_len_ils_batch(model_output, batch, n_restarts=5, n_iterations=15, k=0, max_perturbs=None):
    best_tour, best_len = tour_len_ils_batch(model_output, batch, n_restarts, n_iterations, k, max_perturbs)
    return best_len.mean().item()



def load_checkpoint(path, verbose=True, **kwargs):
    checkpoint = torch.load(path, map_location='cpu')
    out = {}
    for key, obj in checkpoint.items():
        if key in kwargs:
            if hasattr(kwargs[key], 'load_state_dict'):
                kwargs[key].load_state_dict(obj)
            else:
                out[key] = obj
        else:
            out[key] = obj
            if verbose:
                logger.warning(f"`{key}` not found in checkpoint `{path}`.")
    return out



def get_model(args):
    if args.model == 'custom':
        model = TSPCustomTransformer.from_args(args)
    elif args.model == 'baseline':
        model = TSPTransformer.from_args(args)
    elif args.model == 'networkx':
        model = NetworkxWrapper()
    else:
        raise NotImplementedError()
    try:
        if args.do_train and args.train_mode == 'reinforce' and args.reinforce_baseline == 'baseline':
            model = RLAgentWithBaseline(model)
    except AttributeError:
        pass
    return model.to(args.device)



def get_optimizer(args, model):
    # manage wrapped models
    if hasattr(model, 'model'):
        params = model.model.parameters
    else:
        params = model.parameters
    if args.optimizer == 'adam':
        return Adam(params(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        return SGD(params(), lr=args.learning_rate)
    else:
        raise NotImplementedError()


def get_loss(args):
    if args.loss == 'reinforce_loss':
        loss = TSPLossReinforce()
    elif args.loss == 'reinforce_loss_entropy':
        loss = CustomTSPLossReinforce.from_args(args)
    else:
        raise NotImplementedError()
    return loss.to(args.device)



def get_transformer_lr_scheduler(optim, d_model, warmup_steps):
    for group in optim.param_groups:
        group['lr'] = 1
    def lambda_lr(s):
        d_model_ = d_model
        warm_up = warmup_steps
        s += 1
        return (d_model_ ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    return LambdaLR(optim, lambda_lr)



def get_lr_scheduler(args, optim):
    if args.lr_scheduler is None:
        return
    elif args.lr_scheduler == 'transformer':
        return get_transformer_lr_scheduler(optim, args.d_model, args.warmup_steps)
    elif args.lr_scheduler == 'linear':
        def lambda_lr(s, max_steps=None, lr_start=None, lr_delta=None, min_lr=None):
            tgt_lr = max(min_lr, lr_start - min(s, max_steps) * lr_delta)
            return tgt_lr / lr_start
        lr_delta = (args.learning_rate - args.lr_linear_scheduler_min_lr) / args.lr_scheduler_max_steps
        _lr_scheduler = partial(lambda_lr, max_steps=args.lr_scheduler_max_steps, lr_start=args.learning_rate, lr_delta=lr_delta, min_lr=args.lr_min)
        return LambdaLR(optim, _lr_scheduler)
    else:
        raise NotImplementedError()



def get_metrics(args):
    metrics = {}
    if args.metrics is not None:
        for metric in args.metrics:
            if metric == 'len_to_ref_len_ratio':
                metrics[metric] = len_to_ref_len_ratio
            elif metric == 'avg_tour_len':
                metrics[metric] = avg_tour_len
            elif metric == 'avg_tour_len_ils':
                f = partial(
                    avg_tour_len_ils_batch,
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



def get_training_commons(args):
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    model = get_model(args)
    optimizer = get_optimizer(args, model)

    training_commons = dict(
        model=model,
        train_dataset=get_train_dataset(args),
        eval_dataset=get_eval_dataset(args),
        optimizer=optimizer,
        loss=get_loss(args),
        scheduler=get_lr_scheduler(args, optimizer),
        metrics=get_metrics(args),
        kwargs={k: v for k, v in vars(args).items() if k not in {'model', 'train_dataset', 'eval_dataset', 'optimizer', 'loss', 'epochs', 'metrics'}}
    )

    return dotdict(training_commons)