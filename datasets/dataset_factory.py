import torch
from datasets.datasets import RandomTrainTSPDataset, TSPDataset
from datasets.utils import _nop_collate_fn, custom_collate_fn, _identity, sample_max_n_nodes
from functools import partial



def get_train_dataset(args):
    if args.max_n_nodes is not None:
        sample_n_nodes = partial(sample_max_n_nodes, args.n_nodes, args.max_n_nodes)
    else:
        sample_n_nodes = partial(_identity, args.n_nodes)
    return RandomTrainTSPDataset(
        n_steps=args.train_steps_per_epoch,
        bsz=args.train_batch_size,
        n_features=args.in_features,
        sample_n_nodes=sample_n_nodes,
        # TODO: fix creating data straight on GPU for multiple dataloader
        # device=args.device
        device='cpu')



def get_train_dataloader(dataset: RandomTrainTSPDataset, args):
    return torch.utils.data.DataLoader(
            dataset, 
            shuffle=False,
            batch_size=1,
            num_workers=args.dataloader_num_workers,
            collate_fn=_nop_collate_fn)



def get_eval_dataset(args):
    return TSPDataset(args.eval_dataset)



def get_eval_dataloader(dataset: TSPDataset, args):
    return torch.utils.data.DataLoader(
        dataset, 
        shuffle=args.eval_shuffle_data,
        batch_size=args.eval_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=custom_collate_fn)


