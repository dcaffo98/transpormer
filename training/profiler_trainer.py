from collections import namedtuple
import sys
from tqdm import tqdm
from datasets.dataset_factory import get_eval_dataset
from profiling.utils import get_stats_metrics
from training.trainer import Trainer
import torch 
import numpy as np
import cProfile
import pstats
from uuid import uuid4
from training.utils import get_model, load_checkpoint, logger
import csv
from utils import TSPModelInput, np2nx
import os



class StatsTrainer(Trainer):

    def __init__(
            self, 
            args,
            model,
            dataset,
            filename,
            resume_from_checkpoint=None,
            device='cpu',
            metrics={},
            **kwargs
    ):
        super().__init__(
            args,
            model,
            None,
            None,
            None,
            None,
            dataset,
            None,
            None,
            resume_from_checkpoint,
            device,
            {},
            None,
            None,
            **kwargs
        )

        filename = filename if filename.endswith('.csv') else filename + '.csv'
        if not filename.startswith('/'):
            os.system("mkdir -p profiling_results")
            filename = 'profiling_results/' + filename
        self.filename = filename
        self.metrics = metrics
        self.skip_metrics = {'total_time', 'ref_len', 'id'}
        for name in self.skip_metrics:
            self.metrics[name] = []
        
        self.row = namedtuple('row', self.metrics.keys())
        self.f = open(self.filename, "w")
        self.csv_writer = csv.DictWriter(self.f, fieldnames=self.row._fields)
        self.csv_writer.writeheader()

    
    def __del__(self):
        self.f.close()
        

    def eval_step(self, batch):
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        profiler = cProfile.Profile()
        profiler.enable()
        model_output = self.model(*model_input)
        profiler.disable()
        stats = pstats.Stats(profiler, stream=sys.stdout).sort_stats('cumtime')
        metrics_results = {}
        for metric_name, metric_fun in self.metrics.items():
            if metric_name not in self.skip_metrics:
                metrics_results[metric_name] = metric_fun(model_output, batch)
        metrics_results['total_time'] = stats.total_tt
        metrics_results['ref_len'] = batch.ref_len.cpu().numpy()
        metrics_results['id'] = batch.id
        return metrics_results
    

    def do_eval(self):
        metrics_results = {}
        self.model.eval()
        logger.info("***** Running evaluation *****")
        n_samples = 0
        metrics_results = {k: [] for k in self.metrics.keys()}
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluation...", mininterval=0.5, miniters=2):
                step_metrics_results = self.eval_step(batch)
                flattened_metrics = {k: [] for k in step_metrics_results.keys()}
                for metric_name, metric_value in step_metrics_results.items():
                    flattened_metrics[metric_name].extend(np.array(metric_value).reshape(-1).tolist())
                if len(batch) > 1:
                    flattened_metrics['total_time'].extend([np.nan for _ in range(1, len(batch))])
                for i in range(len(batch)):
                    row = self.row(**{k: v[i] for k, v in flattened_metrics.items()})
                    self.csv_writer.writerow(row._asdict())
        
                if isinstance(batch, (torch.Tensor, TSPModelInput)):
                    n_samples += len(batch)
                else:
                    n_samples += len(batch[0])
        logger.info("***** evaluation completed *****")
        logger.info(f"Processed samples: {n_samples}")
        return metrics_results

    
    @classmethod
    def from_args(cls, args):
        model = get_model(args)
        if args.resume_from_checkpoint:
            checkpoint = load_checkpoint(args.resume_from_checkpoint, verbose=False)
            model.load_state_dict(checkpoint['model'])
        eval_dataset = get_eval_dataset(args) 
        metrics = get_stats_metrics(args)
        if args.filename:
            filename = args.filename
        else:
            filename = args.model + '_' + str(uuid4())
        return cls(
            args=args, 
            model=model, 
            dataset=eval_dataset,
            filename=filename, 
            resume_from_checkpoint=None, 
            device=args.device, 
            metrics=metrics, 
        )



class NetworkxStatsTrainer(StatsTrainer):

    def build_model_input(self, batch):
        if len(batch) == 1:
            out = [np2nx(batch.coords.squeeze())]
        else:
            out = [np2nx(x.coords.squeeze()) for x in batch.coords]
        return (out,)