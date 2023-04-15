import pathlib
import numpy as np
import torch
from datasets.dataset_factory import get_train_dataloader, get_eval_dataloader
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from models.utils import TSPModelInput
from training.utils import _is_new_best_highest, _is_new_best_lowest, logger, load_checkpoint, get_training_commons
from utils import get_tour_len


class Trainer:

    exclude_from_checkpoint = {
        'train_dataset',
        'train_dataloader',
        'eval_dataset',
        'eval_dataloader',
        'save_epochs',
        'epochs',
        'eval_set',
        'checkpoint_dir',
        'metrics',
        'loss',
        'device'
    }

    def __init__(
        self,
        args,
        model,
        train_dataset,
        optimizer,
        loss,
        epochs,
        eval_dataset=None,
        scheduler=None,
        checkpoint_dir=None,
        resume_from_checkpoint=None,
        device='cpu',
        metrics={},
        save_epochs=5,
        tb_comment='',
        **kwargs
    ):

        self.model = model
        self.train_dataloader = get_train_dataloader(train_dataset, args)
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.eval_dataloader = get_eval_dataloader(eval_dataset, args)
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.metrics = metrics
        self.save_epochs = save_epochs
        self.tb_comment = tb_comment 

        if checkpoint_dir:
            pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True) 

        self.best_is_highest = args.best_is_highest
        # self.best_loss = -torch.inf if args.best_is_highest else torch.inf
        if args.best_is_highest:
            self.best_loss = - torch.inf
            self._is_new_best = _is_new_best_highest
        else:
            self.best_loss = torch.inf
            self._is_new_best = _is_new_best_lowest
        self.best_metrics = {k: self.best_loss for k in metrics.keys()}
        self.start_epoch = 0
        self.best_epoch = -1
        
        
        if not args.metric_for_best_checkpoint:
            self.metric_for_best_checkpoint = 'loss'
        else:
            self.metric_for_best_checkpoint = args.metric_for_best_checkpoint


        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint `{resume_from_checkpoint}`...")
            attrs = {k: v for k, v in vars(self).items() if k not in self.exclude_from_checkpoint}
            if args.override_lr_scheduler:
                attrs.pop('scheduler')
            if args.override_lr or args.override_optim:
                attrs.pop('optimizer')
            checkpoint_data = load_checkpoint(resume_from_checkpoint, **attrs)
            if 'epoch' in checkpoint_data:
                self.start_epoch = checkpoint_data['epoch'] + 1
                checkpoint_data.pop('epoch', None)
                checkpoint_data.pop('epochs', None)
                checkpoint_data.pop('start_epoch', None)
            for k, v in attrs.items():
                if k in checkpoint_data:
                    setattr(self, k, checkpoint_data[k])
            logger.info("Checkpoint loaded!")

        self.model.to(device)
        if self.loss is not None:
            self.loss.to(device)


    @classmethod
    def from_args(cls, args):
        training_commons = get_training_commons(args)

        return cls(
            args,
            training_commons.model,
            training_commons.train_dataset,
            training_commons.optimizer,
            training_commons.loss,
            args.epochs,
            training_commons.eval_dataset,
            scheduler=training_commons.scheduler,
            metrics=training_commons.metrics,
            **training_commons.kwargs)
    
    
    def save_checkpoint(self, epoch, is_best=False):
        if self.checkpoint_dir:
            checkpoint = {'epoch': epoch}
            for k, v in vars(self).items():
                if k not in self.exclude_from_checkpoint:
                    try:
                        checkpoint[k] = v.state_dict()
                    except AttributeError:
                        checkpoint[k] = v
            path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}{'_best' if is_best else ''}.pt")
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint for epoch {epoch} saved.")


    def update_metrics(self, metrics_results):
        for k, v in metrics_results.items():
            if k not in self.best_metrics:
                self.best_metrics[k] = v
            else:
                if self._is_new_best(self.best_metrics[k], v):
                    # TODO: currently we assume `better` means either `less than` or `greater than` for ALL metrics.
                    logger.info(f"New best for metric {k}: {v} (previous was {self.best_metrics[k]}).")
                    self.best_metrics[k] = v


    def train_step(self, batch):
        '''Subclass or change this method with MethodType to customize behavior.'''
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        model_output = self.model(*model_input)
        loss_inputs, loss_targets = self.build_loss_forward_input(batch, model_output)
        # model output, gt
        l = self.loss(*loss_inputs, *loss_targets)
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return l


    def process_batch(self, batch):
        if batch.coords.device != self.device:
            batch.coords = batch.coords.to(self.device)
        if not torch.is_floating_point(batch.coords):
            batch.coords = batch.coords.to(torch.float32)
        return batch

    
    def build_model_input(self, batch):
        return (batch.coords, )

    
    def build_loss_inputs(self, batch, model_output):
        return (batch.coords, model_output.sum_log_probs, model_output.tour)

    
    def build_loss_targets(self, batch, model_output):
        return (batch.ref_len.to(model_output.sum_log_probs.device), batch.ref_tour, model_output.attn_matrix)


    def build_loss_forward_input(self, batch, model_output):
        inputs = self.build_loss_inputs(batch, model_output)
        targets = self.build_loss_targets(batch, model_output)
        return inputs, targets

    
    def eval_step(self, batch):
        '''Subclass or change this method with MethodType to customize behavior.'''
        batch = self.process_batch(batch)
        model_input = self.build_model_input(batch)
        model_output = self.model(*model_input)
        loss_inputs, loss_targets = self.build_loss_forward_input(batch, model_output)
        l = self.loss(*loss_inputs, *loss_targets)
        metrics_results = {}
        if self.metrics:
            for metric_name, metric_fun in self.metrics.items():
                # model output, gt
                metrics_results[metric_name] = metric_fun(model_output, batch)
        return l, metrics_results

    
    def do_eval(self):
        eval_loss, metrics_results = torch.inf, {}
        if self.eval_dataloader is not None:
            self.model.eval()
            logger.info("***** Running evaluation *****")
            eval_loss = 0
            n_samples = 0
            n_batches = 0
            metrics_results = {k: [] for k in self.metrics.keys()}
            with torch.no_grad():
                for batch in tqdm(self.eval_dataloader, desc="Evaluation...", mininterval=0.5, miniters=2):
                    step_loss, step_metrics_results = self.eval_step(batch)
                    eval_loss += step_loss.item()
                    for metric_name, metric_value in step_metrics_results.items():
                        metrics_results[metric_name].append(metric_value)
                    if isinstance(batch, (torch.Tensor, TSPModelInput)):
                        n_samples += len(batch)
                    else:
                        n_samples += len(batch[0])
                    n_batches += 1
            eval_loss /= n_batches
            logger.info("***** evaluation completed *****")
            logger.info(f"Eval loss: {eval_loss} | Processed samples: {n_samples}")
            for metric_name in metrics_results.keys():
                avg = np.mean(metrics_results[metric_name])
                metrics_results[metric_name] = avg
                logger.info(f"Eval `{metric_name}`: {metrics_results[metric_name]}")
        return eval_loss, metrics_results


    def epoch_begin_hook(self):
        self.model.train()


    def do_train(self):
        writer = SummaryWriter(comment=self.tb_comment)
        #j=0
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch_begin_hook()
            
            epoch_loss = 0
            n_samples = 0
            n_batches = 0
            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs}", mininterval=1, miniters=5)):
                step_loss = self.train_step(batch)
                epoch_loss += step_loss.item()
                if isinstance(batch, (torch.Tensor, TSPModelInput)):
                    n_samples += len(batch)
                else:
                    n_samples += len(batch[0])
                if i % 5 == 0:
                    writer.add_scalar("train/learning rate", self.optimizer.param_groups[0]['lr'], (i + 1) + epoch * len(self.train_dataloader))
                n_batches += 1
            
            if n_samples:
                epoch_loss /= n_batches
                logger.info(f"[epoch {epoch}] Train loss: {epoch_loss} | Processed samples: {n_samples}")

            writer.add_scalar("Loss/train", epoch_loss, epoch)

            eval_loss, metrics_results = self.do_eval()
            writer.add_scalar("Loss/eval", eval_loss, epoch)


            new_best, new_best_value, old_best_value = self.is_new_best(eval_loss, metrics_results)
            self.update_metrics(metrics_results)
            for k, v in metrics_results.items():
                writer.add_scalar(f"Metrics/{k}", v, epoch)
            logger.info(f"[epoch {epoch}] Eval loss: {eval_loss} | Best is {self.best_loss} (epoch {self.best_epoch})")
            if new_best:
                    logger.info(f"[epoch {epoch}] New best eval {self.metric_for_best_checkpoint}: {new_best_value} (previous was {old_best_value},  epoch {self.best_epoch})")
                    self.best_loss = eval_loss
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, True)
            

            if not new_best and epoch and epoch % self.save_epochs == 0:
                self.save_checkpoint(epoch)
        
        self.save_checkpoint(epoch)
        logger.info("Training completed!")
        writer.close()

    
    def is_new_best(self, eval_loss, metrics_results):
        if self.metric_for_best_checkpoint == 'loss':
            candidate = eval_loss
            reference = self.best_loss
        else:
            candidate = metrics_results[self.metric_for_best_checkpoint]
            reference = self.best_metrics[self.metric_for_best_checkpoint]
        new_best = self._is_new_best(reference, candidate)
        if new_best:
            return new_best, candidate, reference
        else:
            return new_best, reference, candidate



class BaselineReinforceTrainer(Trainer):

    def epoch_begin_hook(self):
        self.model.update_bsln()
        self.model.train()

    
    def build_loss_targets(self, batch, model_output):
        if self.model.training:
            ref_tour = model_output.bsln.tour
            ref_len = get_tour_len(batch.coords, ref_tour)
            return (ref_len, ref_tour, model_output.attn_matrix)
        else:
            return (batch.ref_len.to(model_output.sum_log_probs.device), batch.ref_tour, getattr(model_output, 'attn_matrix', None))



class TestReinforceTrainer(Trainer):
    ...
