from typing import OrderedDict, Sequence
from torch import Tensor
import torch
import torch.nn as nn
from models.utils import TSPModelOutput
from dataclasses import dataclass
import copy
import networkx as nx
from utils import np2nx



@dataclass
class TourModelWithBaselineOutput(TSPModelOutput):
    bsln: TSPModelOutput = None



class RLAgentWithBaseline(nn.Module):

    def __init__(
        self,
        model: nn.Module
    ) -> None:
        super().__init__()
        self.model = model
        self.bsln = copy.deepcopy(model)
        self.bsln.eval()

    
    def train(self, mode: bool = True):
        self.model.train(mode)
        self.training = mode


    def eval(self):
        self.model.eval()
        self.training = False

    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)

    
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                    strict: bool = True):
        self.model.load_state_dict(state_dict, strict)

    
    def update_bsln(self, state_dict: 'OrderedDict[str, Tensor]' = None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        self.bsln.load_state_dict(state_dict)
    
    
    def forward(self, x, *args, **kwargs):
        model_out = self.model(x, *args, **kwargs)
        # if self.model.training:
        with torch.no_grad():
            bsln_out = self.bsln(x, *args, **kwargs)
            return TourModelWithBaselineOutput(**vars(model_out), bsln=bsln_out)
        # return TourModelWithBaselineOutput(**vars(model_out))



class NetworkxWrapper(nn.Module):

    def forward(self, x: Sequence[nx.Graph]):
        tsp = nx.approximation.traveling_salesman_problem
        if len(x) > 1:
            graphs = [np2nx(sample) for sample in x]
            tour = [tsp(g, cycle=True) for g in graphs]
        else:
            tour = [tsp(x[0], cycle=True)]
        return TSPModelOutput(torch.tensor(tour) - 1, None, None)