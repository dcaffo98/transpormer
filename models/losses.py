import torch
from torch import nn
from torch.functional import Tensor
from utils import get_tour_coords, get_tour_len



class TSPLossReinforce(nn.Module):
    
    def forward(
        self,
        coords: Tensor,
        sum_log_probs: Tensor,
        tour: Tensor,
        ref_len: Tensor,
        ref_tour: Tensor = None,
        attn_matrix: Tensor = None,
    ) -> Tensor:
        tour_len = get_tour_len(get_tour_coords(coords, tour))
        return torch.mean((tour_len - ref_len) * sum_log_probs)



class TSPEntropyLoss(nn.Module):

    def __init__(
            self,
            max_len: int = 100
    ) -> None:
        super().__init__()
        h_weights = torch.arange(1, max_len + 1, dtype=torch.float32).unsqueeze(0)
        self.register_buffer('h_weights', h_weights)

    
    def forward(
        self,
        attn_matrix: Tensor = None,
    ) -> Tensor:
        H = (- torch.log(attn_matrix) * attn_matrix).sum(-1)
        h_weights = self.h_weights[..., :attn_matrix.size(1)]
        h_weights = h_weights / h_weights.sum(-1, keepdim=True)
        h_weigths = h_weights.to(attn_matrix.device).expand(attn_matrix.size(0), -1)
        return torch.sum(H * h_weigths, -1).mean()
    

    @classmethod
    def from_args(cls, args):
        max_nodes = args.max_n_nodes if args.max_n_nodes is not None else args.n_nodes * 2
        return cls(max_nodes)



class CustomTSPLossReinforce(TSPLossReinforce):
    
    def __init__(
            self,
            h_loss: TSPEntropyLoss,
            alpha: float = 0.6
    ) -> None:
        super().__init__()
        assert alpha >= 0 and alpha <= 1, "Alpha must be within [0.0, 1.0]"
        self.tsp_loss = TSPLossReinforce()
        self.h_loss = h_loss
        self.tsp_w = alpha
        self.h_w = 1 - alpha

    
    def forward(
        self,
        coords: Tensor,
        sum_log_probs: Tensor,
        tour: Tensor,
        ref_len: Tensor,
        ref_tour: Tensor = None,
        attn_matrix: Tensor = None,
    ) -> Tensor:
        L_tsp = self.tsp_loss(coords, sum_log_probs, tour, ref_len, ref_tour, attn_matrix)
        L_h = self.h_loss(attn_matrix)
        return self.tsp_w * L_tsp + self.h_w * L_h
    

    @classmethod
    def from_args(cls, args):
        h_loss = TSPEntropyLoss.from_args(args)
        return cls(h_loss, args.reinforce_loss_entropy_alpha)
