import math
from typing import Optional
import torch
from torch import nn
from torch.functional import Tensor
from models.activation import multi_head_attn


class MHA(nn.Module):
    def __init__(self, embd_dim, nhead, dropout_p: float = 0.0) -> None:
        super(MHA, self).__init__()
        assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads."
        self. embd_dim = embd_dim
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.in_proj_weight = nn.parameter.Parameter(torch.empty((3 * embd_dim, embd_dim)))
        self.in_proj_bias = nn.parameter.Parameter(torch.zeros((3 * embd_dim, )))
        self.out_proj_weight = nn.parameter.Parameter(torch.empty((embd_dim, embd_dim)))
        self.out_proj_bias = nn.parameter.Parameter(torch.zeros((embd_dim, )))

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, need_weights: bool = True, attn_mask: Optional[Tensor] = None, *args, **kwargs):
        out, attn = multi_head_attn(query, key, value, self.in_proj_weight, self.in_proj_bias,
         self.out_proj_weight, self.out_proj_bias, self.nhead, attn_mask,
            self.dropout_p, self.training) 
        if need_weights:
            return out, attn
        else:
            return out


class SinPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


    def forward(self, x: Tensor, add_to_input: bool = True) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        pe = self.pe[:, :x.size(1)]
        if add_to_input:
            return x + pe
        else:
            return pe



class CustomSinPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.sin_pe = SinPositionalEncoding(d_model, max_len)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor):
        return self.proj(self.sin_pe(x, False))


class CustomPositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.proj = nn.Linear(1, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        bsz, l = x.shape[:2]
        idx = torch.arange(l, dtype=x.dtype, device=x.device).expand(bsz, -1).unsqueeze(-1)
        return self.proj(idx / idx.max())


def get_positional_encoding(positional_encoding: str, d_model: int, max_len: int = 5000, *args, **kwargs):
    if positional_encoding == 'sin':
        return SinPositionalEncoding(d_model, max_len)
    elif positional_encoding == 'custom_sin':
        return CustomSinPositionalEncoding(d_model, max_len)
    elif positional_encoding == 'custom':
        return CustomPositionalEncoding(d_model)


if __name__ == '__main__':
    bsz, src_len, embd_dim, nhead = 4, 100, 128, 4
    mha = MHA(embd_dim, nhead, dropout_p=0.1)
    src = torch.rand(bsz, src_len, embd_dim)
    memory = torch.rand(bsz, 10, embd_dim)
    out, attn_w = mha(src, memory, memory)
    assert out.shape == src.shape