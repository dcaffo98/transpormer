import torch
from torch.functional import Tensor
from torch.nn.functional import _in_projection_packed, dropout, linear, softmax
import math


def multi_head_attn(
    query: Tensor,                  # (N, TL, D)
    key: Tensor,                    # (N, SL, D)
    value: Tensor,                  # (N, SL, D)
    in_proj_weight: Tensor,         # (3 * D, D)
    in_proj_bias: Tensor,           # (3 * D, ),
    out_proj_weight: Tensor,        # (HD, D)
    out_proj_bias: Tensor = None,          # (D, )
    nhead: int = 1,
    mask: Tensor = None,                   # (N, TL, SL)
    dropout_p: float = 0.0,
    training: bool = True):

    bsz, ref_len, embd_dim = query.shape
    _, src_len, _ = key.shape
    assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads"
    head_dim = embd_dim // nhead
    q, k, v  = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    q = q.contiguous().view(ref_len, bsz * nhead, head_dim).transpose(0, 1)
    k = k.contiguous().view(src_len, bsz * nhead, head_dim).transpose(0, 1)
    v = v.contiguous().view(src_len, bsz * nhead, head_dim).transpose(0, 1)

    q = q / math.sqrt(head_dim)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if mask is not None:
        if nhead > 1:
            mask = torch.repeat_interleave(mask, repeats=nhead, dim=0).unsqueeze(1)
        attn = attn + mask
    attn = softmax(attn, dim=-1)
    if not training:
        dropout_p = 0.0
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    out = torch.bmm(attn, v)
    out = out.transpose(0, 1).contiguous().view(bsz, ref_len, embd_dim)
    out = linear(out, out_proj_weight, out_proj_bias)
    return out, attn


def sinkhorn(x: Tensor, tau: float, i: int):
    """Compute a soft permutation matrix using Sinkhorn algorithm.

    Args:
        x (Tensor): batch of squared matrices (N, L, L)
        tau (float): temperature
        i (int): number of iterations

    Returns:
        [type]: [description]
    """ 
    
    x = torch.exp(x / tau)
    for _ in range(i):
       x = x / (10e-8 + torch.sum(x, -2, keepdim=True))
       x = x / (10e-8 + torch.sum(x, -1, keepdim=True))
    return x
    