from typing import Optional
from torch import nn
import torch
from torch.functional import Tensor
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from torch.nn.functional import dropout, softmax
from models.activation import sinkhorn
from models.layers import SinPositionalEncoding, get_positional_encoding
from scipy.optimize import linear_sum_assignment
from models.utils import TSPModelOutput
from math import sqrt
from torch.distributions import Categorical



def custom_multi_head_attn(
    query: Tensor,                  # (N, TL, D)
    key: Tensor,                    # (N, SL, D)
    value: Tensor,                  # (N, SL, D)
    out_proj_weight: Tensor = None,        # (HD, D)
    out_proj_bias: Tensor = None,          # (D, )
    nhead: int = 1,
    mask: Tensor = None,                   # (N, TL, SL)
    dropout_p: float = 0.0,
    training: bool = True,
    clip_logit_c: int = None):

    bsz, ref_len, embd_dim = query.shape
    _, key_value_len, _ = key.shape
    assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads"
    head_dim = embd_dim // nhead
    k, v = key, value

    q = query.transpose(0, 1).contiguous().view(ref_len, bsz * nhead, head_dim).transpose(0, 1)
    k = k.transpose(0, 1).contiguous().view(key_value_len, bsz * nhead, head_dim).transpose(0, 1)
    v = v.transpose(0, 1).contiguous().view(key_value_len, bsz * nhead, head_dim).transpose(0, 1)

    attn = torch.bmm(q, k.transpose(-2, -1))
    attn /= sqrt(head_dim)
    if clip_logit_c is not None:
        attn = clip_logit_c * torch.tanh(attn)
    if mask is not None:
        mask = torch.repeat_interleave(mask, nhead, dim=0)
        attn = attn + mask
    attn = softmax(attn, dim=-1)
    if not training:
        dropout_p = 0.0
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    out = torch.bmm(attn, v)

    out = out.transpose(0, 1).contiguous().view(ref_len, bsz, embd_dim).transpose(0, 1)

    # if nhead > 1 and out_proj_weight is not None:
    #     out = linear(out, out_proj_weight, out_proj_bias)
    return out, attn



class CustomMHA(nn.Module):
    def __init__(self, 
        embd_dim,
        nhead, 
        dropout_p: float = 0.0,
        use_q_proj: float = True, 
        use_kv_proj: float = True,
        is_self_attn: float = True,
        clip_logit_c: int = None) -> None:
        
        super().__init__()
        assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads."
        self.embd_dim = embd_dim
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.clip_logit_c = clip_logit_c

        if is_self_attn: 
            self.qkv_proj = nn.Linear(embd_dim, 3 * embd_dim)
        else:
            if use_q_proj: 
                self.q_proj = nn.Linear(embd_dim, embd_dim)
            if use_kv_proj:
                self.kv_proj = nn.Linear(embd_dim, 2 * embd_dim)

        if nhead > 1:
            self.out_proj_weight = nn.parameter.Parameter(torch.empty((embd_dim, embd_dim)))
            self.out_proj_bias = nn.parameter.Parameter(torch.zeros((embd_dim, )))
            nn.init.xavier_uniform_(self.out_proj_weight)
        else:
            self.out_proj_weight = None
            self.out_proj_bias = None


    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        attn_mask: Optional[Tensor] = None, 
        *args, 
        **kwargs):
        if hasattr(self, 'qkv_proj'):
            qkv = self.qkv_proj(query)
            query, key, value = torch.split(qkv, self.embd_dim, -1)
        else:
            if hasattr(self, 'q_proj'):
                query = self.q_proj(query)
            if kwargs.get('cached_key_value', None):
                key, value = kwargs['cached_key_value']
            elif hasattr(self, 'kv_proj'):
                kv = self.kv_proj(key)
                key, value = torch.split(kv, self.embd_dim, -1)
            
        out, attn = custom_multi_head_attn(query, key, value, self.out_proj_weight, self.out_proj_bias, self.nhead, attn_mask,
            self.dropout_p, self.training, self.clip_logit_c) 
        return out, attn, (key, value)



class CustomBatchNorm(nn.Module):

    def __init__(self, d_model, norm_eps=1e-5):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model, norm_eps)

    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x)
        return x.transpose(1, 2)

    

def get_norm_layer(norm: str, d_model: int, norm_eps: float = 1e-5):
    if norm == 'layer':
        return nn.LayerNorm(d_model, norm_eps)
    elif norm == 'batch':
        return nn.BatchNorm1d(d_model, norm_eps)
    elif norm == 'custom_batch':
        return CustomBatchNorm(d_model, norm_eps)
    else:
        raise NotImplementedError()



class TransformerFeedforwardBlock(nn.Module):

    def __init__(
        self,
        d_model, 
        dim_feedforward, 
        activation=F.relu,
        norm='layer', 
        norm_eps=1e-5, 
        dropout_p=0.1
    ) -> None:
        super().__init__()
        self.activation = activation
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm = get_norm_layer(norm, d_model, norm_eps)
        self.dropout = Dropout(dropout_p)


    def forward(self, x: Tensor, **kwargs) -> Tensor:
        out = self.dropout(self.linear2(self.activation(self.linear1(x))))
        return self.norm(x + out)



class TSPCustomEncoderBlock(nn.Module):
    def __init__(self, 
        d_model, 
        nhead, 
        dim_feedforward=1024, 
        dropout_p=0.1, 
        activation=F.relu, 
        norm='layer',
        norm_eps=1e-5,
        use_q_proj=True, 
        use_kv_proj=True,
        use_feedforward_block=True,
        use_q_residual=True,
        is_self_attn=True,
        clip_logit_c_ca: int = None) -> None:

        super().__init__()

        self.attn = CustomMHA(d_model, nhead, dropout_p, use_q_proj, use_kv_proj, is_self_attn=is_self_attn, clip_logit_c=clip_logit_c_ca)
        self.norm = get_norm_layer(norm, d_model, norm_eps)
        if use_feedforward_block:
            self.ff_block = TransformerFeedforwardBlock(
            d_model,
            dim_feedforward,
            activation,
            norm,
            norm_eps,
            dropout_p
           ) 
        self.use_feedforward_block = use_feedforward_block
        self.use_q_residual = use_q_residual    # True -> vanilla transformer; False -> ours


    def forward(self, query: Tensor, key_value: Tensor, attn_mask: Tensor = None, **kwargs):
        attn_out, attn_weight, cached_key_value = self.attn(query, key_value, key_value, need_weights=True, attn_mask=attn_mask, **kwargs)
        residual = query if self.use_q_residual else key_value
        out = self.norm(residual + attn_out)
        if self.use_feedforward_block:
            out = self.ff_block(out, **kwargs)
        return out, attn_weight, cached_key_value



class TSPCustomEncoderLayer(nn.Module):
    def __init__(self, 
        d_model, 
        nhead, 
        dim_feedforward=1024, 
        activation=F.relu, 
        norm='layer',
        norm_eps=1e-5,
        add_cross_attn=True,
        use_q_proj_ca=False,
        use_feedforward_block_sa=False,
        use_feedforward_block_ca=True,
        use_q_residual_sa=True,
        use_q_residual_ca=True,
        dropout_p_sa=0.1,
        dropout_p_ca=0.1,
        clip_logit_c_ca=None) -> None:

        super().__init__()
        self.sa_block = TSPCustomEncoderBlock(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout_p_sa, 
            activation, 
            norm,
            norm_eps,
            True,
            True,
            use_feedforward_block_sa,
            use_q_residual=use_q_residual_sa)
    
        self.add_cross_attn = add_cross_attn
        if add_cross_attn:
            self.ca_block = TSPCustomEncoderBlock(
                d_model, 
                nhead, 
                dim_feedforward, 
                dropout_p_ca, 
                activation, 
                norm,
                norm_eps,
                use_q_proj_ca,
                True,
                use_feedforward_block_ca,
                use_q_residual=use_q_residual_ca,
                is_self_attn=False,
                clip_logit_c_ca=clip_logit_c_ca)

        self.use_q_residual = use_q_residual_sa and use_q_residual_ca


    def forward(self, query: Tensor, key_value: Tensor, attn_mask: Tensor = None, **kwargs):
        if self.use_q_residual:
            # vanilla transformer: query comes from decoder; key_value from encoder
            query, attn_weight, _ = self.sa_block(query, query, attn_mask=None, **kwargs)
            attn_out = query
        else:
            # our implementation: query is the pos encoding, key_value is the output of the self-attention
            key_value, attn_weight, _ = self.sa_block(key_value, key_value, attn_mask=None, **kwargs)
            attn_out = key_value
        if self.add_cross_attn:
            attn_out, attn_weight, cached_key_value = self.ca_block(query, key_value, attn_mask, **kwargs)
        else:
            cached_key_value = None
        return attn_out, attn_weight, cached_key_value



class TSPCustomEncoder(nn.Module):
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward=1024, 
        dropout_p=0.1, 
        activation=F.relu, 
        layers=2, 
        norm='layer', 
        norm_eps=1e-5, 
        add_cross_attn=True, 
        use_q_proj_ca=False,
        use_feedforward_block_sa=True,
        use_feedforward_block_ca=True,        
    ) -> None:

        super().__init__()

        self.layers = nn.ModuleList([
            TSPCustomEncoderLayer(
                d_model, 
                nhead, 
                dim_feedforward, 
                activation, 
                norm,
                norm_eps, 
                add_cross_attn,
                use_q_proj_ca,
                use_feedforward_block_sa=use_feedforward_block_sa,
                use_feedforward_block_ca=use_feedforward_block_ca,
                use_q_residual_sa=False,
                use_q_residual_ca=False,
                dropout_p_sa=dropout_p,
                dropout_p_ca=dropout_p,
                ) for _ in range(layers)])
    

    def forward(self, query: Tensor, key_value: Tensor, attn_mask: Tensor = None):
        output, attn_weight = key_value, None

        for mod in self.layers:
            output, attn_weight, _ = mod(query, output, attn_mask)
        
        return output, attn_weight



class TSPCustomTransformer(nn.Module):

    def _handle_sin_pe(self):
        if type(self.pe) is SinPositionalEncoding:
            # ugly workaround
            from functools import partial
            fwd = partial(self.pe.forward, add_to_input=False)
            self.pe.forward = fwd


    def __init__(self,
        in_features=2,
        d_model=128, 
        nhead=4,
        dim_feedforward=512,
        dropout_p=0.1,
        activation=F.relu,
        norm='layer',
        norm_eps=1e-5,
        num_hidden_encoder_layers=2,
        sinkhorn_tau=5e-2,
        sinkhorn_i=20,
        add_cross_attn=True,
        use_q_proj_ca=False,
        positional_encoding='sin',
        use_feedforward_block_sa=False,
        use_feedforward_block_ca=True,
        clip_logit_c_ca=None,
        **kwargs) -> None:

        super().__init__()
        assert use_feedforward_block_sa or use_feedforward_block_ca, "You're running a model without non-linearities."
        self.pe = get_positional_encoding(positional_encoding, d_model)
        self.input_ff = nn.Linear(in_features=in_features, out_features=d_model)
        self.input_norm = get_norm_layer(norm, d_model, norm_eps)

        self.encoder = TSPCustomEncoder(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout_p, 
            activation, 
            num_hidden_encoder_layers, 
            norm,
            norm_eps, 
            add_cross_attn,
            use_q_proj_ca,
            use_feedforward_block_sa=use_feedforward_block_sa,
            use_feedforward_block_ca=use_feedforward_block_ca)
        
        self.head = TSPCustomEncoderLayer(
            d_model,
            1,
            dim_feedforward,
            activation,
            norm,
            norm_eps,
            True,
            use_q_proj_ca,
            use_feedforward_block_sa=use_feedforward_block_sa,
            use_feedforward_block_ca=False,
            use_q_residual_sa=False,
            use_q_residual_ca=False,
            dropout_p_sa=dropout_p,
            dropout_p_ca=0.,
            clip_logit_c_ca=clip_logit_c_ca)

        self.d_model = d_model
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model / nhead
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_i = sinkhorn_i
        
        self._handle_sin_pe()


    @classmethod
    def from_args(cls, args):
        activation = getattr(F, args.activation, F.relu)
        return cls(
            in_features=args.in_features,
            d_model=args.d_model, 
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout_p=args.dropout_p,
            activation=activation,
            norm=args.norm,
            norm_eps=args.norm_eps,
            num_hidden_encoder_layers=args.num_hidden_encoder_layers,
            sinkhorn_tau=args.sinkhorn_tau,
            sinkhorn_i=args.sinkhorn_i,
            add_cross_attn=args.add_cross_attn,
            use_q_proj_ca=args.use_q_proj_ca,
            positional_encoding=args.positional_encoding,
            use_feedforward_block_sa=args.use_feedforward_block_sa,
            use_feedforward_block_ca=args.use_feedforward_block_ca,
            clip_logit_c_ca=args.clip_logit_c)


    def encode(self, key_value, attn_mask=None):
        key_value = self.input_ff(key_value)
        query = self.pe(key_value)
        query = query.expand(len(key_value), *query.shape[1:])
        memory, attn_weight = self.encoder(query, key_value, attn_mask)
        memory, attn_weight, _ = self.head(query, memory, attn_mask)
        return memory, attn_weight


    def forward(self, x, attn_mask=None):
        bsz, nodes = x.shape[:2]

        _, attn_matrix = self.encode(x, attn_mask)

        attn_matrix = sinkhorn(attn_matrix, self.sinkhorn_tau, self.sinkhorn_i)
        tour = torch.empty((bsz, nodes), requires_grad=False)

        # build tour using hard permutation matrix with hungarian algorithm
        for i in range(tour.shape[0]):
            tour[i] = torch.tensor(linear_sum_assignment(attn_matrix[i].detach().cpu().numpy(), maximize=True)[1])
        
        tour = torch.cat((tour, tour[:, 0:1]), dim=1).to(attn_matrix.device).to(torch.long)
        sum_probs = torch.gather(attn_matrix, -1, tour[..., :-1].view(bsz, nodes, 1)).log().sum((1, 2))
        
        return TSPModelOutput(
            tour=tour.cpu(),
            sum_log_probs=sum_probs,
            attn_matrix=attn_matrix)
        


class TSPDecoderLayer(nn.Module):
    """
    Implementation inspired by Bresson et al. (https://github.com/xbresson/TSP_Transformer).
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout_p,
        activation,
        norm,
        norm_eps,
    ) -> None:
        super().__init__()

        self.partial_tour_ca_block = TSPCustomEncoderBlock(
            d_model,
            nhead,
            dim_feedforward,
            dropout_p,
            activation,
            norm,
            norm_eps,
            use_q_proj=True,
            use_kv_proj=False,
            use_feedforward_block=False,
            use_q_residual=True,
            is_self_attn=False
        )

        self.encoder_ca_block = TSPCustomEncoderBlock(
            d_model,
            nhead,
            dim_feedforward,
            dropout_p,
            activation,
            norm,
            norm_eps,
            use_q_proj=True,
            use_kv_proj=False,
            use_feedforward_block=True,
            use_q_residual=True,
            is_self_attn=False
        )

        self.pt_key_value_cache = None
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.d_model = d_model

    
    def reset_partial_tour_cache(self):
        self.pt_key_value_cache = None


    def forward(self, query, key_value_partial_tour, key_value, mask=None, **kwargs):
        key_value_partial_tour = list(torch.split(self.kv_proj(key_value_partial_tour[:, -1:]), self.d_model, -1))
        if self.pt_key_value_cache is None:
            self.pt_key_value_cache = key_value_partial_tour
        else:
            self.pt_key_value_cache[0] = torch.cat([self.pt_key_value_cache[0], key_value_partial_tour[0]], dim=1)
            self.pt_key_value_cache[1] = torch.cat([self.pt_key_value_cache[1], key_value_partial_tour[1]], dim=1)
        out, _, _ = self.partial_tour_ca_block(query, None, cached_key_value=self.pt_key_value_cache)
        out, attn_weight, cached_key_value = self.encoder_ca_block(out, key_value, mask, **kwargs)
        return out, attn_weight, cached_key_value



class TSPTransformer(nn.Module):

    @classmethod
    def from_args(cls, args):
        activation = getattr(F, args.activation)
        return cls(
            in_features=args.in_features,
            d_model=args.d_model, 
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout_p=args.dropout_p,
            activation=activation,
            encoder_norm=args.norm,
            norm_eps=args.norm_eps,
            num_encoder_layers=args.num_encoder_layers,
            num_hidden_decoder_layers=args.num_hidden_decoder_layers,
            positional_encoding='sin',
            clip_logit_c=args.clip_logit_c)


    def __init__(self,
        in_features=2,
        d_model=128, 
        nhead=4,
        dim_feedforward=512,
        dropout_p=0.1,
        activation=F.relu,
        encoder_norm='custom_batch',
        norm_eps=1e-5,
        num_encoder_layers=3,
        num_hidden_decoder_layers=1,
        positional_encoding='sin',
        clip_logit_c=None,
        **kwargs) -> None:

        super().__init__()
        self.pe = get_positional_encoding(positional_encoding, d_model)
        self.input_ff = nn.Linear(in_features=in_features, out_features=d_model)
        self.encoder = nn.ModuleList([
            TSPCustomEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation,
                encoder_norm,
                norm_eps,
                add_cross_attn=False,
                use_feedforward_block_sa=True,
                dropout_p_sa=dropout_p
            )
            for _ in range(num_encoder_layers)
        ])

        self.decoder = nn.ModuleList([
            TSPDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout_p,
                activation,
                'layer',
                norm_eps
            )
            for _ in range(num_hidden_decoder_layers)
        ])   

        self.head = TSPCustomEncoderBlock(
            d_model,
            1,
            dim_feedforward,
            0.,
            activation,
            use_feedforward_block=False,
            is_self_attn=False,
            clip_logit_c_ca=clip_logit_c
        )

        self.start_node = nn.Parameter(torch.randn(d_model))
        self.register_buffer('PE', self.pe(torch.zeros(1, 5000, d_model)))    

    
    def encode(self, x):
        for layer in self.encoder:
            x, _, _ = layer(x, x)
        return x
    

    def reset_partial_tour_cache(self):
        for layer in self.decoder:
            layer.reset_partial_tour_cache()


    def decode(self, query, key_value, mask=None, key_value_cache=None):
        key_value_cache = key_value_cache if key_value_cache else []
        use_cache = any(key_value_cache)
        query, partial_tour_key_value = query[:, -1:], query
        for i, layer in enumerate(self.decoder):
            query, _, cached_key_value = layer(query, partial_tour_key_value, key_value, mask, cached_key_value=key_value_cache[i] if use_cache else None)
            if len(key_value_cache) < len(self.decoder):
                key_value_cache.append(cached_key_value)
        last_cached_key_value = key_value_cache[-1] if use_cache else None
        attn_out, attn_weight, cached_key_value = self.head(query, key_value, mask, cached_key_value=last_cached_key_value)
        if len(key_value_cache) < len(self.decoder) + 1:
            key_value_cache.append(cached_key_value)
        return attn_out, attn_weight, key_value_cache


    def forward(self, x):
        bsz, n_nodes, _ = x.shape
        z = self.start_node.expand(bsz, 1, -1)
        x = self.input_ff(x)
        x = torch.concat([z, x], dim=1)
        key_value = self.encode(x)
        query = key_value[:, 0:1] + self.PE[:, 0]
        zero2bsz = torch.arange(bsz)
        tour = torch.empty((bsz, n_nodes + 1), dtype=torch.long)
        log_probs = torch.empty((bsz, n_nodes), device=x.device)
        visited_node_mask = torch.zeros((bsz, 1, n_nodes + 1), dtype=x.dtype, device=x.device)
        visited_node_mask[..., 0] = -torch.inf
        key_value_cache = None
        self.reset_partial_tour_cache()
        for t in range(n_nodes - 1):
            if t > 0:
                query = torch.concat([query, key_value[zero2bsz, idxs].view(bsz, 1, -1) + self.PE[:, t]], dim=1)
            # query_pe = query + self.PE[:, t]
            _, attn_weight, key_value_cache = self.decode(query, key_value, visited_node_mask, key_value_cache=key_value_cache)
            if self.training:
                idxs = Categorical(probs=attn_weight).sample()
            else:
                idxs = torch.argmax(attn_weight, dim=-1)
            idxs = idxs.view(-1)
            tour[:, t] = idxs - 1
            log_probs[:, t] = attn_weight[zero2bsz, :, idxs].view(-1)
            visited_node_mask[zero2bsz, :, idxs] = -torch.inf
            if t == n_nodes - 2:
                last_idxs = torch.nonzero(visited_node_mask == 0)[:, -1]
                try:
                    tour[:, t + 1] = last_idxs - 1
                except RuntimeError as e:
                    print(e)
                tour[:, -1] = tour[:, 0]
                log_probs[:, t + 1] = attn_weight[zero2bsz, :, last_idxs].view(-1)
        return TSPModelOutput(
            tour=tour,
            sum_log_probs=log_probs.log().sum(dim=-1)
        )
