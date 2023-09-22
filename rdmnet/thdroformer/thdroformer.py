"""

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, PEConditionalTransformer
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer
from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput
from einops import rearrange

def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))

def dynamic_attention(query, key, value, k):
    batch, head, n, dim = query.shape
    device = query.device
    scores = torch.einsum('bhnd,bhmd->bhnm', query, key) / dim**.5
    if k == None:
        scores = torch.nn.functional.softmax(scores, dim=-1)
    else:
        k = int(n*k)
        values, indices = scores.topk(k, dim=3, largest=True, sorted=False)
        # idx = scores<values[:,:,:,k-1][:,:,:,None]
        prob =  torch.nn.functional.softmax(values, dim=-1) # perform softmax on the top k nodes
        scores = torch.zeros_like(scores)

        B = torch.arange(0, batch, device=device).view(-1, 1, 1, 1).repeat(1, head, n,k)
        H = torch.arange(0, head, device=device).view(1, -1, 1, 1).repeat(batch, 1, n,k)
        N = torch.arange(0,  n, device=device).view(1, 1, -1, 1).repeat(batch, head, 1,k)

        scores[B,H,N,indices] = prob


    return torch.einsum('bhnm,bhmd->bhnd', scores, value), scores

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, num_heads):
        super(RotaryPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        d_model = int(d_model/num_heads)
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model)).view(1, 1, -1)
        div_term = F.interpolate(div_term, scale_factor=2, mode='nearest').view(1, 1, 1, -1)
        self.register_buffer('div_term', div_term)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, desc0, pos_emb):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        batch_dim, num_heads, num_points, feature_dim = desc0.size()
        _, _, _, pos_dim = pos_emb.size()
        device=torch.device('cuda')

        # position = pos[:, :3, :]

        desc = desc0.view(batch_dim,num_heads,num_points,np.int(feature_dim/2),2)
        desc = torch.cat((-desc[:,:,:,:,1].unsqueeze(-1), desc[:,:,:,:,0].unsqueeze(-1)), 4)
        desc = desc.view(batch_dim,num_heads,num_points,feature_dim)


        pos_emb = F.interpolate(pos_emb.squeeze(0), scale_factor=2, mode='nearest').unsqueeze(0)

        theta = self.sigmoid(pos_emb) * 3.14159265359 * 2
        # theta = self.relu(pos_emb)


        # theta = torch.mul(pos_emb, self.div_term.repeat(batch_dim, num_heads, num_points, 1))
        # theta = pos_emb

        return torch.mul(desc0,torch.cos(theta))+torch.mul(desc,torch.sin(theta))


class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, k=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.pos_encoder = RotaryPositionalEmbedding(self.d_model, num_heads)

        self.dropout = build_dropout_layer(dropout)

        self.k = k

    def forward(self, input_q, input_k, input_v, embed_qk, layer, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        embed_qk = rearrange(embed_qk, 'b m (h c) -> b h m c', h=self.num_heads)
        q = self.pos_encoder(q, embed_qk)
        k = self.pos_encoder(k, embed_qk)

        if self.k is not None:
            hidden_states, attention_scores = dynamic_attention(q,k,v,self.k[layer])
        else:
            hidden_states, attention_scores = dynamic_attention(q,k,v,None)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, k=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout, k=k)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        layer,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            layer,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU',k=None):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout, k=k)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        layer,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            layer,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores

class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
        k=None
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn, k=k))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        self_layer_idx=0
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, self_layer_idx, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, self_layer_idx, memory_masks=masks1)
                self_layer_idx+=1
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1

class posEmbedding(nn.Module):
    def __init__(self, hidden_dim, reduction_a='max'):
        super(posEmbedding, self).__init__()

        self.proj = nn.Linear(3, int(hidden_dim/2))

    def forward(self, points):

        embeddings = self.proj(points)

        return embeddings


class ThDRoFormer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        num_layers,
        k=None,

        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(ThDRoFormer, self).__init__()

        self.embedding = posEmbedding(hidden_dim, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            ['self', 'cross']*num_layers, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, k=k
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)
        # ref_embeddings=None
        # src_embeddings=None

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats

#sch, ablation
class posEmbedding2(nn.Module):
    def __init__(self, hidden_dim, reduction_a='max'):
        super(posEmbedding2, self).__init__()

        # self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj = nn.Linear(3, int(hidden_dim))

        # self.reduction_a = reduction_a
        # if self.reduction_a not in ['max', 'mean']:
        #     raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    def forward(self, points):

        embeddings = self.proj(points)


        return embeddings

class APETransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        blocks,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(APETransformer, self).__init__()

        self.embedding = posEmbedding2(hidden_dim, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.transformer = PEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)
        # ref_embeddings=None
        # src_embeddings=None

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats