# MIT License

# Copyright (c) [2023] [Anima-Lab]

# This code is adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py
# and https://github.com/facebookresearch/DiT/blob/main/models.py.
# The original code is licensed under a Attribution-NonCommercial 4.0 InternationalCreative Commons License,
# which is can be found at licenses/LICENSE_MAE.txt and licenses/LICENSE_DIT.txt.


import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import einops
from models.Graphormer import GraphEncoder,GraphTransformer
from  train_utils.helper import dropout_nodes_in_graph
import torch.nn.functional as F
import random
from torch import nn, einsum

from torch.nn.utils.rnn import pad_sequence
from inspect import isfunction
from typing import Any, Optional, Tuple, Type
from models.prompt_encoder import PromptEncoder

from models.swiglu_ffn import SwiGLUFFN

from models.rmsnorm import RMSNorm
from torch.utils.checkpoint import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def modulate(x, shift, scale):
    if len(scale.shape) == 2:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob = 0.1):
        super().__init__()
        self.embedding_table = nn.Linear(num_classes, hidden_size, bias=False)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, y):
        embeddings = self.embedding_table(y)
        return embeddings


#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################

def get_mask(batch, length, mask_ratio, device):
    """
    Get the binary mask for the input sequence.
    Args:
        - batch: batch size
        - length: sequence length
        - mask_ratio: ratio of tokens to mask
    return:
        mask_dict with following keys:
        - mask: binary mask, 0 is keep, 1 is remove
        - ids_keep: indices of tokens to keep
        - ids_restore: indices to restore the original order
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {'mask': mask,
            'ids_keep': ids_keep,
            'ids_restore': ids_restore}


def mask_out_token(x, ids_keep):
    """
    Mask out the tokens specified by ids_keep.
    Args:
        - x: input sequence, [N, L, D]
        - ids_keep: indices of tokens to keep
    return:
        - x_masked: masked sequence
    """
    N, L, D = x.shape  # batch, length, dim
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked


def mask_tokens(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def unmask_tokens(x, ids_restore, mask_token, extras=0):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+1, D]
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + extras - x.shape[1], 1)
    x_ = torch.cat([x[:, extras:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :extras, :], x_], dim=1)  # append cls token
    return x


def seg_drop(seg, train, force_drop_ids=None, drop_ratio=0.2):
    """
    Drops labels to enable classifier-free guidance.
    """
    # bs, c, h, w = seg.shape
    if (force_drop_ids is None) and (train is True):

        # mask = (torch.rand([seg.shape[0], 1, 1]) > self.dropout_prob).float().to(seg.device)
        # seg = seg * mask
        for b in range(seg.shape[0]):
            if random.random() > drop_ratio:
                seg[b] = seg[b] * 0  # set seg to background for classifier-free
    return seg


from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class GraphFusion(nn.Module):
    def __init__(self, in_dim, graph_dim, num_heads=4, target_nodes=96,
                 pe_layer=PositionEmbeddingRandom(num_pos_feats=768)):
        super(GraphFusion, self).__init__()

        # self.attn_x_to_node = CrossAttention(query_dim = graph_dim, key_dim = in_dim, heads=num_heads)
        self.attn_node_to_x = CrossAttention(query_dim=in_dim, key_dim=graph_dim, heads=num_heads)
        self.norm_node = nn.LayerNorm(graph_dim)
        self.target_nodes = target_nodes

    # def _contrust_graph(self, x, graph, text_context):
    #     """
    #     Constructs and encodes a graph using cross-attention and GCNConv.
    #
    #     Args:
    #         x (torch.Tensor): Visual embeddings, shape [B, L, D].
    #         mask_dict (dict): Mask information for input tokens.
    #         instance_map (torch.Tensor): Instance map, shape [B, H, W].
    #         graph (dict): Graph information with keys `l1` containing `x`, `edge_index`, and `batch`.
    #         text_context (torch.Tensor): Text embeddings, shape [num_classes, D].
    #
    #     Returns:
    #         torch.Tensor: Encoded graph node features.
    #     """
    #     # Extract graph components
    #     node_l1, edge_index, batch = graph["l1"].x, graph["l1"].edge_index, graph["l1"].batch
    #
    #     # Step 1: Cross-Attention (Q=x, K=text_context, V=x)
    #     # Use x as Q, text_context as K and V
    #     x_proj = x.view(-1, x.shape[-1])  # Flatten x to [B * L, D]
    #     text_context_proj = text_context.unsqueeze(0).repeat(x_proj.size(0), 1, 1)  # Repeat text context for all nodes
    #
    #     # Apply cross-attention
    #     fused_nodes = self.cross_attention_1(x_proj.unsqueeze(1), text_context_proj).squeeze(1)
    #
    #     # Step 2: Graph Convolution
    #     # Use fused_nodes as input to GCN
    #     updated_node_features = fused_nodes.view(node_l1.shape[0], -1)  # Reshape back to [num_nodes, graph_dim]
    #     encoded_nodes = self.graph_conv(updated_node_features, edge_index)  # Apply GCNConv
    #
    #     # Normalize graph features
    #     encoded_nodes = self.norm1(encoded_nodes)
    #
    #     return encoded_nodes

    def _embed_edges(
            self,
            points: torch.Tensor,
            labels: torch.Tensor,
            pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def aggregate_features(self, fused_nodes, instance_map, num_nodes=96):
        """
        Convert fused_nodes (B, L, D) to (B, num_nodes, D) based on instance_map (B, H, W).

        Args:
            fused_nodes (torch.Tensor): Tensor of shape [B, L, D], features to be aggregated.
            instance_map (torch.Tensor): Instance map of shape [B, H, W], mapping pixels to node IDs.
            num_nodes (int): Target number of nodes per batch.

        Returns:
            torch.Tensor: Aggregated node features of shape [B, num_nodes, D].
        """
        B, L, D = fused_nodes.shape
        _, H, W = instance_map.shape
        ori_w = ori_h = int(L ** 0.5)  # Calculate the resolution of fused_nodes.

        # Reshape fused_nodes to match the spatial resolution of instance_map
        fused_nodes_reshaped = fused_nodes.view(B, ori_h, ori_w, D)  # Shape: [B, ori_h, ori_w, D]
        fused_nodes_upsampled = F.interpolate(
            fused_nodes_reshaped.permute(0, 3, 1, 2),  # Convert to [B, D, ori_h, ori_w]
            size=(H, W),  # Target resolution matches instance_map
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # Shape: [B, H, W, D]

        # Aggregate features based on instance_map
        aggregated_features = torch.zeros(B, num_nodes, D, device=fused_nodes.device)  # Initialize [B, num_nodes, D]
        for b in range(B):
            for node_id in range(num_nodes):
                mask = (instance_map[b] == node_id)  # Mask for the current node
                if mask.sum() > 0:  # If the node has valid pixels
                    aggregated_features[b, node_id] = fused_nodes_upsampled[b][mask].mean(dim=0)

        return aggregated_features

    def _encode_graph(self, x, graph_feats, text_context):

        fused_nodes = graph_feats + self.attn_x_to_node(graph_feats, x).squeeze(1)  # B  L D
        return fused_nodes
        # than use GCNConv to encode Q , edge_index

    def forward(self, x, graph):
        """
        Args:
            x (torch.Tensor): Visual embeddings, shape [B, L, D].
            text_context (torch.Tensor): Text embeddings, shape [num_classes, 768].
            batch_graph (list[Data]): A list of PyTorch Geometric Data objects, one per batch.
            instance_map (torch.Tensor): Instance map, shape [B, H, W].

        Returns:
            torch.Tensor: Fused embeddings, shape [B, L, D].
        """

        # encoded_nodes = self.norm_node(self._encode_graph(x, graph, text_context))

        # Use encoded graph features for further processing (e.g., fusion with x)

        # assert "node_feats" in graph.keys()
        # node_feats = graph["node_feats"]
        # node_mask = graph["node_mask"] if "node_mask" in graph.keys() else None
        if len(graph.shape) == 2:
            graph = graph.unsqueeze(1)

        fused_x = self.attn_node_to_x(x, graph)

        return fused_x
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused_attn: bool = True,
            use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LightningGDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            use_qknorm=False,
            use_swiglu=False,
            use_rmsnorm=False,
            wo_shift=False,
            **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class GDiTBlock(nn.Module):
    """
        A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
        """

    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4.0,
                 drop_path=0.1,
                 with_graph=False,
                 use_qknorm=False,
                 use_swiglu=False,
                 use_rmsnorm=False,
                 **block_kwargs):
        super().__init__()
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        self.with_graph = with_graph
        self.attn_node_to_x = CrossAttention(query_dim=hidden_size, key_dim=hidden_size, heads=num_heads, use_rmsnorm=use_rmsnorm, qk_norm=use_qknorm)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        # self.attn = CrossAttention(hidden_size, hidden_size, heads=num_heads, **block_kwargs)
        #self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )




        #
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        # self.context_dense = SPADE(9, hidden_size, hidden_size, fusion_type='AdnLn')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, c, graph = None, mask = None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + c.reshape(c.shape[0], 6, -1)).chunk(6, dim=1)


        #cross_attn
        #x = self.attn_node_to_x(x, graph, mask)
        # attn
        x = x + self.drop_path(gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self.attn_node_to_x(x, graph, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DecoderLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, decoder_hidden_size):
        super().__init__()
        self.norm_decoder = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, context, mask_dict=None):
        # if context is not None:
        #     # context = einops.rearrange(context, 'b c h w -> b  (h w)  c')
        #     if mask_dict is not None:
        #         ids_keep = mask_dict['ids_keep']
        #         context = mask_out_token(context, ids_keep)
        #     c = c[:, None, :] + context

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        #
        # if len(c.shape) == 2:
        #     gate_mlp = gate_mlp.unsqueeze(dim=1)
        #     gate_msa = gate_msa.unsqueeze(dim=1)
        #
        # x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # return x
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, c_emb_size) / c_emb_size ** 0.5)

    def forward(self, x, c, context):
        if context is not None:
            # context = einops.rearrange(context, 'b c h w -> b  (h w)  c')
            c = c[:, None, :] + context
        shift, scale = (self.scale_shift_table[None] + c[:, None]).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SPADE(nn.Module):
    def __init__(self, label_nc, out_channels, dropout_prob, fusion_type):
        super().__init__()

        # self.dropout_prob = dropout_prob
        self.fusion_type = fusion_type

        # self.param_free_norm = nn.LayerNorm(norm_nc)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128

        # if fusion_type == 'concat':
        self.conv2d = nn.Conv2d(label_nc, out_channels, kernel_size=1)

    #

    def seg_drop(self, seg, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        # bs, c, h, w = seg.shape
        if force_drop_ids is None:

            # mask = (torch.rand([seg.shape[0], 1, 1]) > self.dropout_prob).float().to(seg.device)
            # seg = seg * mask
            for b in range(seg.shape[0]):
                if random.random() > 1.0:
                    seg[b] = seg[b] * 0  # set seg to background for classifier-free
        else:
            seg = seg * 0
        return seg

    def forward(self, x, segmap, train=False, force_drop_ids=None, mask_dict=None):

        # segmap = einops.rearrange(segmap, 'b c h w -> b  (h w)  c')
        if mask_dict is not None:
            ids_keep = mask_dict['ids_keep']
            segmap = mask_out_token(segmap, ids_keep)

        if len(x.shape) == 3:
            h = w = int(math.sqrt(x.shape[1]))
        else:
            _, _, h, w = x.shape
            # Part 1. generate parameter-free normalized activations
            # normalized = self.param_free_norm(x)

            # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=(h, w), mode='nearest')

        # use_dropout = self.dropout_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     segmap = self.seg_drop(segmap, force_drop_ids)
        # if self.fusion_type == 'concat':
        #     out = self.conv2d(segmap)
        # else:
        #     actv = self.mlp_shared(segmap)
        #     gamma = self.mlp_gamma(actv)
        #     beta = self.mlp_beta(actv)
        #
        #     # gamma = einops.rearrange(gamma, 'b c h w -> b (h w) c')
        #     # beta = einops.rearrange(beta, 'b c h w -> b (h w) c')
        #     # apply scale and bias
        #     normalized = self.param_free_norm(x)
        #     out = normalized * (1 + gamma) + beta
        out = self.conv2d(segmap)
        return out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim=None, heads=8, attn_drop=0., dropout=0., qk_norm = True, use_rmsnorm = False):
        super().__init__()
        inner_dim = query_dim
        transformer_width = default(inner_dim, query_dim)
        head_dim = transformer_width // heads
        self.scale = head_dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, transformer_width, bias=False)
        self.to_k = nn.Linear(key_dim, transformer_width, bias=False)
        self.to_v = nn.Linear(key_dim, transformer_width, bias=False)

        # Initialize normalization layers
        if use_rmsnorm:
            norm_layer = RMSNorm

        self.q_norm = norm_layer(transformer_width) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(transformer_width) if qk_norm else nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(transformer_width, inner_dim),
            nn.Dropout(dropout)
        )
        self.attn_drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, querys, key, mask=None):
        B, N, C = querys.shape

        h = self.heads

        q = self.to_q(querys)

        k = self.to_k(key)
        q, k = self.q_norm(q), self.k_norm(k)



        v = self.to_v(key)




        q = q * self.scale  # , q2 * self.scale

        q = einops.rearrange(q, 'b n (h1  c) -> b  h1 n c', h1=self.heads)
        k = einops.rearrange(k, 'b n  (h1   c) -> b  h1 n c', h1=self.heads)
        v = einops.rearrange(v, 'b n  (h1   c) -> b  h1 n c', h1=self.heads)

        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:

            if mask.dim() == 2:  # [B, M], need to expand it to [B, h, N, M]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
            elif mask.dim() == 3:  # [B, N, M], just expand it to [B, h, N, M]
                mask = mask.unsqueeze(1)  # [B, 1, N, M]
            # Apply the mask
            attn = attn.masked_fill(mask == 0, float('-inf'))
            # 若某行被完全mask掉，attention softmax计算会出现nan，因此需要特殊处理
            mask_sum = mask.sum(dim=-1, keepdim=True)  # [B, h, N, 1]
            attn = torch.where(mask_sum == 0, torch.zeros_like(attn), attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = attn @ v
        x = (attn @ v).transpose(1, 2)
        x = einops.rearrange(x, ' b n h1 c -> b  n (h1 c)', h1=self.heads)
        x = self.to_out(x)
        return x  # , attn


# class CrossAttention(nn.Module):
#     def __init__(self, dim,   num_heads = 8,qkv_bias=False,qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#
#
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.num_heads = num_heads
#         assert dim % num_heads == 0
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()
#         self.parm = nn.Parameter(torch.zeros(1))
#         self.norm = nn.LayerNorm(dim)
#     def forward(self, seg_f, graph_f):
#         #seg_f #shape:B N C
#         #graph_f #shape: B C
#
#
#         B, N ,C = seg_f.size()
#
#         q = self.q(graph_f)
#         k = self.k(seg_f)
#         v = self.v(seg_f)
#
#
#
#         q= q * self.scale#, q2 * self.scale
#         q = einops.rearrange(q, 'b (h1  c) -> b  h1 c',  h1 = self.num_heads)
#         k = einops.rearrange(k, 'b n  (h1  c) -> b  h1 n c', h1 = self.num_heads)
#         v = einops.rearrange(v, 'b n  (h1  c) -> b  h1 n c', h1 = self.num_heads)
#
#         q  = q.unsqueeze(2)
#         attn= (q @ k.transpose(-2, -1))
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#
#         #x = attn @ v
#         x = (attn @ v).transpose(1, 2).squeeze(1).view(B,-1)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, attn

class Graph_encode(nn.Module):
    def __init__(self, graph_dim=1024, text_context=768, num_heads=16, num_layers=1, dropout_prob=0.1):
        super(Graph_encode, self).__init__()
        self.layers = nn.ModuleList([
            Graph_encode_layer(graph_dim, text_context, num_heads, dropout_prob) for _ in range(num_layers)
        ])

    def forward(self, graph, text_context):
        # print(text_context)
        for layer in self.layers:
            graph = layer(graph, text_context)
        return graph


class Graph_encode_layer(nn.Module):
    def __init__(self, graph_dim, text_context, num_heads, dropout_prob=0.1):
        super(Graph_encode_layer, self).__init__()
        self.attn = CrossAttention(query_dim=graph_dim, context_dim=text_context, heads=num_heads)
        self.graph_conv = GCNConv(graph_dim, graph_dim)
        self.norm1 = nn.LayerNorm(graph_dim)
        self.norm2 = nn.LayerNorm(graph_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, graph, text_context):
        # Input projection
        residual = graph.x
        graph_nodes_proj = graph.x
        text_context_proj = text_context.unsqueeze(0).repeat(graph_nodes_proj.shape[0], 1, 1)

        # Cross attention
        fused_nodes = self.attn(graph_nodes_proj.unsqueeze(1), text_context_proj).squeeze(1).detach()

        # Dropout after cross attention
        fused_nodes = self.dropout(fused_nodes)

        # Graph convolution + residual connection
        graph_encoded = self.graph_conv(fused_nodes, graph.edge_index)
        graph_encoded = self.norm1(graph_encoded) + residual

        # Dropout after graph convolution
        graph_encoded = self.dropout(graph_encoded)

        # Update graph node features
        graph.x = graph_encoded
        return graph


import matplotlib.pyplot as plt


def visualize_mask(instance_map, resized_mask, instance_id, b):
    plt.figure(figsize=(10, 5))

    # Original instance map
    plt.subplot(1, 2, 1)
    plt.title(f"Original Instance Map (ID: {instance_id})")
    plt.imshow(instance_map[b].cpu().numpy(), cmap="tab20")
    plt.axis("off")

    # Resized mask
    plt.subplot(1, 2, 2)
    plt.title("Resized Mask")
    plt.imshow(resized_mask.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")

    plt.show()


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Geo_Struct(nn.Module):
    def __init__(self, embedding_size=[32, 32], img_size=[256, 256], out_channels=256, npts = 2):
        super().__init__()

        self.prompt_layer = PromptEncoder(out_channels, image_embedding_size=embedding_size, input_image_size=img_size,
                                          mask_in_chans=64)

        self.out_channels = out_channels * npts


    def seg_drop(self, masks, drop_ratio=0.2):
        """
        Drops labels to enable classifier-free guidance.
        """
        # bs, c, h, w = seg.shape
        if drop_ratio > 0.0 and (self.training is True):
            mask = (torch.rand([masks.shape[0], 1, 1, 1]) > drop_ratio).float().to(masks.device)
            masks = masks * mask

        return masks

    def extract_instancemap(self, seg, instancemap):
        """
        Extract the instance map from (B, 1, H, W) to (B, IDS, H, W).

        Args:
            instancemap: Tensor of shape (B, 1, H, W)

        Returns:
            instance_masks: Tensor of shape (B, IDS, H, W)
            instance_ids: List of Tensors, each of shape (IDS,)
                          containing the instance IDs for each batch.
        """
        B, H, W = instancemap.shape

        labels = []

        for batch_idx in range(B):
            # Extract the instance map for the current batch
            batch_instancemap = instancemap[batch_idx]  # Shape (H, W)
            batch_seg = seg[batch_idx]
            # Get unique instance IDs, excluding the background (ID=0)
            unique_ids = torch.unique(batch_instancemap)
            unique_ids = unique_ids[unique_ids > 0]  # Exclude background

            # Create a mask for each unique ID
            batch_masks = []
            for instance_id in unique_ids:
                mask = (batch_instancemap == instance_id).float()  # Binary mask (H, W)
                label = batch_seg[:, mask.bool()].mean()
                batch_masks.append(mask)
                labels.append(label)

        batch_masks = torch.stack(batch_masks, dim=0)  # Shape (IDS, H, W)

        return batch_masks, labels

    def extract_object_regions(self, x, instance_map):
        objects = []
        B = instance_map.shape[0]
        for b in range(B):

            unique_ids = torch.unique(instance_map[b])
            for obj_id in unique_ids:
                if obj_id == 0:  # Skip background
                    continue
                mask = (instance_map[b] == obj_id)
                coords = torch.nonzero(mask, as_tuple=False)  # Coordinates of non-zero elements
                min_y, min_x = coords.min(0)[0]
                max_y, max_x = coords.max(0)[0]

                # Compute label/class using the mean of the corresponding pixels in x
                label = x[b, :, mask].mean(dim=-1)  # Take mean over the spatial dimension

                # Append the object information
                objects.append({
                    "cls": label.item(),  # Convert to Python scalar
                   # "mask": mask,
                    "bbox": [min_x.item(), min_y.item(), max_x.item(), max_y.item()],
                    "batch_id": b
                })
        return objects

    def four_points_to_aabb_torch(self, bboxes_4pts: torch.Tensor) -> torch.Tensor:
        """
        Convert 4-point boxes (N, 4, 2) to axis-aligned bounding boxes (N, 4)
        Format: [x_min, y_min, x_max, y_max]
        """
        x_min = bboxes_4pts[:, :, 0].min(dim=1).values
        y_min = bboxes_4pts[:, :, 1].min(dim=1).values
        x_max = bboxes_4pts[:, :, 0].max(dim=1).values
        y_max = bboxes_4pts[:, :, 1].max(dim=1).values
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)  # (N, 4)
    def forward_bboxes(self, bboxes):
        #bboxes = self.four_points_to_aabb_torch(bboxes)
        sparse_embeddings = self.prompt_layer._embed_boxes(boxes=bboxes)
        #sparse_embeddings = self.prompt_layer._embed_boxes_4pts(boxes=bboxes)
        sparse_embeddings = sparse_embeddings.view(sparse_embeddings.size(0), -1)

        # geo_embedding = sparse_embeddings
        geo_embedding = F.normalize(sparse_embeddings, p=2, dim=-1)

        return geo_embedding

    def forward(self, seg, instance_map):
        """
            Args:
                seg: Tensor of shape (B, 1, H, W) - Segmentation input
                instancemap: Tensor of shape (B, 1, H, W) - Instance segmentation map

            Returns:
                features: Tensor of shape (B, N, C) - Encoded features for each instance
                labels: Tensor of shape (B, N) - Instance labels for each instance
        """

        objects = self.extract_object_regions(seg, instance_map)

        bboxes = torch.stack([torch.tensor(obj['bbox']) for obj in objects], dim=0).to(seg.device)



        # masks = torch.stack([obj['mask'] for obj in objects], dim=0).float().unsqueeze(1).to(seg.device)
        # masks = self.seg_drop(masks, self.training)

        # use roialgin catch every features of mask

        # boxes = torch.cat(objects['bbox'],dim=0)

        # masks = torch.cat(objects['mask'],dim=0)
        sparse_embeddings = self.prompt_layer._embed_boxes(boxes=bboxes)

        sparse_embeddings = sparse_embeddings.view(sparse_embeddings.size(0), -1)

        geo_embedding = sparse_embeddings

        #
        # node_feats = []
        # labels = []
        # for obj in objects:
        #     mask = obj['mask']
        #     cls = int(obj['cls'] + 0.5)
        #     mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
        #     resized_mask = F.interpolate(mask.float(), size=(64, 64), mode='nearest')  # Resize to fixed size
        #     features = self.geo_embed(resized_mask)
        #     node_feats.append(features)
        #     labels.append(cls)
        # B, C, H, W = seg.shape
        # node_feats = []
        # labels = []
        # for b in range(B):
        #     max_ids = instance_map[b].max()
        #     unique_ids = torch.unique(instance_map[b])
        #     for instance_id in unique_ids:
        #         if instance_id == 0:  # Skip background
        #             continue
        #         instance_id = instance_id.long()
        #         mask = instance_map[b] == instance_id
        #         label = seg[b, :, mask].mean()
        #         labels.append(label)
        #
        #         resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode='nearest')  # Resize to fixed size
        #         #visualize_mask(instance_map,resized_mask,instance_id,b)
        #         #show resize_mask and original instance_instance_map
        #         mask_feats = self.geo_embed(resized_mask).view(1,-1)
        #         node_feats.append(mask_feats)

        # num_nodes, C
        return geo_embedding


from ldm.modules.diffusionmodules.openaimodel_ADE20K import AttentionPool2d


class SegEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_size=768, drop_out=0.3, image_size=256, ds=16):
        super(SegEncoder, self).__init__()


        self.seg_embed = PatchEmbed(image_size, ds, in_channels, hidden_size, norm_layer=nn.BatchNorm2d,
                                    output_fmt='NCHW')

        self.out = nn.Sequential(
            LayerNorm2d(hidden_size),
            nn.SiLU(),
            AttentionPool2d(
                (image_size // ds), hidden_size, 8
            ),
        )
        self.drop_rate = drop_out

    def _random_dropout(self, seg):
        """
        Drops labels to enable classifier-free guidance.
        """

        mask = (torch.rand([seg.shape[0], 1, 1, 1]) > self.drop_rate).float().to(seg.device)
        seg = seg * mask

        return seg

    def forward(self, seg, instance_map=None):
        """
        Forward pass of the segmentation encoder.

        Args:
            seg (torch.Tensor): Input segmentation tensor of shape (B, C, H, W).
            instance_map (torch.Tensor): Instance map tensor of shape (B, H, W).
            dropout_rate (float): Probability of dropping an instance (used during training).

        Returns:
            torch.Tensor: Encoded segmentation features.
        """
        if self.training and self.drop_rate > 0:
            seg = self._random_dropout(seg)

        seg_feats = self.seg_embed(seg.float())
        out = self.out(seg_feats)

        return out.permute(0, 2, 1)

import cv2
def generate_colors(n):
    """生成 n 种 RGB 颜色"""
    cmap = plt.cm.get_cmap('tab20', n)
    return [tuple(int(255 * c) for c in cmap(i)[:3]) for i in range(n)]

def draw_batched_bboxes_on_instance_map(instance_map, bboxes_batch, thickness=2):
    """
    批量绘制 bounding boxes 到每张 instance_map 上，每个框不同颜色。

    Args:
        instance_map (Tensor): [B, H, W] torch.Tensor
        bboxes_batch (list[Tensor] or Tensor): 每个 batch 元素为 [N_i, 4] 的框
        thickness (int)

    Returns:
        vis_imgs (list[np.ndarray]): 每个 batch 元素的彩色图像
    """
    B = instance_map.shape[0]
    vis_imgs = []

    for b in range(B):
        img = instance_map[b]
        bboxes = bboxes_batch # 每个 batch 的框

        # 转为 RGB 彩色图
        img_np = img.detach().cpu().numpy()
        vis_img = np.stack([img_np] * 3, axis=-1).astype(np.uint8)

        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()

        n_box = len(bboxes)
        colors = generate_colors(n_box)

        for i, box in enumerate(bboxes):
            pts = box.astype(int).reshape(-1, 2)  # 确保是 (4, 2)
            for j in range(4):
                pt1 = tuple(pts[j])
                pt2 = tuple(pts[(j + 1) % 4])  # 连下一点，最后一个点连第一个
                vis_img = cv2.line(vis_img, pt1, pt2, colors[i], thickness)

        vis_imgs.append(vis_img)

    return vis_imgs


class GDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size = 32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            graph_dim=768,
            depth=28,
            num_heads=16,
            graph_tokens = 128,
            mlp_ratio=4.0,
            node_dropout_prob=0.1,
            seg_dropout_prob=0.1,
            num_classes=1000,  # 0 = unconditional
            learn_sigma=False,
            with_graph=False,
            fusion_type='AdaLn',  # or 'concat'
            mode='seg',  # or graph or seg with graph
            use_qknorm=False,
            use_swiglu=False,
            use_rmsnorm=False,


            **kwargs
    ):
        super(GDiT, self).__init__()
        self.seg_drop = seg_dropout_prob
        self.node_drop = node_dropout_prob
        self.fusion_type = fusion_type
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.node_dropout_prob = node_dropout_prob
        self.num_classes = num_classes
        self.mode = mode
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats = hidden_size // 2)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size) if num_classes else None
        boxes_npts = 4
        self.box_embeder = Geo_Struct(embedding_size=[input_size//patch_size, input_size//patch_size], img_size=[input_size * 8, input_size * 8])

        self.graph_embedder = nn.Linear(graph_dim, hidden_size)

        self.seg_encoder = SegEncoder(in_channels=1, hidden_size=hidden_size, drop_out = seg_dropout_prob, image_size= input_size * 8)

        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )



        self.blocks = nn.ModuleList([
            GDiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                with_graph=True,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
            )
            for _ in range(depth)
        ])





        self.with_graph = with_graph

        if with_graph:
            self.graph_encoder = GraphEncoder(
                in_dim=  graph_dim + self.box_embeder.out_channels,
                #in_dim=graph_dim * 2,
                out_dim=hidden_size,
                num_heads=8
            )
            #self.node_predictor = nn.Linear(hidden_size, hidden_size)
            # self.seg_to_graph_attn = CrossAttention(hidden_size, num_heads=8, qkv_bias=True)
        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, self.out_channels)

        self.target_nodes = graph_tokens
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


        # Zero-out adaLN modulation layers in DiT blocks:

        for block in self.blocks:
            nn.init.constant_(block.attn_node_to_x.to_out[0].weight, 0)
            nn.init.constant_(block.attn_node_to_x.to_out[0].bias, 0)

        # Zero-out output layers:
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    def update_keep_mask(self, keep_mask, batch, target_nodes=None):
        target_nodes = int(target_nodes if target_nodes is not None else self.num_tokens)
        bs = batch.max().item() + 1


        node_mask = torch.zeros(bs, target_nodes, device=keep_mask.device)

        for b in range(bs):
            idx = (batch == b).nonzero(as_tuple=False).squeeze(-1)
            keep_b = keep_mask[idx]
            N = keep_b.size(0)
            if N >= target_nodes:
                node_mask[b] = 1
            else:
                node_mask[b, :N] = keep_b

        return node_mask

    def forward(self, x, t, y, seg,  feat=None, graph=None,instance_map=None, use_checkpoint = True, **model_kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        import time
        x_pe = self.pe_layer.forward(size=(x.shape[2] // self.patch_size, x.shape[3] // self.patch_size))
        x_pe = einops.rearrange(x_pe, 'c h w -> ( h w) c').unsqueeze(0)
        x = self.x_embedder(x)
        x = x + x_pe



        t = self.t_embedder(t)  # (N, D)
        c = t
        if self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        # if feat is not None:
        #     # feat: [B, style_dim]
        #     style_emb = self.style_proj(feat)  # [B, hidden_size]
        #     c = c + style_emb


        c2 = self.t_block(c)


        #extract graph_token
        if graph is not None:


            bboxes = graph.node_boxxes

            #bboxes = torch.from_numpy(np.stack(bboxes, axis=0)).cuda()

            box_embeddings = self.box_embeder.forward_bboxes(bboxes)
            #graph.x = torch.cat((graph.x ,box_embeddings),dim = 1)
            #box_embeddings = self.box_embeder(seg, instance_map) if feat is None else self.box_embeder.forward_bboxes(boxes)

            if not self.training:
                self.target_nodes = instance_map.max()
            #if self.training:
            # with torch.no_grad() :
            #     # t0 = time.time()
            #     _, node_tokens_gt, _ = self.graph_encoder(graph, self.target_nodes)
                # t1 = time.time()

            # Dropout + fusion
            #start_d = time.time()
            drop_graph, pe_embeddings, instance_map, semantic_map , keep_mask = dropout_nodes_in_graph(
                graph, instance_map, semantic_map=seg, pe_embeddings=box_embeddings,
                dropout_prob=self.node_dropout_prob if self.training else 0.0, fusion='Add')

            keep_mask = self.update_keep_mask(keep_mask, graph.batch, target_nodes=self.target_nodes)


            # Graph encoding
            #start_enc = time.time()
            _, node_tokens, node_mask = self.graph_encoder(drop_graph, self.target_nodes)
            node_mask = node_mask.bool() & keep_mask.bool()
            #end_enc = time.time()

            # Node prediction
            #start_pred = time.time()
            #node_pred = self.node_predictor(node_tokens_drop)
            #end_pred = time.time()

            # Masked conditional
            if node_tokens.shape[0] == x.shape[0] // 2:
                node_tokens = torch.cat((node_tokens,torch.zeros_like(node_tokens)), dim=0)
                node_mask = torch.cat((node_mask, torch.zeros_like(node_mask)), dim=0)
                # node_tokens = torch.cat((torch.zeros_like(node_tokens),node_tokens), dim=0)
                # node_mask = torch.cat(( torch.zeros_like(node_mask),node_mask), dim=0)


            if "flag" in model_kwargs.keys() and not self.training:
                if model_kwargs["flag"] == 'regions':
                    node_tokens = 0.0 * node_tokens
                    semantic_map = 0.0 * semantic_map
                    node_mask = 0.0 * node_mask
                elif model_kwargs['flag'] == 'regions_graph':
                    semantic_map = 0.0 * semantic_map
                elif model_kwargs['flag'] == 'regions_sem':
                    node_tokens = 0.0 * node_tokens
            g_seg = self.seg_encoder(semantic_map)
            # end_seg = time.time()
            if g_seg is not None:
                x = x + g_seg[:, 1:, :]

        else:
            semantic_map = seg * 0.0
            B = x.shape[0]
            N = self.target_nodes  # 你可以在 __init__ 里默认设个值，例如 graph_tokens
            D = x.shape[-1]
            node_tokens = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
            node_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            semantic_map = seg * 0.0  # 或直接构造一个全 0 mask，shape 跟源域一样

            g_seg_tgt = self.seg_encoder(semantic_map)
            x = x + g_seg_tgt[:, 1:, :]
        # Semantic encoder
        #start_seg = time.time()

        out = dict()

        for block in self.blocks:
            x = block(x, c2, node_tokens, node_mask)


        x = self.final_layer(x, c, context=None)  # (N, T or T+1, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        out['x'] = x

        return out

    # 在 GDiT 里新增
    # def forward_with_multi_cfg(
    #         self,
    #         x, t,
    #         y,  # region 文本嵌入/one-hot 等，原来的 y
    #         seg,  # 语义图
    #         cfg_region=1.0,  # region 控制强度
    #         cfg_seg=1.0,  # seg 控制强度
    #         cfg_graph=1.0,  # graph 控制强度
    #         feat=None,
    #         graph=None,
    #         instance_map=None,
    #         **model_kwargs
    # ):
    #     """
    #
    #     eps = eps_u
    #         + cfg_region * (eps_R - eps_u)
    #         + cfg_seg    * (eps_S - eps_u)
    #         + cfg_graph  * (eps_G - eps_u)
    #     """
    #
    #     B = x.shape[0]
    #
    #     region_zero = torch.zeros_like(y) if y is not None else None
    #     seg_zero = torch.zeros_like(seg) if seg is not None else None
    #
    #     # 1) 完全无条件：R 关，S 关，G 关
    #     out_u = self.forward(
    #         x, t,
    #         y=region_zero,
    #         seg=seg_zero,
    #         feat=feat,
    #         graph=None,
    #         instance_map=instance_map,
    #         **model_kwargs
    #     )['x']
    #
    #     eps_u, rest = out_u[:, :self.in_channels], out_u[:, self.in_channels:]
    #
    #     # 2) 只有 Region：R 开，S 关，G 关
    #     out_R = self.forward(
    #         x, t,
    #         y=y,
    #         seg=seg_zero,
    #         feat=feat,
    #         graph=None,
    #         instance_map=instance_map,
    #         **model_kwargs
    #     )['x']
    #     eps_R = out_R[:, :self.in_channels]
    #
    #     # 3) 只有 Seg：R 关，S 开，G 关
    #     out_S = self.forward(
    #         x, t,
    #         y=region_zero,
    #         seg=seg,
    #         feat=feat,
    #         graph=None,
    #         instance_map=instance_map,
    #         **model_kwargs
    #     )['x']
    #     eps_S = out_S[:, :self.in_channels]
    #
    #     # 4) 只有 Graph：R 关，S 关，G 开
    #     if graph is not None:
    #         out_G = self.forward(
    #             x, t,
    #             y=region_zero,
    #             seg=seg_zero,
    #             feat=feat,
    #             graph=graph,
    #             instance_map=instance_map,
    #             **model_kwargs
    #         )['x']
    #         eps_G = out_G[:, :self.in_channels]
    #     else:
    #         eps_G = eps_u  # 没有 graph 时，等价于无条件
    #
    #     # ---------- 按权重组合 ----------
    #     eps = (
    #             eps_u
    #             + cfg_region * (eps_R - eps_u)
    #             + cfg_seg * (eps_S - eps_u)
    #             + cfg_graph * (eps_G - eps_u)
    #     )
    #
    #     x_out = torch.cat([eps, rest], dim=1)
    #     return {'x': x_out}

    def forward_with_cfg(self, x, t, y, seg, cfg_scale, feat=None, graph=None, instance_map=None, **model_kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        out = dict()

        # Setup classifier-free guidance
        # x = torch.cat([x, x], 0)
        # y_null = torch.zeros_like(y)
        # y = torch.cat([y, y_null], 0)
        # if feat is not None:
        #     feat = torch.cat([feat, feat], 0)

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        # assert self.num_classes and y is not None
        model_out = self.forward(combined, t, y, seg, feat=feat, graph=graph, instance_map=instance_map, **model_kwargs)['x']
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)



        eps = torch.cat([half_eps, uncond_eps], dim=0)

        # uncond_eps = uncond_eps + cfg_scale * (uncond_eps - cond_eps)
        # eps = torch.cat([cond_eps, uncond_eps], dim=0)


        x = torch.cat([eps, rest], dim=1)
        out['x'] = x

        return out
        # half_rest = rest[: len(rest) // 2]
        # x = torch.cat([half_eps, half_rest], dim=1)
        # out['x'] = x
        # return out

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size = 32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            graph_dim=768,
            depth=28,
            num_heads=16,
            graph_tokens = 128,
            mlp_ratio=4.0,
            node_dropout_prob=0.1,
            seg_dropout_prob=0.1,
            num_classes=1000,  # 0 = unconditional
            learn_sigma=False,
            with_graph=False,
            fusion_type='AdaLn',  # or 'concat'
            mode='seg',  # or graph or seg with graph
            use_qknorm=False,
            use_swiglu=False,
            use_rmsnorm=False,


            **kwargs
    ):
        super(DiT, self).__init__()
        self.seg_drop = seg_dropout_prob
        self.node_drop = node_dropout_prob
        self.fusion_type = fusion_type
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.node_dropout_prob = node_dropout_prob
        self.num_classes = num_classes
        self.mode = mode
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats = hidden_size // 2)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size) if num_classes else None
        self.box_embeder = Geo_Struct(embedding_size=[input_size//patch_size, input_size//patch_size], img_size=[input_size * 8, input_size * 8], out_channels=128, npts = 4)

        self.graph_embedder = nn.Linear(graph_dim, hidden_size)

        self.seg_encoder = SegEncoder(in_channels=1, hidden_size=hidden_size, drop_out = seg_dropout_prob, image_size= input_size * 8)

        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )



        self.blocks = nn.ModuleList([
            GDiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                with_graph=True,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
            )
            for _ in range(depth)
        ])





        self.with_graph = with_graph

        if with_graph:
            self.graph_encoder = GraphEncoder(
                in_dim=graph_dim + self.box_embeder.out_channels,
                out_dim=hidden_size,
                num_heads=8
            )
            # self.seg_to_graph_attn = CrossAttention(hidden_size, num_heads=8, qkv_bias=True)
        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, self.out_channels)

        self.target_nodes = graph_tokens
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


        # Zero-out adaLN modulation layers in DiT blocks:

        for block in self.blocks:
            nn.init.constant_(block.attn_node_to_x.to_out[0].weight, 0)
            nn.init.constant_(block.attn_node_to_x.to_out[0].bias, 0)

        # Zero-out output layers:
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


    def forward(self, x, t, y, seg,  feat=None, graph=None,instance_map=None, use_checkpoint = True, **model_kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        x_pe = self.pe_layer.forward(size=(x.shape[2] // self.patch_size, x.shape[3] // self.patch_size))
        x_pe = einops.rearrange(x_pe, 'c h w -> ( h w) c').unsqueeze(0)
        x = self.x_embedder(x)
        x = x + x_pe



        t = self.t_embedder(t)  # (N, D)
        c = t
        if self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        c2 = self.t_block(c)


        #extract graph_token
        if graph is not None:


            bboxes = graph.node_boxxes



            box_embeddings = self.box_embeder.forward_bboxes(bboxes)

            #box_embeddings = self.box_embeder(seg, instance_map) if feat is None else self.box_embeder.forward_bboxes(boxes)

            if not self.training:
                self.target_nodes = instance_map.max()


            drop_graph, pe_embeddings, instance_map, semantic_map = dropout_nodes_in_graph(graph, instance_map, semantic_map = seg, pe_embeddings = box_embeddings, dropout_prob = self.node_dropout_prob if self.training else 0.0)


            _, node_tokens, node_mask = self.graph_encoder(drop_graph, self.target_nodes)  # only graph ?

            #c = g_c[:,0,:] + c

            if node_tokens.shape[0] == x.shape[0] // 2:
                node_tokens = torch.cat((node_tokens, torch.zeros_like(node_tokens)), dim=0)
                node_mask = torch.cat((node_mask, torch.zeros_like(node_mask)), dim=0)

            if "flag" in model_kwargs.keys() and not self.training:
                if model_kwargs["flag"] == 'regions':
                    node_tokens = 0.0 * node_tokens
                    semantic_map = 0.0 * semantic_map
                elif model_kwargs['flag'] == 'regions_graph':
                    semantic_map = 0.0 * semantic_map
                elif model_kwargs['flag'] == 'regions_sem':
                    node_tokens = 0.0 * node_tokens




            g_seg = self.seg_encoder(semantic_map)
            if g_seg is not None:

                x = x + g_seg[:, 1:, :]





        out = dict()


        for block in self.blocks:

            # if use_checkpoint:
            #     x = checkpoint(
            #         lambda x: block(x, c2, node_tokens, node_mask), x,
            #         use_reentrant=False
            #     )
            # else:
            x = block(x, c2, node_tokens, node_mask)

        # for block in self.blocks:
        #     x = block(x, c2, node_tokens, node_mask)  # (N, T, D)

        x = self.final_layer(x, c, context=None)  # (N, T or T+1, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        out['x'] = x

        return out

    def forward_with_cfg(self, x, t, y, seg, cfg_scale, feat=None, graph=None, instance_map=None, **model_kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        out = dict()

        # Setup classifier-free guidance
        # x = torch.cat([x, x], 0)
        # y_null = torch.zeros_like(y)
        # y = torch.cat([y, y_null], 0)
        # if feat is not None:
        #     feat = torch.cat([feat, feat], 0)

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        # assert self.num_classes and y is not None
        model_out = self.forward(combined, t, y, seg, feat=feat, graph=graph, instance_map=instance_map, **model_kwargs)['x']
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, uncond_eps], dim=0)
        x = torch.cat([eps, rest], dim=1)
        out['x'] = x

        return out
        # half_rest = rest[: len(rest) // 2]
        # x = torch.cat([half_eps, half_rest], dim=1)
        # out['x'] = x
        # return out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=1):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def GDiT_XL_2(**kwargs):
    return GDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)




DiT_models = {'GDiT-XL/2': GDiT_XL_2}

# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

from ldm.modules.diffusionmodules.openaimodel_ADE20K import UNetModel



class EDMPrecond(nn.Module):
    def __init__(self,
                 img_resolution=32,  # Image resolution.
                 img_channels=4,  # Number of color channels.
                 num_classes=0,  # Number of class labels, 0 = unconditional.
                 sigma_min=0,  # Minimum supported noise level.
                 sigma_max=float('inf'),  # Maximum supported noise level.
                 sigma_data=0.5,  # Expected standard deviation of the training data.
                 model_type='DiT-B/2',  # Class name of the underlying model.
                 **model_kwargs,  # Keyword arguments for the underlying model.
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        if 'GDiT' in model_type:
            self.model = DiT_models[model_type](num_classes=num_classes, **model_kwargs)

        else:
            self.model = UNetModel(**model_kwargs)

    def encode(self, x, sigma, class_labels=None, **model_kwargs):

        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        class_labels = None if self.num_classes == 0 else \
            torch.zeros([x.shape[0], self.num_classes], device=x.device) if class_labels is None else \
                class_labels.to(x.dtype).reshape(-1, self.num_classes)

        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        feat, mask_dict = self.model.encode((c_in * x).to(x.dtype), c_noise.flatten(), y=class_labels, **model_kwargs)
        return feat

    def forward(self, x, sigma, class_labels=None, seg=None, cfg_scale=None, graph = None, instance_map = None, **model_kwargs):
        model_fn = self.model if cfg_scale is None else partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)

        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        class_labels = None if self.num_classes == 0 else \
            torch.zeros([x.shape[0], self.num_classes], device=x.device) if class_labels is None else \
                class_labels.to(x.dtype).reshape(-1, self.num_classes)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        model_out = model_fn((c_in * x).to(x.dtype), c_noise.flatten(), y=class_labels, seg=seg, graph = graph,instance_map = instance_map, **model_kwargs)
        F_x = model_out['x']
        D_x = c_skip * x + c_out * F_x
        model_out['x'] = D_x
        # show_denoised(D_x, vae)
        return model_out

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)





Precond_models = {
    'edm': EDMPrecond
}




if __name__ == '__main__':
    # Dummy data
    # Dummy data
    B, L, D = 4, 16 * 16, 64  # Batch size, sequence length, embedding dimension
    H, W = 256, 256  # Target spatial dimensions
    num_classes, text_dim = 9, 768  # Text context dimensions

    x = torch.randn(B, L, D)  # Visual embeddings
    text_context = torch.randn(num_classes, text_dim)  # Text embeddings
    instance_map = torch.randint(1, 10, (B, H, W))  # Instance map

    # Pre-stored graph data
    graph = [
        {"edge_index": torch.randint(0, 100, (2, 300)), "batch": torch.zeros(100, dtype=torch.long)},
        # Repeat for additional batches
    ]

    # Initialize and run the model
    model = GraphFusion(in_dim=D, text_dim=text_dim, graph_dim=128)
    output = model.forward(x, text_context, graph, instance_map)

    print(f"Output shape: {output.shape}")  # Expected: [B, L, D]