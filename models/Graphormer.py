
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import models.clip as clip
import einops
class GATLayer(nn.Module):
    def __init__(self, in_features=768, out_features=768, dropout=0.1, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)


    def forward(self, h0, h1, node_l1, edge_l1, node_l2, edge_l2, node_l3, edge_l3):
        #h0  h1
        Wh0 = torch.einsum('bnd,de->bne', [h0, self.W])
        Wh1 = torch.einsum('bnd,de->bne', [h1, self.W])

        a_input = self._prepare_attentional_mechanism_input(Wh0, Wh1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        e_l1 = self.leakyrelu(torch.matmul(a_input, node_l1).squeeze(3))
        e_l2 = self.leakyrelu(torch.matmul(a_input, node_l2).squeeze(3))
        e_l3 = self.leakyrelu(torch.matmul(a_input, node_l3).squeeze(3))


        zero_vec = -9e15 * torch.ones_like(e)
        attention = e

        zero_vec = torch.zeros_like(e_l1)
        attention_l1 = torch.where(edge_l1 > 0, e_l1, zero_vec)
        attention_l2 = torch.where(edge_l2 > 0, e_l2, zero_vec)
        attention_l3 = torch.where(edge_l3 > 0, e_l3, zero_vec)


        attention = F.softmax(attention + 0.01*(attention_l1 + attention_l2 + attention_l3), dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh0, Wh1):
        N0, N1 = Wh0.size()[1], Wh1.size()[1]
        Wh0_repeated_in_chunks = Wh0.repeat_interleave(N1, dim=1)
        Wh1_repeated_alternating = Wh1.repeat(1, N0, 1)
        all_combinations_matrix = torch.cat([Wh0_repeated_in_chunks, Wh1_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(-1, N0, N1, 2 * self.out_features)


class BasicTransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicTransformerBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=512, dropout=0.1)

    def forward(self, x):
        return self.transformer(x)


class MultiStageTransformer(nn.Module):
    def __init__(self,
                 iterations_stages = [2, 4, 6],
                 in_dim = 512,
                 out_dim = 512,
                 num_heads = 4):
        super(MultiStageTransformer, self).__init__()

        self.stage1 = nn.ModuleList([
            TransformerConv(in_dim, out_dim, heads=num_heads, concat=False)
            for _ in range(iterations_stages[0])])
        self.stage2 = nn.ModuleList([
            TransformerConv(in_dim, out_dim, heads=num_heads, concat=False)
            for _ in range(iterations_stages[1])])
        self.stage3 = nn.ModuleList([
            TransformerConv(in_dim, out_dim, heads=num_heads, concat=False)
            for _ in range(iterations_stages[2])])


    def forward(self, x1, x1_edge, batch_l1,  x2 = None, x2_edge = None, x3 = None, x3_edge = None, batch_l2 = None,  batch_l3 = None):
        # Stage 1 Processing
        for layer in self.stage1:
            x1 = layer(x1, x1_edge)
        x1_g = global_mean_pool(x1, batch_l1)

        x2_g, x3_g = None,None
        # Stage 2 Processing
        if x2 is not  None:
            x2 = torch.cat((x2, x1_g), dim=0)
            for layer in self.stage2:
                x2 = layer(x2, x2_edge)
            x2_g = global_mean_pool(x2[:batch_l2.shape[0]], batch_l2)
        if x3 is not None:
            # Stage 3 Processing
            x3 = torch.cat((x3, x1_g, x2_g), dim=0)
            for layer in self.stage3:
                x3 = layer(x3, x3_edge)
            x3_g = global_mean_pool(x3[:batch_l3.shape[0]], batch_l3)
        return x1_g, x2_g ,x3_g

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, attn_drop = 0.,dropout=0.):
        super().__init__()
        inner_dim = query_dim

        head_dim = query_dim // heads
        self.scale = head_dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attn_drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    def forward(self, q, k ,v):


        h = self.heads

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = q * self.scale  # , q2 * self.scale

        q = einops.rearrange(q, 'n  (h1  c) -> h1 n c', h1 = self.heads)
        k = einops.rearrange(k, 'n  (h1   c) -> h1 n c', h1 = self.heads)
        v = einops.rearrange(v, 'n  (h1   c) -> h1 n c', h1 = self.heads)

        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)


        #x = attn @ v
        x = (attn @ v).transpose(1, 2)
        x = einops.rearrange(x, ' h1 c n-> n (h1 c)', h1 = self.heads)
        x = self.to_out(x)
        return  x
        # sim =  einsum('b h i d, b j d -> b i j', q, k) * self.scale
        #
        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)
        #
        # # if exists(label) and exists(class_ids):
        # #     B = label.shape[0]
        # #     H = int(math.sqrt(q.shape[1]))
        # #     W = H
        # #     mask = torch.ones(B, H, W, 77)
        # #     for ii in range(B):
        # #         index = []
        # #         ids = np.argwhere(class_ids[ii].detach().cpu().numpy() > 0)
        # #         for jj in range(len(ids)):
        # #             index.append(ids[jj][0])
        # #             ### For those words with more than one token, we need to apply rectification to multiple attention maps
        # #             if ids[jj] in [39, 41, 44, 57, 59, 63, 65, 71, 75, 76, 79, 80, 87, 88, 92, 96, 97, 98, 100, 106, 109, 110, 135, 137, 139, 145]:
        # #                 index.append(ids[jj][0])
        # #             if ids[jj] in [49]:
        # #                 index.append(ids[jj][0])
        # #                 index.append(ids[jj][0])
        # #         for kk in range(len(index)):
        # #             tmp_mask = torch.zeros_like(label[ii]) # [h,w]
        # #             tmp_mask[label[ii]==index[kk]] = 1
        # #             tmp_mask = F.interpolate(tmp_mask.unsqueeze(0).unsqueeze(0), (H, W), mode="nearest")[0,0,:,:]
        # #             mask[ii,:,:,kk+1] = tmp_mask
        # #             del tmp_mask
        # #     mask = rearrange(mask, 'b h w c -> b (h w) c')
        # #     mask = repeat(mask, 'b n c -> (b h) n c', h=h)
        # #     mask = mask.to(q.device)
        # #     mask = mask > 0.5
        # #     max_neg_value = -torch.finfo(sim.dtype).max
        # #     sim.masked_fill_(~mask, max_neg_value)
        # #     del mask
        #
        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim=-1)
        #
        # out = einsum('b i j, b j d -> b i d', attn, v)
        # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # return self.to_out(out)


def global_mean_pool_with_mask(x, batch, mask):
    """
    带掩码的 global_mean_pool，实现对有效节点的均值计算。

    Args:
        x (Tensor): 节点特征，形状为 [num_nodes, feature_dim]。
        batch (Tensor): 节点对应的图索引，形状为 [num_nodes]。
        mask (Tensor): 节点掩码，形状为 [num_nodes]，1 表示有效，0 表示被忽略。

    Returns:
        Tensor: 每个图的均值特征，形状为 [num_graphs, feature_dim]。
    """
    # 1. 获取图的数量和特征维度
    num_graphs = batch.max().item() + 1
    feature_dim = x.size(1)

    # 2. 初始化图特征求和张量
    graph_sum = torch.zeros((num_graphs, feature_dim), dtype=x.dtype, device=x.device)

    # 3. 计算有效节点特征的总和
    masked_x = x * mask.unsqueeze(-1)  # 将无效节点特征置零
    for node_idx in range(batch.size(0)):
        graph_sum[batch[node_idx]] += masked_x[node_idx]

    # 4. 计算每个图的有效节点数量
    graph_node_count = torch.zeros(num_graphs, dtype=torch.float32, device=mask.device)
    for node_idx in range(batch.size(0)):
        graph_node_count[batch[node_idx]] += mask[node_idx]

    # 5. 归一化，避免除以 0
    graph_node_count[graph_node_count == 0] = 1  # 防止空图
    graph_mean = graph_sum / graph_node_count.unsqueeze(-1)

    return graph_mean

from torch_geometric.utils import subgraph
class GraphTransformer(nn.Module):
    def __init__(self, in_dim=768, out_dim=64, num_heads=8, depth=1, drop_prob = 0.2):
        super(GraphTransformer, self).__init__()

        self.node_emb = nn.Linear(in_dim, in_dim)

        self.gat_layers = nn.ModuleList([
            TransformerConv(in_dim, out_dim, heads=num_heads, concat=False)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(out_dim)
        self.drop_prob = drop_prob  # 节点丢弃概率



    def drop_out_node_and_edge(self, graph, training=False):
        """
        随机丢弃节点和边，并返回掩码和更新后的图结构。

        Args:
            graph (Data): PyTorch Geometric 格式的图数据。
            training (bool): 是否处于训练模式。

        Returns:
            Data: 更新后的图数据。
            Tensor: 节点掩码，形状为 [num_nodes]。
            Tensor: 边掩码，形状为 [num_edges]。
        """
        if not training or self.drop_prob <= 0:
            # 如果不是训练模式或丢弃概率为0，则保留所有节点和边
            num_nodes = graph.x.size(0)
            num_edges = graph.edge_index.size(1)
            node_mask = torch.ones(num_nodes, dtype=torch.float32, device=graph.x.device)
            edge_mask = torch.ones(num_edges, dtype=torch.bool, device=graph.x.device)
            return graph, node_mask, edge_mask

        # 节点丢弃掩码
        num_nodes = graph.x.size(0)
        node_mask = torch.bernoulli(torch.full((num_nodes,), 1 - self.drop_prob, device=graph.x.device))

        # 边掩码：根据节点掩码推导
        edge_index = graph.edge_index
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        edge_mask = (node_mask[src_nodes] * node_mask[dst_nodes]).bool()

        # 更新边索引
        new_edge_index = edge_index[:, edge_mask]

        # 更新后的图数据
        new_graph = graph.clone()
        new_graph.edge_index = new_edge_index
        return new_graph, node_mask, edge_mask




    def forward(self, graph, add_feats = None):
        """
        改进后的前向传播，支持子图实现，并在所有节点被 mask 的情况下用 -1 填充特征。
        """
        # 更新子图，移除被 mask 的节点及其相关边
        graph, node_mask, edge_mask = self.drop_out_node_and_edge(graph, self.training)
        node_idx = node_mask.bool()  # 有效节点掩码

        # 获取批次和图数量
        batch_size = graph.batch.max().item() + 1
        feature_dim = self.gat_layers[-1].out_channels  # GAT 最后一层输出特征维度

        # 如果所有节点都被 mask，则用 0 填充
        if node_idx.sum() == 0:
            return torch.zeros((batch_size, feature_dim),  dtype=graph.x.dtype, device=graph.x.device)

        # 子图处理：移除被 mask 的节点及边
        edge_l1, _ = subgraph(node_idx, graph.edge_index, relabel_nodes=True, num_nodes=node_mask.size(0))
        node_l1 = graph.x[node_idx]

        if add_feats is not None:
            node_l1 = torch.cat([node_l1, add_feats[node_idx]], dim=-1)

        # 节点特征映射
        node_l1 = self.node_emb(node_l1.float())

        # 通过 GAT 层
        x = node_l1
        for layer in self.gat_layers:
            x = layer(x, edge_l1)

        # 获取未被 mask 的节点的 batch 索引
        batch_l1 = graph.batch[node_idx]

        unique_batch_l1, _ = torch.unique(batch_l1, return_inverse=True)
        # 使用 global_mean_pool 对有效节点进行池化
        x_pooled = global_mean_pool(x, batch_l1)[unique_batch_l1]

        # 计算每个图的有效节点数量

        output = torch.zeros((batch_size, feature_dim), dtype=graph.x.dtype, device=graph.x.device)
        output[unique_batch_l1] = x_pooled

        return output



        #
        # for stage in range(0, 3):
        #     if stage == 1:
        #         l1_g =  self.node_emb(node_l1.float())
        #         for layer in self.layers_stage1:
        #             l1_g = layer(l1_g, edge_l1)
        #             l1_g = F.relu(l1_g)
        #             l1_g = self.norm(l1_g)
        #             l1_g = global_mean_pool(l1_g, l1_g.shape[0])
        #     elif stage == 2:
        #         l2_g = torch.cat((l1_g, node_l2))
        #         l2_g = self.node_emb(l2_g)
        #         for layer in self.layers_stage2:
        #             l2_g = layer(l2_g, edge_l2)
        #             l2_g = F.relu(l2_g)
        #     else :
        #         l3_g = torch.cat((l2_g, node_l3))
        #         l3_g = self.node_emb(l3_g)
        #         for layer in self.layers_stage2:
        #             l3_g = layer(l3_g, edge_l3)
        #             l3_g = F.relu(l3_g)








        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = x[:, self.select_num]
        # edge_index = edge_index
        # # 初始节点特征转换
        # x = self.node_emb(x.float())
        #
        #
        # for layer in self.layers:
        #     x = layer(x, edge_index)
        #     x = F.relu(x)
        #
        # x = self.norm(x)
        # x = global_mean_pool(x, batch)

        # return x
from torch_geometric.data import Batch, Data
from train_utils.helper import _contrust_graph

@staticmethod
def compute_box_area(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return ((x2 - x1) * (y2 - y1)).clamp(min=1e-6)

@staticmethod
def compute_polygon_area(boxes):
    x = boxes[:, :, 0]
    y = boxes[:, :, 1]
    area = 0.5 * torch.abs(
        x[:, 0]*y[:, 1] + x[:, 1]*y[:, 2] + x[:, 2]*y[:, 3] + x[:, 3]*y[:, 0]
        - y[:, 0]*x[:, 1] - y[:, 1]*x[:, 2] - y[:, 2]*x[:, 3] - y[:, 3]*x[:, 0]
    )
    return area + 1e-6
class GraphEncoder(nn.Module):
    def __init__(self, in_dim=768, out_dim=64, num_heads=8, depth=1, drop_prob = 0.2 , num_tokens = 128, image_size = 512):
        super(GraphEncoder, self).__init__()



        self.gat_layers = nn.ModuleList([
            TransformerConv(in_dim, in_dim, heads=num_heads, concat=False)
            for _ in range(depth)
        ])
        self.node_emb = nn.Linear(in_dim, out_dim)
        self.drop_prob = drop_prob  # 节点丢弃概率
        self.num_tokens = num_tokens
        self.input_imgsz  = image_size
    # def resize_graph_to_num_tokens(self, node_feats, batch, target_nodes = None):
    #     target_nodes = int(target_nodes if target_nodes is not None else self.num_tokens)
    #
    #
    #
    #     bs = batch.max() + 1
    #     target_node_feats = torch.zeros(bs, target_nodes, node_feats.size(1), device=node_feats.device)
    #     node_mask  = torch.ones(bs, target_nodes, device=node_feats.device)
    #     for b in range(bs):
    #         b_node_feats = node_feats[batch == b]
    #         num_nodes, feature_dim = b_node_feats.shape
    #         if num_nodes < target_nodes:
    #             padding = torch.zeros((target_nodes - num_nodes, feature_dim), device=b_node_feats.device)
    #             node_features = torch.cat([b_node_feats, padding], dim=0)
    #             node_mask[b, num_nodes:target_nodes] = 0
    #         else:
    #             selected_nodes = torch.randperm(num_nodes)[:target_nodes]
    #             # Create a mask for the selected nodes
    #             node_features = b_node_feats[selected_nodes]
    #         target_node_feats[b, :] = node_features
    #     return target_node_feats, node_mask

    def resize_graph_to_num_tokens(self, node_feats, batch, target_nodes=None):
        target_nodes = int(target_nodes if target_nodes is not None else self.num_tokens)
        bs = batch.max().item() + 1
        feature_dim = node_feats.size(1)

        target_node_feats = torch.zeros(bs, target_nodes, feature_dim, device=node_feats.device)
        node_mask = torch.zeros(bs, target_nodes, device=node_feats.device)

        for b in range(bs):
            b_mask = (batch == b)
            b_node_feats = node_feats[b_mask]
            num_nodes = b_node_feats.size(0)

            if num_nodes <= target_nodes:
                target_node_feats[b, :num_nodes] = b_node_feats
                node_mask[b, :num_nodes] = 1
            else:
                selected_indices = torch.randperm(num_nodes, device=node_feats.device)[:target_nodes]
                target_node_feats[b] = b_node_feats[selected_indices]
                node_mask[b, :] = 1

        return target_node_feats, node_mask.bool()

    def forward(self, graph, target_nodes):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch




        for layer in self.gat_layers:
            x = layer(x, edge_index)



        x = self.node_emb(x)
        g_cls = global_mean_pool(x, batch).unsqueeze(dim=1)
        #x = einops.rearrange(x, '(B L)D -> B L D', B=batch.max() + 1)
        node_tokens, node_mask = self.resize_graph_to_num_tokens(x, batch, target_nodes)
        return g_cls, node_tokens, node_mask

    # def forward(self, graph, target_nodes):
    #     x, edge_index, batch = graph.x, graph.edge_index, graph.batch
    #     boxes = graph.node_boxes  # [N, 4] or [N, 4, 2]
    #
    #     # 1. 图卷积编码
    #     for layer in self.gat_layers:
    #         x = layer(x, edge_index)
    #
    #     x = self.node_emb(x)  # [N, out_dim]
    #
    #     # 2. 面积计算与归一化（默认图像大小为 512×512）
    #     if boxes.ndim == 3:  # 旋转框 4点
    #         areas = self.compute_polygon_area(boxes)
    #     else:  # [x1, y1, x2, y2]
    #         areas = self.compute_box_area(boxes)
    #     areas = areas / (self.input_imgsz * self.input_imgsz)  # 归一化面积
    #
    #     # 3. 面积反比权重（越大越小）
    #     inv_area = 1.0 / (areas + 1e-6)
    #     batch_sum = torch.zeros(batch.max().item() + 1, device=inv_area.device).scatter_add(0, batch, inv_area)
    #     norm_weight = inv_area / (batch_sum[batch] + 1e-6)
    #
    #     # 4. 加权全局池化作为 g_cls token
    #     weighted_x = x * norm_weight.unsqueeze(-1)
    #     B, D = int(batch.max().item()) + 1, x.shape[1]
    #     g_cls = torch.zeros(B, D, device=x.device)
    #     g_cls = g_cls.scatter_add(0, batch.unsqueeze(-1).expand(-1, D), weighted_x)
    #     g_cls = g_cls.unsqueeze(1)  # [B, 1, D]
    #
    #
    #     # 5. 缩放图为 token 格式
    #     node_tokens, node_mask = self.resize_graph_to_num_tokens(x, batch, target_nodes)
    #
    #     return g_cls, node_tokens, node_mask
