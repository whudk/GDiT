# MIT License

# Copyright (c) [2023] [Anima-Lab]
from collections import OrderedDict
import torch
import numpy as np
from torch_geometric.data import Batch, Data
import matplotlib.pyplot as plt
import networkx as nx
from torch.nn.utils.rnn import pad_sequence
import cv2
def get_mask_ratio_fn(name='constant', ratio_scale=0.5, ratio_min=0.0):
    if name == 'cosine2':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 2 + ratio_min
    elif name == 'cosine3':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 3 + ratio_min
    elif name == 'cosine4':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 4 + ratio_min
    elif name == 'cosine5':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 5 + ratio_min
    elif name == 'cosine6':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 6 + ratio_min
    elif name == 'exp':
        return lambda x: (ratio_scale - ratio_min) * np.exp(-x * 7) + ratio_min
    elif name == 'linear':
        return lambda x: (ratio_scale - ratio_min) * x + ratio_min
    elif name == 'constant':
        return lambda x: ratio_scale
    else:
        raise ValueError('Unknown mask ratio function: {}'.format(name))
    

def get_one_hot(labels, num_classes=1000):
    one_hot = torch.zeros(labels.shape[0], num_classes, device=labels.device)
    one_hot.scatter_(1, labels.long().view(-1, 1), 1)
    return one_hot


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# ------------------------------------------------------------
# Training Helper Function

@torch.no_grad()
def update_ema(ema_model, model, decay=0.995):
    """
    EMA update of model parameters and buffers.
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if not param.requires_grad:
            continue
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    # 同步 BN 的 running_mean 和 running_var 等 buffer（保持 eval 表现一致）
    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.data.copy_(buf.data)


def unwrap_model(model):
    """
    Unwrap a model from any distributed or compiled wrappers. 
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return model

def extract_bboxes(seg, instance_map):
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
            label = seg[b, :, mask].mean(dim=-1)  # Take mean over the spatial dimension

            # Append the object information
            objects.append({
                "cls": label.item(),  # Convert to Python scalar
                "bbox": [min_x.item(), min_y.item(), max_x.item(), max_y.item()],
                "batch_id": b
            })
    bboxes = torch.stack([torch.tensor(obj['bbox']) for obj in objects], dim=0).to(seg.device)
    return bboxes
import  time
@torch.no_grad()
def dropout_nodes_in_graph(graph, instance_map, semantic_map, pe_embeddings, dropout_prob=0.1, fusion = 'concat'):
    """
    Randomly drop out nodes in a graph.

    Args:
        node_features (torch.Tensor): Node features tensor of shape [num_nodes, feature_dim].
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        batch (torch.Tensor): Batch indices of shape [num_nodes].
        dropout_prob (float): Probability of dropping out a node.

    Returns:
        torch.Tensor: Updated node features with dropped nodes.
        torch.Tensor: Updated edge indices with isolated edges removed.
        torch.Tensor: Updated batch indices (unchanged in this case).
    """

    #start_time = time.time()
    # if dropout_prob == 0.0:
    #     graph.x = torch.cat((graph.x, pe_embeddings),
    #                               dim=1) if fusion == 'concat' else graph.x + pe_embeddings
    #     return graph, pe_embeddings, instance_map, semantic_map

    node_features, edge_index, batch, node_boxes = graph.x, graph.edge_index, graph.batch, graph.node_boxxes
    device = node_features.device
    num_nodes = node_features.size(0)

    keep_mask = torch.rand(num_nodes, device=device) > dropout_prob
    keep_mask_unsq = keep_mask.unsqueeze(-1)

    # Node + PE fusion
    # if pe_embeddings is not None:
    #     if fusion == 'concat':
    #         node_features = torch.cat([node_features, pe_embeddings], dim=1)
    #     elif fusion.lower() == 'add':
    #         node_features = node_features + pe_embeddings

    node_features = node_features * keep_mask_unsq

    edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    # Accelerated dropped region construction
    B, H, W = instance_map.shape
    drop_indices = (~keep_mask).nonzero(as_tuple=False).squeeze(-1)
    drop_batch = batch[drop_indices]
    drop_instance_ids = drop_indices + 1  # assume instance_id = node_id + 1

    dropped_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
    for b in range(B):
        ids = drop_instance_ids[drop_batch == b]
        if ids.numel() > 0:
            dropped_mask[b] = (instance_map[b:b + 1] == ids[:, None, None]).any(0)

    instance_map[dropped_mask] = 0
    semantic_map[dropped_mask.unsqueeze(1).expand_as(semantic_map)] = 0
    node_features = torch.cat((node_features, pe_embeddings),dim=1) #"if fusion == 'concat' else node_features + pe_embeddings"
    droped_graph = Data(
        x=node_features,
        edge_index=edge_index,
        batch=batch,
        node_boxxes=node_boxes
    )

    # end_time = time.time()
    # print(f"[dropout_nodes_in_graph] Start time: {start_time:.6f} s")
    # print(f"[dropout_nodes_in_graph] End time:   {end_time:.6f} s")
    # print(f"[dropout_nodes_in_graph] Time elapsed: {end_time - start_time:.6f} seconds")

    return droped_graph, pe_embeddings, instance_map, semantic_map, keep_mask

# @torch.no_grad()
# def dropout_nodes_in_graph(graph, instance_map, semantic_map, pe_embeddings, dropout_prob=0.1, fusion='concat', area_thresh=0.001):
#     """
#     Drop nodes randomly and remove small-area nodes based on bounding box size.
#
#     Args:
#         graph: PyG Data object with attributes x, edge_index, batch, node_boxxes.
#         instance_map (Tensor): [B, H, W] instance map.
#         semantic_map (Tensor): [B, H, W] semantic map.
#         pe_embeddings (Tensor): [num_nodes, D]
#         dropout_prob (float): probability to randomly drop a node.
#         fusion (str): 'concat' or 'add'
#         area_thresh (float): area threshold relative to image (e.g., 0.001 means 0.1%)
#
#     Returns:
#         Updated graph, pe_embeddings, instance_map, semantic_map
#     """
#
#     node_features, edge_index, batch = graph.x, graph.edge_index, graph.batch
#     node_boxes = graph.node_boxxes  # [ num_nodes, 4, 2]
#
#     num_nodes = node_features.size(0)
#     device = node_features.device
#
#     # ========== 1. Compute object area (normalized by image size) ==========
#     xy = node_boxes  # [N, 4, 2]
#     x = xy[:, :, 0]
#     y = xy[:, :, 1]
#
#     # Shoelace formula to compute polygon area
#     area = 0.5 * torch.abs(
#         x[:, 0] * y[:, 1] + x[:, 1] * y[:, 2] + x[:, 2] * y[:, 3] + x[:, 3] * y[:, 0] -
#         y[:, 0] * x[:, 1] - y[:, 1] * x[:, 2] - y[:, 2] * x[:, 3] - y[:, 3] * x[:, 0]
#     )
#
#     # Normalize by image area
#     img_h, img_w = instance_map.shape[-2:]
#     area_norm = area / (img_w * img_h)
#     # Area mask
#     area_mask = area_norm > area_thresh
#     # ========== 2. Random dropout ==========
#     random_mask = torch.rand(num_nodes, device=device) > dropout_prob
#
#     # ========== 3. Combined mask ==========
#     keep_mask = area_mask & random_mask  # Only keep nodes that are big enough and not randomly dropped
#
#     keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
#
#     # ========== 4. Filter graph ==========
#     graph.x = node_features[keep_indices]
#     graph.batch = batch[keep_indices]
#     graph.node_boxxes = node_boxes[keep_indices]
#
#     # Update edge_index to only keep valid node indices
#     index_map = -torch.ones(num_nodes, dtype=torch.long, device=device)
#     index_map[keep_indices] = torch.arange(keep_indices.size(0), device=device)
#
#     src, tgt = edge_index
#     src_new, tgt_new = index_map[src], index_map[tgt]
#     valid_edge_mask = (src_new >= 0) & (tgt_new >= 0)
#
#     graph.edge_index = torch.stack([src_new[valid_edge_mask], tgt_new[valid_edge_mask]], dim=0)
#
#     # ========== 5. Update pe_embeddings ==========
#     pe_embeddings = pe_embeddings[keep_indices]
#
#     # ========== 6. Fusion ==========
#     if fusion == 'concat':
#         graph.x = torch.cat([graph.x, pe_embeddings], dim=-1)
#     else:
#         graph.x = graph.x + pe_embeddings
#
#     return graph, pe_embeddings, instance_map, semantic_map


@torch.no_grad()
def random_dropout_nodes(graph,   instance_map, semantic_map,dropout_prob=0.5):
    """
    Randomly drop out nodes in a graph.

    Args:
        node_features (torch.Tensor): Node features tensor of shape [num_nodes, feature_dim].
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        batch (torch.Tensor): Batch indices of shape [num_nodes].
        dropout_prob (float): Probability of dropping out a node.

    Returns:
        torch.Tensor: Updated node features with dropped nodes.
        torch.Tensor: Updated edge indices with isolated edges removed.
        torch.Tensor: Updated batch indices (unchanged in this case).
    """


    node_features, node_mask_features, edge_index, batch = graph.x, graph.node_masks, graph.edge_index, graph.batch


    assert node_features.shape == node_mask_features.shape

    node_features = torch.cat((node_features, node_mask_features), dim=1)

    num_nodes = node_features.size(0)

    # Randomly generate a mask for keeping or dropping nodes
    keep_mask = torch.rand(num_nodes, device=node_features.device) > dropout_prob


    #
    B = batch.max().item() + 1  # Number of batches


    node_features = node_features * keep_mask.unsqueeze(-1)

    edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    B = semantic_map.shape[0]
    # Iterate over batches to update the instance map
    for b in range(B):
        # Get node indices corresponding to batch `b`
        batch_node_indices = torch.nonzero((batch == b), as_tuple=True)[0]  # Node indices in this batch

        # Get the keep mask for nodes in this batch
        batch_keep_mask = ~keep_mask[batch_node_indices]
        batch_keep_mask = batch_keep_mask.nonzero(as_tuple=True)[0]
        unique_nodes = torch.unique(batch_keep_mask)
        for id in unique_nodes:
            mask = (instance_map[b, :, :] == (int(id) + 1))
            semantic_map[b, :, mask] = 0


    droped_graph = Data(
        x=node_features,
        edge_index=edge_index,
        batch=graph.batch
    )
    # Iterate over all nodes in this batch
    return droped_graph, semantic_map

    # droped_graph = Data(
    #     x = node_features,
    #     edge_index=edge_index,
    #     batch=graph.batch
    # )
    # # Iterate over all nodes in this batch
    # return droped_graph


@torch.no_grad()
def _contrust_graph( graph,instance_map, semantic_map, pe_embeddings,target_nodes = 96):


    node_l1, edge_index, batch, pe_embeds = graph.x, graph.edge_index, graph.batch




    node_l1 = torch.cat((node_l1, pe_embeddings), dim=-1)

    # Split the graph into batches
    batch_graphs = []
    unique_batches = torch.unique(batch)
    B = unique_batches.max() + 1
    pad_mask = torch.ones((B,target_nodes), device=node_l1.device)
    start_id = 0
    for b in unique_batches:
        batch_mask = batch == b
        node_features = node_l1[batch_mask]
        batch_edge_mask = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
        batch_edge_index = edge_index[:, batch_edge_mask]

        # Adjust `node_features` to have `target_nodes`
        num_nodes, feature_dim = node_features.shape
        if num_nodes < target_nodes:
            # Pad with zeros if fewer than target_nodes
            padding = torch.zeros((target_nodes - num_nodes, feature_dim), device=node_features.device)
            node_features = torch.cat([node_features, padding], dim=0)
            pad_mask[b, num_nodes:target_nodes] = 0
            batch_edge_index = batch_edge_index - start_id  #重新编号
        elif num_nodes >= target_nodes:
            node_features = node_features[:target_nodes]
            batch_edge_index = batch_edge_index - start_id
            edge_mask = (batch_edge_index[0] < target_nodes) & (batch_edge_index[1] < target_nodes)
            batch_edge_index = batch_edge_index[:, edge_mask]


            # # Truncate if more than target_nodes
            # # Randomly select target_nodes from num_nodes
            # selected_nodes = torch.randperm(num_nodes)[:target_nodes]
            # # Create a mask for the selected nodes
            # node_mask = torch.zeros(num_nodes, dtype=torch.float32, device=graph.x.device)
            # node_mask[selected_nodes] = 1
            #
            # # Keep only the selected nodes
            # node_features = node_features[selected_nodes]
            # # Update the edge_index to only include edges between the selected nodes
            # #
            #
            # edge_mask = (node_mask[batch_edge_index[0] - start_id] == 1) & (node_mask[batch_edge_index[1] - start_id] == 1)
            # batch_edge_index = batch_edge_index[:, edge_mask]


        start_id = start_id + num_nodes


        # Create the graph object for the current batch
        batch_graph = Data(
            x = node_features,
            edge_index=batch_edge_index,
            batch=torch.full((target_nodes,), b.item(), dtype=batch.dtype, device=batch.device)
        )
        batch_graphs.append(batch_graph)

    # Combine all batch graphs into a single batch
    adjusted_graph = Batch.from_data_list(batch_graphs)



    return adjusted_graph,instance_map,semantic_map, pad_mask


def collate_graph(batch):
    # Stack images
    images = torch.stack([item['image'] for item in batch], dim=0)  # Stack images into a tensor (N, C, H, W)
    img_records =  [item['img_records'] for item in batch]
    # Stack segmentation maps (make sure to have the same size for each image)
    cond_segs = torch.stack([item['cond_seg'] for item in batch],dim=0)  # Stack cond_segs

    # Stack vae_feats (assuming they have consistent dimensions for all items)
    vae_feats = torch.stack([item['vae_feats'] for item in batch], dim=0)
    instance_map = torch.stack([item['cond_inst'] for item in batch], dim=0)
    # Stack graphs into a batch
   # graphs = [item['cond_graph'] for item in batch]

    # graphs：保留为 list，里面可以是 Data 或 None
    graphs = [item['cond_graph'] for item in batch]

    return dict(
        image=images,
        cond_seg=cond_segs,
        cond_graph=graphs,  # 不再在这里 Batch.from_data_list
        cond_inst=instance_map,
        vae_feats=vae_feats,
        img_records=img_records,
    )
    # # Combine all graph data into a single batch (for graph neural networks)
    # graph_batch = Batch.from_data_list(graphs).cpu()
    #
    #
    #
    #
    # return dict(image=images, cond_seg=cond_segs, cond_graph=graph_batch, cond_inst = instance_map, vae_feats=vae_feats, img_records = img_records)
def m2t_collate(batch):
    # 解压batch
    imgs, labels, vae_feats,instance_map, kmeans_labels ,bboxes, img_records, data  = zip(*batch)
    #imgs, labels, sam_mask, img_records, node_feats, edge_indices = zip(*batch)

    # data = {}
    # data["captions"] = captions
    # data["l1_node"] = node_l1
    # data["l1_edge"] = l1_edge
    # data["l2_node"] = node_l2
    # data["l2_edge"] = l2_edge
    # data["l3_node"] = node_l3
    # data["l3_edge"] = l3_edge
    # data["l12_node"] = node_12
    # data["l12_edge"] = l12_edge
    # data["l23_node"] = node_23
    # data["l23_edge"] = l23_edge
    #data[""]
    graphs_hire = {}

    if "l1_node" and "l1_edge" in data[0].keys():
        if data[0]["l1_node"] is not None and data[0]["l1_edge"] is not None:
            node_feats = [d["l1_node"] for d in data]
            edge_indices = [d["l1_edge"] for d in data]
            node_feats = [nf if nf is not None else None for nf in node_feats]
            edge_indices = [ei if ei is not None else None for ei in edge_indices]
            graph_data_list = [Data(x=nf, edge_index=ei) if nf is not None else None for nf, ei in zip(node_feats, edge_indices)]
            graphs = Batch.from_data_list([g for g in graph_data_list if g is not None])
            graphs_hire["l1"] = graphs
        # batch_out["l1_node"] = torch.cat([d["l1_node"] for d in data], dim=0)
        # batch_out["l1_edge"] = torch.cat([d["l1_edge"] for d in data], dim=1)
    if "l2_node" and "l2_edge" in data[0].keys():
        #if data[0]["l2_node"] is not None and data[0]["l2_edge"] is not None:
            node_feats = [d["l2_node"] for d in data]
            edge_indices = [d["l2_edge"] for d in data]
            node_feats = [nf if nf is not None else None for nf in node_feats]
            edge_indices = [ei if ei is not None else None for ei in edge_indices]
            graph_data_list = [Data(x=nf, edge_index=ei) if nf is not None else None for nf, ei in
                               zip(node_feats, edge_indices)]

            g_list = [g for g in graph_data_list if g is not None]
            graphs = None
            if len(g_list) > 0:
                graphs = Batch.from_data_list([g for g in graph_data_list if g is not None])
            graphs_hire["l2"] = graphs
    if "l3_node" and "l3_edge" in data[0].keys():
        #if data[0]["l3_node"] is not None and data[0]["l3_edge"] is not None:
            node_feats = [d["l3_node"] for d in data]
            edge_indices = [d["l3_edge"] for d in data]
            node_feats = [nf if nf is not None else None for nf in node_feats]
            edge_indices = [ei if ei is not None else None for ei in edge_indices]
            graph_data_list = [Data(x=nf, edge_index=ei) if nf is not None else None for nf, ei in
                               zip(node_feats, edge_indices)]
            g_list = [g for g in graph_data_list if g is not None]
            graphs = None
            if len(g_list) > 0:
                graphs = Batch.from_data_list([g for g in graph_data_list if g is not None])
            graphs_hire["l3"] = graphs


    # 处理图像和标签
    imgs = torch.stack(
        [torch.from_numpy(np.array(img)).permute(2, 0, 1) if isinstance(img, np.ndarray) else img for img in imgs])

    labels = torch.stack([torch.from_numpy(lbl) if isinstance(lbl, np.ndarray) else lbl for lbl in labels if lbl is not None])
    instance_map = torch.stack(
        [torch.from_numpy(inst) if isinstance(inst, np.ndarray) else inst for inst in instance_map if instance_map is not None])
    vae_feats = torch.cat([torch.from_numpy(vae.squeeze(0)) if isinstance(vae, np.ndarray) else vae for vae in vae_feats if vae is not None],dim = 0)

    kmeans_labels = torch.stack(kmeans_labels, dim=0) if kmeans_labels is not None else None
    bboxes = torch.cat(bboxes, dim=0) if bboxes is not None else None
    # 返回组合的数据
    return imgs, labels, vae_feats, instance_map, bboxes, kmeans_labels, img_records, graphs_hire

@torch.no_grad()
def extract_rotated_bboxes_from_instmap(instance_map):


    instance_ids = np.unique(instance_map)
    bboxes = []
    for inst_id in instance_ids:
        if inst_id == 0:
            continue  # 跳过背景

        mask = (instance_map == inst_id).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)   # 4 个角点坐标
        box = np.int0(box)          # 转成整数


        bboxes.append(box)
    bboxes = torch.from_numpy(np.stack(bboxes, axis=0))
    return bboxes


@torch.no_grad()
def extract_bboxes(x, instance_map):
    objects = []
    B = instance_map.shape[0]
    if len(instance_map.shape) == 4:
        instance_map = instance_map.squeeze(1)

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
    bboxes = torch.stack([torch.tensor(obj['bbox']) for obj in objects], dim=0).to(x.device)
    return bboxes

import torch.nn as nn
@torch.no_grad()
def extract_masks_embedings(model , instance_map):

    objects = []
    B = instance_map.shape[0]
    if len(instance_map.shape) == 4:
        instance_map = instance_map.squeeze(1)

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


            # Append the object information
            objects.append({
                "mask": mask,
                "bbox": [min_x.item(), min_y.item(), max_x.item(), max_y.item()],
                "batch_id": b
            })


    masks  = torch.stack([torch.tensor(obj['mask']) for obj in objects], dim=0).float().unsqueeze(1).cuda()
    masks = nn.functional.interpolate(masks, size=(224, 224), mode="nearest").repeat(1, 3, 1, 1)
    masks_embed = model(masks)
    return masks_embed
def build_region_prompt(dataset_name, class_names):
    """拼成统一风格的区域文本。"""
    dataset_name = str(dataset_name)
    class_names = [c.strip() for c in class_names if c and c.strip()]
    class_names = list(dict.fromkeys(class_names))
    "'a satellite image from the [LEVIR-CD-B], containing Rangeland, Tree'"
    if len(class_names) == 0:
        return f"a satellite image from the [{dataset_name}] dataset."
    if len(class_names) == 1:
        return f"a satellite image from the [{dataset_name}] dataset containing {class_names[0]}"

    cls_body = ", ".join(class_names[:-1])
    cls_last = class_names[-1]
    return f"a satellite image from the [{dataset_name}] dataset, containing {cls_body} and {cls_last}"


def update_region_captions(region_captions, idx_src, x_cond, oem_dict):

    bg_id = 0


    # 统一成 [B,H,W]
    if x_cond.dim() == 4 and x_cond.size(1) == 1:
        seg_batch = x_cond[:, 0].long()      # [B,H,W]
    elif x_cond.dim() == 3:
        seg_batch = x_cond.long()
    else:
        raise ValueError(f"x_cond shape not supported: {x_cond.shape}")

    region_captions_out = list(region_captions)  # 拷一份

    for i in idx_src:
        seg = seg_batch[i]                      # [H,W]
        uniq_ids = torch.unique(seg).tolist()
        dataset_name = region_captions[i]
        class_names = []
        for cid in uniq_ids:
            cid = int(cid)
            if cid == bg_id:
                continue  # 跳过背景
            name = oem_dict.get(cid)
            if name is not None:
                class_names.append(name)

        prompt = build_region_prompt(dataset_name, class_names)
        region_captions_out[i] = prompt

    return region_captions_out