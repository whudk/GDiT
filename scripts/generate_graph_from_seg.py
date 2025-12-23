import sys

import torch
import glob
import numpy as np
import cv2
from diffusers.models import AutoencoderKL
import os
from  tqdm import tqdm
from semantic_to_captions import  semantic_to_instance_map,get_adjacency_matrix_from_instace_map, get_boxes_and_captions
from train_utils.helper import extract_bboxes, extract_masks_embedings, extract_rotated_bboxes_from_instmap

from torch_geometric.data import Data
import torch.nn as nn
import torchvision.models as models
from PIL import Image
oem_text = {
  0:"Background",
  1:"Bareland",
  2:"Rangeland",
  3:"Developed space",
  4:"Road",
  5:"Tree",
  6:"Water",
  7:"Agriculture land",
  8:"Building"
}
osm_map ={
    (160, 160, 160): 0,  # background - gray
    (0, 128, 0) : 1,  # park and green space - green
    (178, 34, 34) :2,  # buildings - firebrick
    (255, 165, 0) :3,  # residential - orange
    (128, 0, 128):4,  # industrial - purple
    (0, 0, 255) :5,  # transportation space - blue
    (0, 191, 255):6  # water - deep sky blue
}
osm_text ={
    0: "Background",
    1: "Green Space",  # park and green space - green
    2: "Building",  # buildings - firebrick
    3: "Residential",  # residential - orange
    4: "Industrial",  # industrial - purple
    5: "Transportation Space",  # transportation space - blue
    6: "Water"  # water - deep sky blue
}
def get_node_feats(model, captions):
    word_embeddings = dict()
    for key, val in captions.items():
        with torch.no_grad():
          word_embeddings[val] = model(["This node is " +  val  + "."]).cpu()
    return word_embeddings


class MaskEncoder(nn.Module):
    def __init__(self, input_channels=1, output_dim=768):
        super(MaskEncoder, self).__init__()
        # 使用预训练的ResNet模型来提取特征
        self.resnet = models.resnet18(pretrained=True)  # 你可以根据需求更改为其他ResNet模型
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        return self.resnet(x)
def apply_color_index(seg_map, color_to_index = None):
    """
    将 RGB 分割图中的颜色值映射为类别索引。

    参数：
        seg_map: numpy array，形状 (H, W, 3)，RGB 图像
        color_to_index: dict，{tuple(R, G, B): class_index}

    返回：
        label_map: numpy array，形状 (H, W)，每像素为类别编号（0~6）
    """
    if color_to_index is None:
        return seg_map
    h, w, _ = seg_map.shape
    flat_img = seg_map.reshape(-1, 3)
    unique_colors, inverse_indices = np.unique(flat_img, axis=0, return_inverse=True)

    # 建立颜色 → 类别映射表
    color_to_index_int = {tuple(map(int, k)): v for k, v in color_to_index.items()}

    # 每个 unique color 找对应 label，未匹配则为 0（background）
    label_array = np.array([
        color_to_index_int.get(tuple(c), 0)
        for c in unique_colors
    ], dtype=np.uint8)

    label_map = label_array[inverse_indices].reshape(h, w)
    return label_map


# Stack them to create edge indices
# Stack them to create edge indices

import torchvision.models as models
class FrozenResNetMaskEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # 用预训练的ResNet18
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后fc
        for p in self.feature_extractor.parameters():
            p.requires_grad = False  # 全部冻结
       # self.fc = nn.Linear(512, out_dim)  # 默认resnet18最后输出512维

    def forward(self, x):
        # x: (N, 3, 224, 224)
        feat = self.feature_extractor(x)  # (N, 512, 1, 1)
        feat = feat.flatten(1)            # (N, 512)
        return feat              # (N, out_dim)

def generate_graph(data_dir, outdir):
    img_paths = glob.glob(os.path.join(data_dir, "labels/*.tif"))
    # img_paths = glob.glob(os.path.join(data_dir, "*", "source\*.png"))
    # out_inst_dir = os.path.join(os.path.dirname(outdir), "insts_256")
    #img_paths = glob.glob(os.path.join(data_dir,  "*.tif"))
    out_inst_dir = os.path.join(os.path.dirname(outdir), "insts_256")
    os.makedirs(out_inst_dir, exist_ok=True)


    from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder
    clip_model = FrozenCLIPTextEmbedder().cuda().eval()
    node_dict = get_node_feats(clip_model, osm_text)


    # 加载 ResNet18 模型并冻结其参数
    model = FrozenResNetMaskEncoder().eval().cuda()
    for param in model.parameters():
        param.requires_grad = False  # 冻结所有层的参数

    for i, image_path in tqdm(enumerate(img_paths)):
        #image_path = r"F:\geosynth_dataset\train\images\Adams\source\osm_tile_18_54575_99427_label.png"
        #image_path = r"F:\geosynth_dataset\train\images\Adams\source\osm_tile_18_54576_99396_label.png"
        # if i<=14060:
        #     continue
        seg_map = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # seg_map = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)[:, :, :3]
        # seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)
        # seg_map = apply_color_index(seg_map, osm_map)



        instance_map, boxes_info = get_boxes_and_captions(seg_map, oem_text)
        if len(boxes_info) == 0:
            continue
        # 获取每个节点的旋转框
        node_boxxes = [b["box"] for b in boxes_info]
        node_boxxes = torch.from_numpy(np.stack(node_boxxes, axis=0))
        # 获取每个节点的语义描述
        node_captions = [b["caption"] for b in boxes_info]


        #node_boxxes = []

        #instance_map, node_captions = semantic_to_instance_map(seg_map, osm_text)

        #node_boxxes = extract_rotated_bboxes_from_instmap(instance_map)



        edges = get_adjacency_matrix_from_instace_map(instance_map)


        node_l1 = []
        for caption in node_captions:
            #node_l1.append(node_dict[caption])  # 获取每个节点的特征
            with torch.no_grad():
                node_l1.append(clip_model([caption]).cpu())
        if len(node_l1) == 0:
            print(image_path)
            continue
        # 确保将所有节点特征拼接成一个张量
        node_l1 = torch.cat(node_l1, dim=0)  # 如果 node_dict[caption] 是一个张量，这里应该能正确拼接

        # 你的边数据
        l1_edge = torch.from_numpy(edges)  # 应该是一个形状为 [2, num_edges] 的张量

        # 创建图数据对象
        # node_boxxes = extract_bboxes()
        # node_boxxes = extract_bboxes(torch.from_numpy(seg_map).float().unsqueeze(0).unsqueeze(0),
        #                              torch.from_numpy(instance_map).unsqueeze(0))





        node_masks_embed, shape_feats = extract_masks_embedings(model,torch.from_numpy(instance_map).unsqueeze(0))



        #node_embeddings =

        assert node_l1.shape[0] == node_boxxes.shape[0] == len(node_captions)
        graph = Data(
            x=node_l1,  # 节点特征
            node_captions=node_captions,
            node_masks = node_masks_embed.cpu(),
            physical_feats =shape_feats.cpu(),
            node_boxxes = node_boxxes,
            edge_index = l1_edge,  # 边的索引
            batch = torch.full((node_l1.shape[0],), 0, dtype=torch.long)
        )

        save_name = os.path.basename(image_path).replace(".tif", ".pth")
        np.save(os.path.join(out_inst_dir, save_name.replace(".pth", ".npy")), instance_map)
        save_name = os.path.basename(image_path).replace(".tif", ".pth")
        save_path = os.path.join(outdir, save_name)
        torch.save(graph, save_path)


import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Generate graphs for dataset splits")

    # 推荐：给 root 自动推断 train/val
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=r'Dataset root containing splits, e.g. F:\data\OpenEarthMap\Size_256'
    )

    # 可选：分别指定 split 目录（优先于 data_root）
    parser.add_argument("--train_dir", type=str, default=None, help="Path to train split directory")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to val split directory")

    # 输出子目录名（默认在 split 目录下创建 graphs）
    parser.add_argument("--out_name", type=str, default="graphs", help="Output folder name under each split")

    # 需要处理哪些 split
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process, e.g. train val")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_dir = args.train_dir
    val_dir = args.val_dir

    if train_dir is None or val_dir is None:
        if args.data_root is None:
            raise ValueError("Please provide either --data_root or both --train_dir and --val_dir.")
        if train_dir is None:
            train_dir = os.path.join(args.data_root, "train")
        if val_dir is None:
            val_dir = os.path.join(args.data_root, "val")

    for split in args.splits:
        if split.lower() == "train":
            data_dir = train_dir
        elif split.lower() == "val":
            data_dir = val_dir
        else:
            if args.data_root is None:
                raise ValueError(f"Unknown split '{split}' without --data_root to resolve it.")
            data_dir = os.path.join(args.data_root, split)

        out_dir = os.path.join(data_dir, args.out_name)
        os.makedirs(out_dir, exist_ok=True)

        generate_graph(data_dir, out_dir)
        print(f"process {split} done -> {out_dir}")
