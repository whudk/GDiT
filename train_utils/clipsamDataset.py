from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import sys

import cv2
import torch

import numpy as np
import random
from torch.utils import data
import math
from tqdm import tqdm
from xml.dom.minidom import parse
from scripts.cluster_semanticmap import extract_features,load_kmeans_model
from pathlib import Path

OEM_dict = {
'0':'Background',
'1':'Bareland',
'2':'Rangeland',
'3':'Developed space',
'4':'Road',
'5':'Tree',
'6':'Water',
'7':'Agriculture land',
'8':'Building',
}

import torch.nn.functional as F
from skimage import measure

from pycocotools import mask as maskUtils
from PIL import Image
import json

#from scripts.merge_mask_to_one import merge_all_mask_to_one_RoPE
from pycocotools import mask as mask_utils

import networkx as nx
import matplotlib.pyplot as plt

import os
import rasterio
from rasterio.windows import Window
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square
from models.clip import  clip
import torch
import clip
from torch_geometric.data import Data
import torch.nn.functional as F



class ToLabel(object):
    def __call__(self, inputs):
        return torch.from_numpy(np.array(inputs)).long()


class ReLabel(object):
    """
      255 indicate the background, relabel 255 to some value.
    """

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        assert isinstance(inputs, torch.LongTensor), 'tensor needs to be LongTensor'

        inputs[inputs == self.olabel] = self.nlabel
        return inputs
import torchvision.transforms.functional as TF
class SyncGeometricTransform:
    def __init__(self,  hflip_p=0.5):

        self.hflip_p = hflip_p

    def __call__(self, img, instance_label, label):

        hflip = random.random() < self.hflip_p

        # 对 img 和 label 同步应用变换
        if hflip:
            img = TF.hflip(img)
            instance_label = TF.hflip(instance_label)
            label = TF.hflip(label)


        return img, instance_label.squeeze(0),label

# 使用示例



# class CD_dataset(torch.utils.data.Dataset):
#
#     def __init__(self, json_file,
#                  transform=None,
#                  label_transform = None,
#                  split="train",
#                  transxml = None,
#                  **kwargs):
#
#
#
#
#
#         self.transform = transform
#
#         self.label_transform = label_transform
#
#
#         # refine label value
#
#         self.dict = None
#         if transxml is not None:
#             if os.path.exists(transxml):
#                 _, self.dict = self.readtransxml(self.transxml)
#
#         with open(json_file, 'r', encoding="utf-8") as file:
#             self.annotations = json.load(file)
#
#
#
#
#         self.images = self.annotations['images']
#         if "categories" in self.annotations.keys():
#             self.categories = self.annotations["categories"]
#         if isinstance(self.annotations, list):
#             self.anns = {ann['image_id']: [] for ann in self.annotations['annotations']}
#             for ann in self.annotations['annotations']:
#                 bbox = ann['bbox']
#                 segmentation = ann["segmentation"]
#                 x_min, y_min, width, height = bbox
#                 x_max = x_min + width
#                 y_max = y_min + height
#                 center_coor = [(x_min + x_max) / 2, (y_min + y_max) / 2]  # Calculate center
#                 self.anns[ann['image_id']].append(
#                     {'bbox': [x_min, y_min, x_max, y_max], 'center_coor': center_coor, 'segmentation': segmentation,
#                      'label': self.find_id_by_name(ann['category_id'], self.categories),
#                      'category_id': ann['category_id']})
#         else:
#             self.anns = self.annotations["annotations"]
#
#         self.num_sams = 100
#
#
#         self.sam_old, self.sam_new = None,None
#         if "sam_old" in kwargs:
#             self.sam_old = kwargs["sam_old"]
#
#         if "sam_new" in kwargs:
#             self.sam_new = kwargs["sam_new"]
#         # indices = np.random.choice(range(len(self.images)), 10, replace=True).tolist()
#         # self.images = [self.images[i] for  i in indices]
#
#
#
#     def random_select(self, n,  type = "label"):
#         labels = []
#         oldimgs, newimgs = [], []
#
#
#         indices = np.random.choice(range(len(self.images)), n, replace=True).tolist()
#         for idx in indices:
#
#             label_path = self.images[idx]["labelpath"].replace("\\", "/")
#
#             oldimage_path = self.images[idx]["oldpath"].replace("\\", "/")
#             newimage_path = self.images[idx]["newpath"].replace("\\", "/")
#
#             # oldimage_path = r"F:\data\02_1M2M_BHYB\202202\全要素\pre\pre_9.tif"
#             # newimage_path = r"F:\data\02_1M2M_BHYB\202202\全要素\cur\cur_9.tif"
#             # label_path = r"F:\data\02_1M2M_BHYB\202202\全要素\curbhlx\curbhlx_9.tif"
#
#             label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
#
#
#             label[label > 0] = 1
#
#
#
#             oldimg = cv2.imdecode(np.fromfile(oldimage_path, dtype=np.uint8), -1)
#             newimg = cv2.imdecode(np.fromfile(newimage_path, dtype=np.uint8), -1)
#
#             oldimg = cv2.cvtColor(oldimg, cv2.COLOR_BGR2RGB)
#             newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
#
#             oldimg = oldimg.copy()
#             newimg = newimg.copy()
#             label = label.copy()
#             oldimg_zero = ((oldimg[:, :, 0] == 0) & (oldimg[:, :, 1] == 0) & (oldimg[:, :, 2] == 0))  | ((oldimg[:, :, 0] == 1) & (oldimg[:, :, 1] == 1) & (oldimg[:, :, 2] == 1))
#             newimg_zero = ((newimg[:, :, 0] == 0) & (newimg[:, :, 1] == 0) & (newimg[:, :, 2] == 0))  | ((oldimg[:, :, 0] == 1) & (oldimg[:, :, 1] == 1) & (oldimg[:, :, 2] == 1))
#             maskzero = oldimg_zero | newimg_zero
#             oldimg[maskzero, :] = 0
#             newimg[maskzero, :] = 0
#             label[maskzero] = 0
#
#             oldimg = Image.fromarray(oldimg)
#             newimg = Image.fromarray(newimg)
#             label = torch.from_numpy(label).unsqueeze(0)
#             if self.label_transform is not None:
#                 label = self.label_transform(label)
#
#             if self.transform is not None:
#                 oldimg = self.transform(oldimg).unsqueeze(0)
#                 newimg = self.transform(newimg).unsqueeze(0)
#             oldimgs.append(oldimg)
#             newimgs.append(newimg)
#             labels.append(label)
#         if type == "label":
#             return labels
#         elif type == "pre":
#             return oldimgs
#         elif type == "cur":
#             return  newimgs
#         else:
#             return labels, oldimgs, newimgs
#
#     # def random_select_samples(self, n, type = "label"):
#     #     '''
#     #     type = "all", "pre", "cur", or "label"
#     #     '''
#     #     labels = []
#     #     oldimgs, newimgs = [], []
#     #
#     #     indices = np.random.choice(range(len(self.images)), n, replace=True).tolist()
#     #
#     #
#     #
#     #     for idx in indices:
#     #
#     #         label_path = self.images[idx]["labelpath"].replace("\\", "/")
#     #
#     #         oldimage_path = self.images[idx]["oldpath"].replace("\\", "/")
#     #         newimage_path = self.images[idx]["newpath"].replace("\\", "/")
#     #
#     #         # oldimage_path = r"F:\data\02_1M2M_BHYB\202202\全要素\pre\pre_9.tif"
#     #         # newimage_path = r"F:\data\02_1M2M_BHYB\202202\全要素\cur\cur_9.tif"
#     #         # label_path = r"F:\data\02_1M2M_BHYB\202202\全要素\curbhlx\curbhlx_9.tif"
#     #
#     #         label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
#     #
#     #         label[label > 0] = 1
#     #
#     #         oldimg = cv2.imdecode(np.fromfile(oldimage_path, dtype=np.uint8), -1)
#     #         newimg = cv2.imdecode(np.fromfile(newimage_path, dtype=np.uint8), -1)
#     #
#     #         oldimg = cv2.cvtColor(oldimg, cv2.COLOR_BGR2RGB)
#     #         newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
#     #
#     #         oldimg = oldimg.copy()
#     #         newimg = newimg.copy()
#     #         label = label.copy()
#     #         oldimg_zero = ((oldimg[:, :, 0] == 0) & (oldimg[:, :, 1] == 0) & (oldimg[:, :, 2] == 0)) | (
#     #                     (oldimg[:, :, 0] == 1) & (oldimg[:, :, 1] == 1) & (oldimg[:, :, 2] == 1))
#     #         newimg_zero = ((newimg[:, :, 0] == 0) & (newimg[:, :, 1] == 0) & (newimg[:, :, 2] == 0)) | (
#     #                     (oldimg[:, :, 0] == 1) & (oldimg[:, :, 1] == 1) & (oldimg[:, :, 2] == 1))
#     #         maskzero = oldimg_zero | newimg_zero
#     #         oldimg[maskzero, :] = 0
#     #         newimg[maskzero, :] = 0
#     #         label[maskzero] = 0
#     #
#     #         oldimg = Image.fromarray(oldimg)
#     #         newimg = Image.fromarray(newimg)
#     #         label = torch.from_numpy(label).unsqueeze(0)
#     #         if self.label_transform is not None:
#     #             label = self.label_transform(label)
#     #
#     #         if self.transform is not None:
#     #             oldimg = self.transform(oldimg).unsqueeze(0)
#     #             newimg = self.transform(newimg).unsqueeze(0)
#     #         oldimgs.append(oldimg)
#     #         newimgs.append(newimg)
#     #         labels.append(label)
#     #
#     #     return labels, oldimgs, newimgs
#
#
#     def find_id_by_name(self, name, categories):
#         for category in categories:
#             if category['name'] == name:
#                 return category['id']
#         return None  # Return None if no match is found
#
#     def find_boxes_and_centers(self, target):
#
#         point_coords = []
#         point_labels = []
#         bboxes = []
#
#         label, N  = measure.label(target, 0, True,2)
#         props = measure.regionprops(label)
#
#         for n in range(1, N+1):
#             point_label = int(np.unique(target[label == n]))
#             point_coords.append(props[n-1].centroid)
#             point_labels.append(point_label)
#             bboxes.append(props[n-1].bbox)
#         return point_coords,point_labels,bboxes
#     def checkpaths(self, lines, shuffle=True, clampN=False):
#         self.ids = []
#         for id, line in tqdm(enumerate(lines)):
#             paths = line.rstrip().split()
#             exist_flag = True
#             for path in paths:
#                 if not os.path.exists(path):
#                     print("{} is not existed".format(path))
#                     # print("check line:{} in {}".format(id+1,f.__dir__))
#                     exist_flag = False
#                     break
#             if exist_flag:
#                 self.ids.append(line)
#         if shuffle:
#             random.shuffle(self.ids)
#         if clampN:
#             self.ids = self.ids[:1000]
#         return self.ids
#     def get_classnames(self):
#         return  self.class_names
#     def readtransxml(self,transxml):
#         path_transxml = transxml
#         if not os.path.exists(path_transxml):
#             print('Error:{} is not existed.'.format(path_transxml))
#         transBM = parse(path_transxml)
#         root = transBM.documentElement
#         all_codes = root.getElementsByTagName('BM')
#         all_dict = {}
#         num_class = 0
#         for node in all_codes:
#             class_geoid_name = node.attributes['key'].value
#             class_id = node.attributes['val'].value
#             all_dict[int(class_geoid_name)] = int(class_id)
#             if int(class_id) > num_class:
#                 num_class = num_class + 1
#         return num_class,all_dict
#
#     def random_select_points(self, mask, N = 20):
#         y_coords, x_coords = np.where(mask == 1)
#
#         # 检查是否有足够的点
#         if len(y_coords) < N:
#             indices = np.random.choice(range(len(y_coords)), N, replace=True)
#             selected_coords = [(y_coords[i], x_coords[i]) for i in indices]
#         else:
#             # 从这些坐标中随机选择 N 个
#             indices = np.random.choice(range(len(y_coords)), N, replace=False)
#             selected_coords = [(y_coords[i], x_coords[i]) for i in indices]
#
#             # 打印所选点的坐标
#             #print("选中的点的坐标:", selected_coords)
#         return  selected_coords
#
#     def prepare_masks(self, masks, max_masks = 200, pad_value = 0):
#
#         """
#             使用Numpy对分割mask进行填充，使得所有样本的mask数量一致。
#
#             Args:
#                 masks (list of numpy.ndarray): 每个元素是一个样本的mask集合，形状为[num_masks, height, width]。
#                 max_masks (int): 所有样本中mask的最大数量。
#                 pad_value (int): 用于填充的值。
#
#             Returns:
#                 numpy.ndarray: 填充后的mask数组，形状为[batch_size, max_masks, height, width]。
#             """
#         padded_masks = []
#         num_masks = masks.shape[0]
#         #for mask in masks:
#
#         if num_masks < max_masks:
#             padding_size = max_masks - num_masks
#             # 注意mask.shape[1:]将提供高度和宽度
#             padding = np.full((padding_size, *masks.shape[1:]), pad_value, dtype=masks.dtype)
#             padded_mask = np.concatenate([masks, padding], axis=0)
#         else:
#             padded_mask = masks[:max_masks]
#             # 这里使用[np.newaxis]或者None增加一个新的批次维度
#
#         # 使用np.concatenate沿着新的批次维度连接所有的padded masks
#         return padded_mask #
#
#
#     def decode_masks(self, ann, key):
#         masks = []
#         if key in ann:
#             for obj in ann[key]:
#                 if len(masks) < self.num_sams:
#                     rle = obj["segmentation"]
#                     mask = maskUtils.decode(rle)
#                     masks.append(mask)
#                 else:
#                     break
#         if masks:
#             masks = np.dstack(masks).transpose(2, 0, 1)
#             masks = self.prepare_masks(masks, max_masks= self.num_sams)
#
#             masks = torch.from_numpy(masks)
#         else:
#             masks = torch.tensor([])  # 或其他合适的默认值
#         return masks
#     def __getitem__(self, idx):
#
#         oldimage_path = self.images[idx]["oldpath"].replace("\\", "/")
#         newimage_path = self.images[idx]["newpath"].replace("\\", "/")
#         label_path = self.images[idx]["labelpath"].replace("\\", "/")
#         # oldimage_path = oldimage_path.replace("/home/dk/","F:").replace("\\", "/")
#         # newimage_path = newimage_path.replace("/home/dk/","F:").replace("\\", "/")
#         # label_path = label_path.replace("/home/dk/", "F:").replace("\\", "/")
#
#         # oldimage_path = oldimage_path.replace("/home/dk/", "F:/")
#         # newimage_path = newimage_path.replace("/home/dk/", "F:/")
#         # label_path = label_path.replace("/home/dk/", "F:/")
#
#         oldimg  = cv2.imdecode(np.fromfile(oldimage_path,dtype=np.uint8),-1)
#         newimg = cv2.imdecode(np.fromfile(newimage_path,dtype=np.uint8),-1)
#
#         oldimg = cv2.cvtColor(oldimg, cv2.COLOR_BGR2RGB)
#         newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
#
#         H,W,C = oldimg.shape
#
#         if not os.path.exists(label_path):
#             label = None
#             print.error('cannot find {}'.format(label_path))
#         else:
#             label =  cv2.imdecode(np.fromfile(label_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
#             if self.dict is not None:
#                 label_map = np.vectorize(self.dict.get)(label)
#                 if None in label_map:
#                     pix_val = np.unique(label)
#                     mix_index = []
#                     for pix in pix_val:
#                         if str(pix_val) not in self.dict.keys():
#                             mix_index.append(pix)
#                     assert len(mix_index) == 0,'{}\n cannot find {} in transxml.'.format(label_path,mix_index)
#                 else:
#                     label = label_map
#
#
#
#         oldimg = oldimg.copy()
#         newimg = newimg.copy()
#         label = label.copy()
#         oldimg_zero = (oldimg[:, :, 0] == 0) & (oldimg[:, :, 1] == 0) & (oldimg[:, :, 2] == 0)
#         newimg_zero = (newimg[:, :, 0] == 0) & (newimg[:, :, 1] == 0) & (newimg[:, :, 2] == 0)
#         maskzero = oldimg_zero | newimg_zero
#         oldimg[:, :, 0][maskzero] = 0
#         oldimg[:, :, 1][maskzero] = 0
#         oldimg[:, :, 2][maskzero] = 0
#         newimg[:, :, 0][maskzero] = 0
#         newimg[:, :, 1][maskzero] = 0
#         newimg[:, :, 2][maskzero] = 0
#
#         label[label > 0] = 1
#         label[maskzero] = 0
#         img_records = {}
#
#         # if self.aug_transform is not None:
#         #     oldimg, newimg, label = self.aug_transform(oldimg, newimg, labelmap=label)
#
#         oldimg = Image.fromarray(oldimg)
#         newimg = Image.fromarray(newimg)
#         label = torch.from_numpy(label)
#         if self.transform is not None:
#             oldimg = self.transform(oldimg)
#             newimg = self.transform(newimg)
#         if self.label_transform is not None:
#             label = self.label_transform(label)
#
#         if isinstance(oldimg, np.ndarray):
#             oldimg = torch.from_numpy(oldimg).permute(2, 0, 1)
#
#         if isinstance(newimg, np.ndarray):
#             newimg = torch.from_numpy(newimg).permute(2, 0, 1)
#             # mask_patch = torch.from_numpy(mask_patch)
#             # print(torch.min(labelimg),torch.max(labelimg))
#
#         #img = torch.cat([oldimg, newimg], dim=0)
#
#
#         # ann = self.anns.get(str(self.images[idx]["id"]))
#         #
#         # mask_old = self.decode_masks(ann[-2], "sam_old")
#         # mask_new = self.decode_masks(ann[-1], "sam_new")
#         # mask_old, mask_new = [], []
#         # if "sam_old" in ann[-2].keys():
#         #     sam_old =  ann[-2]["sam_old"]
#         #     for obj in sam_old:
#         #         rle = obj["segmentation"]
#         #         mask = maskUtils.decode(rle )
#         #         mask_old.append(mask)
#         #     mask_old = np.dstack(mask_old).transpose(2,0,1)
#         #     mask_old = torch.from_numpy(mask_old)
#         #
#         # if "sam_new" in ann[-1].keys():
#         #     sam_new = ann[-1]["sam_new"]
#         #     for obj in sam_new:
#         #         rle = obj["segmentation"]
#         #         mask = maskUtils.decode(rle)
#         #         mask_new.append(mask)
#         #     mask_new = np.dstack(mask_new).transpose(2, 0, 1)
#         #     mask_new = torch.from_numpy(mask_new)
#
#
#
#
#         img_records["old_path"] = oldimage_path
#         img_records["new_path"] = newimage_path
#         img_records["label_path"] = label_path
#
#
#         #set sam_old and sam_new
#         if self.sam_old is not None:
#             name = os.path.basename(oldimage_path).split('.')[0]
#             file = os.path.join(self.sam_old, f'{name}.npy')
#             if os.path.exists(file):
#                 img = torch.from_numpy(np.load(file))
#                 img_records["sam_old"] = img
#             else:
#                 img_records["sam_old"] = torch.zeros_like(label)
#         if self.sam_new is not None:
#             name = os.path.basename(newimage_path).split('.')[0]
#             file = os.path.join(self.sam_new, f'{name}.npy')
#             if os.path.exists(file):
#                 img = torch.from_numpy(np.load(file))
#                 img_records["sam_new"] = img
#             else:
#                 img_records["sam_new"] = torch.zeros_like(label)
#
#
#         #img_records["mask_inputs"] = label
#
#
#
#         return oldimg, newimg, label, img_records
#     def __len__(self):
#         return len(self.images)




class GraphConstructor:
    def __init__(self, clip_model, cnn_model, device):
        self.clip_model = clip_model
        self.cnn_model = cnn_model
        self.device = device
        self.OEM_dict = {
            '0': 'Background',
            '1': 'Bareland',
            '2': 'Rangeland',
            '3': 'Developed space',
            '4': 'Road',
            '5': 'Tree',
            '6': 'Water',
            '7': 'Agriculture land',
            '8': 'Building',
        }
        self.class_descriptions = list(self.OEM_dict.values())
        self.class_tokens = clip.tokenize(self.class_descriptions).to(self.device)
        self.text_features = self.clip_model.encode_text(self.class_tokens)

    def compute_node_and_edges(self, semantic_map, image):
        # Step 1: Convert semantic map to instance map
        instance_map, node_classes = self._create_instance_map(semantic_map)

        # Step 2: Compute node features (CLIP text features and visual features)
        node_features = self._compute_node_features(instance_map, node_classes, image)

        # Step 3: Construct the graph (define edges based on adjacency or other criteria)
        edge_index = self._construct_edges(instance_map)

        # Create and return graph data structure
        data = Data(x=node_features, edge_index=edge_index)
        return data

    def _create_instance_map(self, semantic_map):
        # Label each connected region in the semantic map
        instance_map = label(semantic_map)
        node_classes = [self.OEM_dict[str(int(c))] for c in np.unique(semantic_map)]
        return instance_map, node_classes

    def _compute_node_features(self, instance_map, node_classes, image):
        # Resize image for CNN feature extraction
        image_input = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

        # Step 2.1: Extract CNN visual features for the entire image
        with torch.no_grad():
            cnn_feature_map = self.cnn_model(image_input)

        # Step 2.2: Compute features for each node (region)
        node_features = []
        for class_id, class_name in enumerate(node_classes):
            # Find all pixels in instance_map belonging to this class
            mask = (instance_map == class_id).astype(np.uint8)

            # Calculate the proportion of each pixel for the class
            class_pixel_count = mask.sum()
            if class_pixel_count == 0:
                continue

            # Use CLIP to get text features for the class name
            text_feature = self.text_features[class_id]

            # Mask and pool visual features from CNN feature map
            visual_feature = self._pool_visual_features(cnn_feature_map, mask)

            # Combine text and visual features
            combined_feature = torch.cat([text_feature, visual_feature])
            node_features.append(combined_feature)

        return torch.stack(node_features)

    def _pool_visual_features(self, cnn_feature_map, mask):
        # Upsample the mask to match cnn_feature_map dimensions
        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(self.device)
        mask_resized = F.interpolate(mask_tensor, size=cnn_feature_map.shape[2:], mode='bilinear')

        # Apply mask to the CNN feature map and compute the average (weighted by mask)
        masked_features = cnn_feature_map * mask_resized
        visual_feature = masked_features.sum(dim=(2, 3)) / (mask_resized.sum() + 1e-5)
        return visual_feature.squeeze()

    def _construct_edges(self, instance_map):
        # Define edges between adjacent nodes based on instance_map
        # Placeholder example: fully connected graph for simplicity
        node_indices = np.unique(instance_map)
        edge_index = []

        for i, node1 in enumerate(node_indices):
            for j, node2 in enumerate(node_indices):
                if i != j:  # No self-loops
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index).t().contiguous()
        return edge_index.to(self.device)


class M2I_dataset(torch.utils.data.Dataset):

    def __init__(self,
                 img_dir,
                 lbl_dir,
                 clip_model,
                 transform=None,
                 graph_dir  = None,
                 vae_dir = None,
                 instance_dir = None,
                 label_transform = None,
                 transxml = None,
                 with_graph = False,
                 kmeans_path = None,
                 texts = None,
                 #sam_dir = None,
                 device = 'cuda',
                 **kwargs):



        #read
        self.with_graph = with_graph
       # self.with_sam  = True if sam_dir is not None else False


        self.ids = self.find_matching_paths(img_dir,lbl_dir,vae_dir, instance_dir)
        self.model = clip_model
        self.kmeans_model = None
        if kmeans_path is not None:
            self.kmeans_model = load_kmeans_model(kmeans_path)


        if self.with_graph:
            assert os.path.exists(graph_dir)
            node_feats_folder = os.path.join(graph_dir, 'node_feats')
            edge_feats_folder = os.path.join(graph_dir, 'edge_indexs')

            self.nodes_l1, self.nodes_l2, self.nodes_l3 = [], [], []
            self.edges_l1, self.edges_l2, self.edges_l3 = [], [], []
            # self.nodes_l1 = glob.glob(os.path.join(node_feats_folder, "*_l1.npy"))
            # self.edges_l1 = glob.glob(os.path.join(edge_feats_folder, "*_l1_edge.npy"))
            # self.nodes_l2 = glob.glob(os.path.join(node_feats_folder, "*_l2.npy"))
            # self.edges_l2 = glob.glob(os.path.join(edge_feats_folder, "*_l2_edge.npy"))
            # self.nodes_l3 = glob.glob(os.path.join(node_feats_folder, "*_l3.npy"))
            # self.edges_l3 = glob.glob(os.path.join(edge_feats_folder, "*_l3_edge.npy"))



            # priors_dict = {os.path.basename(path): path for path in priors}
            # self.node = []
            # self.edge_index = []
            for s in self.ids:
                name = os.path.basename(s.split(" ")[0]).split('.')[0]
                node_name_l1 = name + "_l1.npy"
                edge_name_l1 = name + "_l1_edge.npy"
                node_name_l2 = name + "_l2.npy"
                edge_name_l2 = name + "_l2_edge.npy"
                node_name_l3 = name + "_l3.npy"
                edge_name_l3 = name + "_l3_edge.npy"
                if os.path.exists(os.path.join(node_feats_folder, node_name_l1)):
                    self.nodes_l1.append(os.path.join(node_feats_folder, node_name_l1))
                    self.edges_l1.append(os.path.join(edge_feats_folder, edge_name_l1))
                if os.path.exists(os.path.join(node_feats_folder, node_name_l2)):
                    self.nodes_l2.append(os.path.join(node_feats_folder, node_name_l2))
                    self.edges_l2.append(os.path.join(edge_feats_folder, edge_name_l2))
                if os.path.exists(os.path.join(node_feats_folder, node_name_l3)):
                    self.nodes_l3.append(os.path.join(node_feats_folder, node_name_l3))
                    self.edges_l3.append(os.path.join(edge_feats_folder, edge_name_l3))
            if len(self.nodes_l2) == 0:
                self.nodes_l2, self.edges_l2 = None, None
            if len(self.nodes_l3) == 0:
                self.nodes_l3, self.edges_l3 = None, None
        else:
            #self.node, self.edge_index = None, None
            self.nodes_l1, self.nodes_l2, self.nodes_l3 = None, None, None
            self.edges_l1, self.edges_l2, self.edges_l3 = None, None, None

        # if self.with_graph:
        #     assert  os.path.exists(graph_folder)
        #     priors = glob.glob(os.path.join(graph_folder, "*.npy"))
        #     priors_dict =  {os.path.basename(path): path for path in priors}
        #     self.node = [ ]
        #     self.edge_index = [ ]
        #     for s in self.ids:
        #         name = os.path.basename(s.split(" ")[0]).split('.')[0]
        #         node_name = name + "_node_features.npy"
        #         edge_name = name + "_edge_index.npy"
        #         self.node.append(priors_dict[node_name])
        #         self.edge_index.append(priors_dict[edge_name])
        # else:
        #     self.node, self.edge_index = None, None

        self.transform = transform


        self.label_transform = label_transform

        # refine label value

        self.regions = self.extract_city_names_from_dir(img_dir)

        self.dict = None
        if transxml is not None:
            if os.path.exists(transxml):
                _, self.dict = self.readtransxml(self.transxml)

    def get_regions(self):
        return  self.regions
    def extract_city_names_from_dir(self, dir_path):
        """
        遍历目录，提取影像文件中的城市名称。

        Args:
            dir_path (str): 影像文件所在的目录路径。

        Returns:
            list: 提取的城市名称列表。
        """
        city_names = set()  # 使用集合避免重复城市名称

        # 遍历目录
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".tif"):  # 只处理.tif文件
                    # 提取城市名称
                    city_name = file.split("_")[0]
                    city_names.add(city_name)

        return list(city_names)

    def find_matching_paths(self, img_dir,lbl_dir, vae_dir = None, instance_dir = None):
        img_paths = glob.glob(os.path.join(img_dir, "*.tif"))
        lbl_paths = glob.glob(os.path.join(lbl_dir, "*.tif"))
        vae_paths = glob.glob(os.path.join(vae_dir, "*.npy"))
        instance_paths = glob.glob(os.path.join(instance_dir, "*.npy"))


        #print(len(vae_paths))

        #sam_paths = glob.glob(os.path.join(data_dir, "sam_RoPe/*.npy"))




        matching_paths = []
        # if len(sam_paths) == 0:
        #     self.with_sam = False
        # else:
        #     self.with_sam = True
        # if not self.with_sam:
        #     for img_path, lbl_path in zip(img_paths, lbl_paths):
        #         matching_paths.append(img_path + " " + lbl_path + " " + "null")
        #     return  matching_paths




        # Extract basenames without extensions
        img_basenames = {os.path.basename(img_path).split('.')[0]: img_path for img_path in img_paths}
        lbl_basenames = {os.path.basename(lbl_path).split('.')[0]: lbl_path for lbl_path in lbl_paths}
        vae_basenames = {os.path.basename(vae_path).split('.')[0]: vae_path for vae_path in vae_paths}
        inst_basenames = {os.path.basename(inst_path).split('.')[0][:-5]: inst_path for inst_path in instance_paths}
        #sam_basenames = {os.path.basename(sam_path).split('.')[0]: sam_path for sam_path in sam_paths}

        for basename in img_basenames:
            if basename in lbl_basenames:
                if basename in vae_basenames:
                    matching_paths.append((img_basenames[basename] + " " +   lbl_basenames[basename] + " " +  vae_basenames[basename] + " " + inst_basenames[basename]))
                # else:
                #     matching_paths.append((img_basenames[basename] + "\t" +   lbl_basenames[basename] + "\t" + "null" ))
        return matching_paths


    def get_node_edge(self, level):
        words = [level[str(i)]["words"] for i in range(len(level) - 1)]
        word_embeddings = []


        tokenize_words = [clip.tokenize(["a photo includes " + word + " ,which role is {}.".format(level["0"]["role"]) ]) for word in words]
        for words in tokenize_words:
            with torch.no_grad():
                word_embeddings.append(self.model.encode_text(words.cuda()).cpu())
        edge_index = torch.tensor(level["adj"], dtype=torch.long).t().contiguous()
        #剔除 edge_index 无效的连接点， 如果edge_index中元素 >= len(level）-1
        if edge_index.numel() > 0:
            # 计算最大有效节点索引
            max_valid_index = len(level) - 1

            # 构建一个掩码，标记所有有效的边
            # 两个条件都必须满足：每条边的起点和终点索引都必须小于最大有效节点索引
            mask = (edge_index[0] < max_valid_index) & (edge_index[1] < max_valid_index)

            # 应用掩码过滤edge_index
            edge_index = edge_index[:, mask]
        if edge_index.nelement() == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        word_embeddings = torch.cat(word_embeddings, dim=0)
        return word_embeddings, edge_index


        # for phrase in words:
        #     for word in phrase.split():
        #         if word not in word_embeddings:
        #             tokenized_word = clip.tokenize([word])
        #             with torch.no_grad():
        #                 word_embedding = self.model.encode_text(tokenized_word.cuda())
        #             word_embeddings.append(word_embedding)
        # # 构建 edge_index
        # edge_index = torch.tensor(data["adj"], dtype=torch.long).t().contiguous()
        # word_embeddings = torch.stack(word_embeddings, dim=0)
        # return word_embeddings , edge_index


    def get_whole_hierarchy_node_and_edge(self, level1,level2, level3, contains_12, contains_23 ):

        words_1 = [level1[str(i)]["words"] for i in range(len(level1) - 1)]
        words_2 = [level2[str(i)]["words"] for i in range(len(level2) - 1)]
        words_3 = [level3[str(i)]["words"] for i in range(len(level3) - 1)]
        word_embeddings = []
        n1, n2, n3 = len(words_1), len(words_2), len(words_3)
        words = words_1 + words_2 + words_3
        tokenize_words = []
        for i, word in enumerate(words):
            if i <= n1:
                role = level1["0"]["role"]
            elif i > n1 and i <= n2:
                role = level2["0"]["role"]
            else:
                role = level3["0"]["role"]
            tokenize_words.append(clip.tokenize(["a photo includes " + word + " ,which role is {}".format(role)]))
        for words in tokenize_words:
            with torch.no_grad():
                word_embeddings.append(self.model.encode_text(words.cuda()).cpu())

        edge_index = []
        for key, vals in contains_12.items():
            id1 = int(key)
            for val in vals:
                id2 = int(val) + n1
                edge_index.append([id1, id2])
        for key, vals in contains_23.items():
            id1 = int(key) + n1
            for val in vals:
                id2 = int(val) + n1 + n2
                edge_index.append([id1, id2])

        edge_adj_1 = torch.tensor(level1["adj"], dtype=torch.long).t().contiguous()
        edge_adj_2 = torch.tensor(level2["adj"], dtype=torch.long).t().contiguous() + n1
        edge_adj_3 = torch.tensor(level1["adj"], dtype=torch.long).t().contiguous() + n1 + n2
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = torch.cat((edge_index, edge_adj_1, edge_adj_2, edge_adj_3), dim=1)

        if edge_index.numel() > 0:
            # 计算最大有效节点索引
            max_valid_index = n1 + n2 + n3

            # 构建一个掩码，标记所有有效的边
            # 两个条件都必须满足：每条边的起点和终点索引都必须小于最大有效节点索引
            mask = (edge_index[0] < max_valid_index) & (edge_index[1] < max_valid_index)

            # 应用掩码过滤edge_index
            edge_index = edge_index[:, mask]

        if edge_index.nelement() == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        word_embeddings = torch.cat(word_embeddings, dim=0)
        return word_embeddings, edge_index

    def get_hierarchy_node_and_edge(self, level1, level2,hierarchy_relations ):
        words_1 = [level1[str(i)]["words"] for i in range(len(level1) - 1)]
        words_2 = [level2[str(i)]["words"] for i in range(len(level2) - 1)]



        word_embeddings = []
        n1, n2 =  len(words_1), len(words_2)
        n = n1 + n2
        words = words_1 + words_2
        tokenize_words = []

        for i, word in enumerate(words):
            if i <= n1:
                role = level1["0"]["role"]
            else:
                role = level2["0"]["role"]
            tokenize_words.append(clip.tokenize(["a photo includes " + word + " ,which role is {}".format(role)]))

        for words in tokenize_words:
            with torch.no_grad():
                word_embeddings.append(self.model.encode_text(words.cuda()).cpu())

        edge_index = []
        for key, vals in hierarchy_relations.items():
            id1 = int(key)
            for val in vals:
                id2 = int(val) + n1
                edge_index.append([id1, id2])

        edge_adj = torch.tensor(level2["adj"], dtype=torch.long).t().contiguous() + n1
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = torch.cat((edge_index, edge_adj), dim=1)
        if edge_index.numel() > 0:
            # 计算最大有效节点索引
            max_valid_index = n

            # 构建一个掩码，标记所有有效的边
            # 两个条件都必须满足：每条边的起点和终点索引都必须小于最大有效节点索引
            mask = (edge_index[0] < max_valid_index) & (edge_index[1] < max_valid_index)

            # 应用掩码过滤edge_index
            edge_index = edge_index[:, mask]
        if edge_index.nelement() == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        word_embeddings = torch.cat(word_embeddings, dim=0)
        return word_embeddings, edge_index

    def compute_instance_corners(self, instance_map, epsilon=1.0):
        """
        Computes the corner points (key points) for each ID in the instance map.

        Args:
            instance_map (torch.Tensor): The instance map, shape (H, W).
            epsilon (float): Approximation accuracy. Smaller values result in more points.

        Returns:
            dict: A dictionary where keys are instance IDs and values are corner points (N x 2).
        """
        instance_map_np = instance_map.numpy().astype(np.uint8)  # Convert to NumPy for processing
        unique_ids = np.unique(instance_map_np)  # Get all unique IDs in the instance map
        corners = {}

        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue

            # Create a binary mask for the current instance
            mask = (instance_map_np == instance_id).astype(np.uint8)

            # Find contours for the current instance
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



            # Extract the corner points using polygon approximation
            corner_points = []
            # Flatten the contour points to N x 2 format
            for contour in contours:
                if contour.ndim == 3:
                    contour = contour.squeeze(1)  # Remove extra dimension if necessary
                corner_points.extend(contour)

            corners[instance_id] = np.array(corner_points) # Store corner points


        return corners

    def display_instance_map_with_corners(self, instance_map, corners):
        """
        Displays the instance map with corners overlaid.

        Args:
            instance_map (torch.Tensor): The instance map, shape (H, W).
            corners (dict): A dictionary where keys are instance IDs and values are corner points (N x 2).
        """
        instance_map_np = instance_map.numpy()
        unique_ids = np.unique(instance_map_np)

        # Create an RGB visualization of the instance map
        h, w = instance_map_np.shape
        instance_visual = np.zeros((h, w, 3), dtype=np.uint8)
        color_map = plt.cm.get_cmap("tab10", len(unique_ids))

        for idx, instance_id in enumerate(unique_ids):
            if instance_id == 0:  # Skip background
                continue
            instance_visual[instance_map_np == instance_id] = np.array(color_map(idx)[:3]) * 255

        # Plot the instance map
        plt.figure(figsize=(10, 10))
        plt.imshow(instance_visual)
        plt.title("Instance Map with Corners")
        plt.axis("off")

        # Overlay corners
        # for instance_id, points in corners.items():
        #     plt.scatter(points[:, 0], points[:, 1], s=15, label=f"Instance {instance_id}")

        for instance_id, points in corners.items():
            if len(points) > 1:
                points = np.vstack([points, points[0]])  # Close the polygon
                plt.plot(points[:, 0], points[:, 1], label=f"Instance {instance_id}")

        plt.legend()
        plt.show()

    def process_instance_map(self, instance_path, epsilon=5.0):
        """
        Processes an instance map to compute and save corner points.

        Args:
            instance_path (str): Path to the instance map `.npy` file.
            epsilon (float): Approximation accuracy for corner detection.
        """

        try:
            instance_map = np.load(instance_path)
        except Exception as e:
            print(f"Failed to load {instance_path}: {e}")
            instance_map = np.zeros_like(instance_map)
        return  instance_map

    def extract_bboxes(self, instance_map):
        objects = []

        # Get unique object ids in the instance map
        unique_ids = np.unique(instance_map)

        for obj_id in unique_ids:
            if obj_id == 0:  # Skip background (assuming 0 is the background)
                continue

            # Create a mask for the current object
            mask = (instance_map == obj_id)

            # Get the coordinates of non-zero elements (object pixels)
            coords = np.column_stack(np.where(mask))  # (y, x) coordinates

            # Get the bounding box (min and max of coordinates)
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)

            # Append the object information with bounding box
            objects.append({
                "bbox": [min_x, min_y, max_x, max_y],
            })

        # Convert the list of bounding boxes to a NumPy array
        bboxes_numpy = np.array([obj['bbox'] for obj in objects])

        return bboxes_numpy


    def __getitem__(self, idx):

        image_path, label_path, vae_path , instance_path = self.ids[idx].split(" ")




        # image_path = 'F:\\data\\OpenEarthMap\\Size_256\\train\\images\\baybay_4_512_768.tif'
        # label_path = 'F:\\data\\OpenEarthMap\\Size_256\\train\\labels\\baybay_4_512_768.tif'
        # vae_path = 'F:\\data\\OpenEarthMap\\Size_256\\train\\vae_feats\\baybay_4_512_768.npy'
        # instance_path = "F:\\data\\OpenEarthMap\\Size_256\\train\\graph_vit_l_14\\instancemap\\baybay_4_512_768_inst.npy"


        region_caption = os.path.basename(image_path).split('_')[0]


        #kmeans_labels = 0

        # desc_path = self.descriptions[idx]
        # with open(desc_path) as f:
        #     desc = json.load(f)
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        data = {}

        data["l1_node"] = torch.from_numpy(np.load(self.nodes_l1[idx])) if  self.nodes_l1 is not None else None
        data["l1_edge"] = torch.from_numpy(np.load(self.edges_l1[idx])) if  self.edges_l1 is not None else None
        data["l2_node"] = torch.from_numpy(np.load(self.nodes_l2[idx])) if  self.nodes_l2 is not None else None
        data["l2_edge"] = torch.from_numpy(np.load(self.edges_l2[idx])) if  self.edges_l2 is not None else None
        data["l3_node"] = torch.from_numpy(np.load(self.nodes_l3[idx])) if  self.nodes_l3 is not None else None
        data["l3_edge"] = torch.from_numpy(np.load(self.edges_l3[idx])) if  self.edges_l3 is not None else None



        if not os.path.exists(label_path):
            label = None
            print.error('cannot find {}'.format(label_path))
        else:
            label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if self.dict is not None:
                label_map = np.vectorize(self.dict.get)(label)
                if None in label_map:
                    pix_val = np.unique(label)
                    mix_index = []
                    for pix in pix_val:
                        if str(pix_val) not in self.dict.keys():
                            mix_index.append(pix)
                    assert len(mix_index) == 0, '{}\n cannot find {} in transxml.'.format(label_path, mix_index)
                else:
                    label = label_map


        if vae_path != "null":
            try:
                vae_feat = np.load(vae_path)
            except Exception as e:
                # print(f"Failed to load {sam_path}: {e}")
                vae_feat = np.zeros_like(label)

            vae_feat = torch.from_numpy(vae_feat).float()
        else:
            vae_feat = None

        if instance_path != "null":
           # instance_map, corners = self.process_instance_map(instance_path)
            instance_map = self.process_instance_map(instance_path)

            bboxes_path = image_path.replace("images", "bboxes").replace(".tif", ".npy")
            if os.path.exists(bboxes_path):
                bboxes = np.load(bboxes_path)
            else:
                bboxes = self.extract_bboxes(instance_map)
            bboxes = torch.from_numpy(bboxes)
        else:
            instance_map = None
            bboxes = None

        img = img.copy()

        label = label.copy()

        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(151)
        text = ''
        for i in range(len(class_ids)):
            text += OEM_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]

        kmeans_labels = 0
        if self.kmeans_model is not None:
            label_feats = extract_features(label, 9).reshape(1, -1)
            kmeans_labels = self.kmeans_model.predict(label_feats)
            kmeans_labels = torch.from_numpy(kmeans_labels)

        img_records = {}

        # if self.aug_transform is not None:
        #     oldimg, newimg, label = self.aug_transform(oldimg, newimg, labelmap=label)


        label = torch.from_numpy(label).float().unsqueeze(0)
        instance_map = torch.from_numpy(instance_map).float()
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1)
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.label_transform is not None:
        #     label = self.label_transform(label)
        if self.transform is not None:
            instance_map = instance_map.unsqueeze(0)
            img, instance_map, label = self.transform(img,instance_map,label)





        img_records["img_path"] = image_path
        img_records["label_path"] = label_path
        img_records["vae_path"] = vae_path
        img_records["captions"] = text
        img_records["regoins"] = region_caption

        return img, label, vae_feat, instance_map, kmeans_labels, bboxes, img_records, data
    # def __getitem__(self, idx):
    #
    #     image_path, label_path, sam_path = self.ids[idx].split(" ")
    #
    #     json = self.descriptions[idx]
    #     with open(json) as f:
    #         desc = json.load(f)
    #     img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)[:, :, :3]
    #
    #     captions = desc['caption'].replace("[","").replace("]","")
    #     l1 = desc["Level-1"]
    #     l2 = desc["Level-2"]
    #     l3 = desc["Level-3"]
    #
    #     contains_12 = desc["contains_12"]
    #     contains_23 = desc["contains_23"]
    #
    #
    #     node_l1, l1_edge = self.get_node_edge(l1)
    #     node_l2, l2_edge = self.get_node_edge(l2)
    #     node_l3, l3_edge = self.get_node_edge(l3)
    #
    #     node_12, l12_edge = self.get_hierarchy_node_and_edge(l1, l2, contains_12)
    #     node_23, l23_edge = self.get_hierarchy_node_and_edge(l2, l3, contains_23)
    #
    #     data = {}
    #     data["captions"] = captions
    #     data["l1_node"] = node_l1
    #     data["l1_edge"] = l1_edge
    #     data["l2_node"] = node_l2
    #     data["l2_edge"] = l2_edge
    #     data["l3_node"] = node_l3
    #     data["l3_edge"] = l3_edge
    #     data["l12_node"] = node_12
    #     data["l12_edge"] = l12_edge
    #     data["l23_node"] = node_23
    #     data["l23_edge"] = l23_edge
    #
    #
    #
    #     #计算node_l1, node_l2, node_l3
    #     #计算node_l1l2, nodel2l3
    #
    #     #sam_mask = None
    #
    #     if not os.path.exists(label_path):
    #         label = None
    #         print.error('cannot find {}'.format(label_path))
    #     else:
    #         label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    #         if self.dict is not None:
    #             label_map = np.vectorize(self.dict.get)(label)
    #             if None in label_map:
    #                 pix_val = np.unique(label)
    #                 mix_index = []
    #                 for pix in pix_val:
    #                     if str(pix_val) not in self.dict.keys():
    #                         mix_index.append(pix)
    #                 assert len(mix_index) == 0, '{}\n cannot find {} in transxml.'.format(label_path, mix_index)
    #             else:
    #                 label = label_map
    #
    #     sam_mask = np.zeros_like(label)
    #     if sam_path != "null":
    #         try:
    #             sam_mask = np.load(sam_path)
    #         except Exception as e:
    #             #print(f"Failed to load {sam_path}: {e}")
    #             sam_mask = np.zeros_like(label)
    #
    #     sam_mask = torch.from_numpy(sam_mask).float().unsqueeze(0)
    #
    #     img = img.copy()
    #
    #     label = label.copy()
    #     img_zero = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    #
    #     label[img_zero] = 0
    #     img_records = {}
    #
    #     # if self.aug_transform is not None:
    #     #     oldimg, newimg, label = self.aug_transform(oldimg, newimg, labelmap=label)
    #
    #     img = Image.fromarray(img)
    #     label = torch.from_numpy(label).float().unsqueeze(0)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.label_transform is not None:
    #         label = self.label_transform(label)
    #
    #     if isinstance(img, np.ndarray):
    #         img = torch.from_numpy(img).permute(2, 0, 1)
    #
    #
    #
    #
    #     # img = torch.cat([oldimg, newimg], dim=0)
    #
    #     # ann = self.anns.get(str(self.images[idx]["id"]))
    #     #
    #     # mask_old = self.decode_masks(ann[-2], "sam_old")
    #     # mask_new = self.decode_masks(ann[-1], "sam_new")
    #     # mask_old, mask_new = [], []
    #     # if "sam_old" in ann[-2].keys():
    #     #     sam_old =  ann[-2]["sam_old"]
    #     #     for obj in sam_old:
    #     #         rle = obj["segmentation"]
    #     #         mask = maskUtils.decode(rle )
    #     #         mask_old.append(mask)
    #     #     mask_old = np.dstack(mask_old).transpose(2,0,1)
    #     #     mask_old = torch.from_numpy(mask_old)
    #     #
    #     # if "sam_new" in ann[-1].keys():
    #     #     sam_new = ann[-1]["sam_new"]
    #     #     for obj in sam_new:
    #     #         rle = obj["segmentation"]
    #     #         mask = maskUtils.decode(rle)
    #     #         mask_new.append(mask)
    #     #     mask_new = np.dstack(mask_new).transpose(2, 0, 1)
    #     #     mask_new = torch.from_numpy(mask_new)
    #
    #     img_records["img_path"] = image_path
    #     img_records["label_path"] = label_path
    #     if self.with_graph:
    #         if self.node is not None and self.edge_index is not None:
    #             node_feats = np.load(self.node[idx])
    #             edge_idx = np.load(self.edge_index[idx])
    #             node_feats = torch.from_numpy(node_feats)
    #             # 获取第二列到第四列的最大值
    #
    #
    #             edge_indexs = torch.from_numpy(edge_idx)
    #             if node_feats.nelement() == 0:
    #                 node_feats = torch.zeros((1,8))
    #             else:
    #                 max_vals = node_feats[:, 0:5].max(dim=0)[0]
    #                 # 对第二列到第四列进行最大值归一化
    #                 node_feats[:, 0:5] = node_feats[:, 0:5] / max_vals
    #                 if edge_indexs.nelement() == 0:
    #                     edge_indexs = torch.tensor([[1], [1]], dtype=torch.long)
    #                 edge_indexs = edge_indexs - 1
    #             #edge_indexs = torch.clamp(edge_indexs, 0, node_feats.shape[0] - 1)
    #         else:
    #             node_feats, edge_indexs = None, None
    #
    #         return img, label, sam_mask, img_records, node_feats, edge_indexs
    #     else:
    #         return img, label, sam_mask, img_records


    def __len__(self):
        return len(self.ids)

    def readtransxml(self,transxml):
        path_transxml = transxml
        if not os.path.exists(path_transxml):
            print('Error:{} is not existed.'.format(path_transxml))
        transBM = parse(path_transxml)
        root = transBM.documentElement
        all_codes = root.getElementsByTagName('BM')
        all_dict = {}
        num_class = 0
        for node in all_codes:
            class_geoid_name = node.attributes['key'].value
            class_id = node.attributes['val'].value
            all_dict[int(class_geoid_name)] = int(class_id)
            if int(class_id) > num_class:
                num_class = num_class + 1
        return num_class,all_dict





def readtransxml( transxml):
    path_transxml = transxml
    if not os.path.exists(path_transxml):
        print('Error:{} is not existed.'.format(path_transxml))
    transBM = parse(path_transxml)
    root = transBM.documentElement
    all_codes = root.getElementsByTagName('BM')
    all_dict = {}
    num_class = 0
    for node in all_codes:
        class_geoid_name = node.attributes['key'].value
        class_id = node.attributes['val'].value
        all_dict[int(class_geoid_name)] = class_id
        # if int(class_id) > num_class:
        #     num_class = num_class + 1
    return num_class, all_dict

import pandas as pd
def addcdtxt2clipcaption(input_txt,transxml,output_dir):
    with open(input_txt, "r") as f:
        ids = f.readlines()
    prompt = 'changes from {} to {}'
    _, class_trans = readtransxml(transxml)

    out_dict = {
        "old_path": [],
        "new_path": [],
        "target_path": [],
        "caption": []
    }

    for line in tqdm(ids):
        old_path, new_path, label_path = line.rstrip().split()
        label_img = cv2.imdecode(np.fromfile(label_path,dtype=np.uint8),-1)

        classes = np.unique(label_img)
        caption_classes  = []
        for cls in classes:
            if cls == 0:#background
                continue
            caption_classes.append(class_trans[cls])


        prompt = 'changes from {} to {}'.format("unknown",', '.join(caption_classes))
        out_dict["old_path"].append(old_path)
        out_dict["new_path"].append(new_path)
        out_dict["target_path"].append(label_path)
        out_dict["caption"].append(prompt)
    df = pd.DataFrame(out_dict)
    df.to_csv(output_dir,index= None, encoding="utf-8")

    return


import numpy as np
import json




def labels_to_coco(all_paths, category_ids, output_json_path, background = 0, change_value = None):
    # Assuming 'maskUtils.encode()' returns a binary format, you might need something like this
    def convert_rle(rle):
        if type(rle) == list:
            # Already in a serializable format
            return rle
        elif 'counts' in rle and type(rle['counts']) == bytes:
            # Convert bytes to string
            rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": id, "name": name} for id, name in category_ids.items()]
    }

    annotation_id = 1

    for image_id, path in tqdm(enumerate(all_paths)):
        oldimage_path, newimage_path, label_path = path.rstrip().split()
        label_image = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if label_image is None:
            continue  # Skip if the image couldn't be read

        # Add image information to COCO
        coco_data["images"].append({
            "id": image_id,
            "width": label_image.shape[1],
            "height": label_image.shape[0],
            "oldpath": oldimage_path,
            "newpath": newimage_path,
            "labelpath": label_path

        })
        pixel_values = np.unique(label_image)
        for pixel_value  in pixel_values:
            if pixel_value == background:
                continue
            category_id = category_ids[pixel_value]

            #print("val = {}, category_id = {}".format(pixel_value, category_id))
            binary_mask = np.uint8(label_image == pixel_value)

            rle = maskUtils.encode(np.asfortranarray(binary_mask))
            rle = convert_rle(rle)


            area = maskUtils.area(rle)
            bbox = maskUtils.toBbox(rle).tolist()



            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "bbox": bbox,
                "area": area.item(),  # Convert from numpy to Python scalar
                "iscrowd": 0
            })
            annotation_id += 1

    # Save the COCO data to a JSON file
    with open(output_json_path, 'w', encoding="utf-8") as json_file:
        json.dump(coco_data, json_file, ensure_ascii=False, indent=4)
# Assuming label_image is a 2D numpy array you have loaded or generated
# label_image = ...

# Convert to COCO format
# coco_data = label_image_to_coco(label_image)

# Save to JSON file
# with open('label_image_coco_format.json', 'w') as f:
#     json.dump(coco_data, f, indent=4)
def split_train_val(json_file, val_ratio=0.1, val_size=500):
    with open(json_file, 'r', encoding="utf-8") as file:
        annotations = json.load(file)
    images = annotations["images"]
    anns = annotations["annotations"]
    # Compute validation size
    val_size = min(val_size, int(len(images) * val_ratio))

    # Randomly select indices for validation images
    val_indices = np.random.choice(range(len(images)), val_size, replace=False)
    val_image_ids = {images[i]['id'] for i in val_indices}

    # Separate images into validation and training sets
    val_images = [img for img in images if img['id'] in val_image_ids]
    train_images = [img for img in images if img['id'] not in val_image_ids]

    # Separate annotations into validation and training sets
    val_anns = [ann for ann in anns if ann['image_id'] in val_image_ids]
    train_anns = [ann for ann in anns if ann['image_id'] not in val_image_ids]

    # Create new dictionaries for training and validation
    train_dataset = {"images": train_images, "annotations": train_anns, "categories": annotations["categories"]}
    val_dataset = {"images": val_images, "annotations": val_anns, "categories": annotations["categories"]}

    # Save the new datasets
    train_json = json_file.replace('.json', '_train.json')
    val_json = json_file.replace('.json', '_val.json')

    with open(train_json, 'w', encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=4)

    with open(val_json, 'w', encoding="utf-8") as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=4)

    print(f"Training and validation sets have been saved to {train_json} and {val_json}, respectively.")

def copyxBD_to_OpenEarthMap(xbd_dataset,OEM_DIR,csv_file ):
    column_names = ["src Path", "dst Path"]
    df_existing = pd.read_csv(csv_file, names=column_names, header=None)
    # df_existing = pd.read_csv(csv_file)

    # 将每一列转换为列表
    src_name = df_existing["src Path"].tolist()
    dest_name = df_existing["dst Path"].tolist()

    # 获取所有.png文件路径并创建一个字典以文件名为键，完整路径为值
    xbd_allpaths = {}
    for dir in xbd_dataset:
        for path in glob.glob(os.path.join(dir, "*.png")):
            xbd_allpaths[os.path.basename(path)] = path

    # 创建输出目录
    os.makedirs(OEM_DIR, exist_ok=True)

    # 查找src_name中的文件并复制到output_dir
    for src, dst in zip(src_name, dest_name):
        # 查找文件名在xbd_allpaths中的完整路径
        if src in xbd_allpaths:
            src_path = xbd_allpaths[src]
            # 构建目标文件路径
            img_folder = os.path.join(OEM_DIR, dst.rsplit('_', 1)[0], "images")
            os.makedirs(img_folder, exist_ok=True)  # 创建目标文件夹（如果不存在）
            dst_path = os.path.join(img_folder, dst)
            # 复制文件
            shutil.copy(src_path, dst_path)
            #print(f"Copied {src_path} to {dst_path}")
        else:
            print(f"File {src} not found in dataset.")

    print("文件复制完成。")

    sys.exit(1)


def extract_geometric_features(label_img, regions):
    features = []

    for region in regions:
        # 计算区域的面积，即区域内像素的总数
        area = region.area

        # 计算区域的周长，即区域边界上的像素数
        perimeter = region.perimeter

        # 计算区域的主轴长度，即区域的长轴
        major_axis_length = region.major_axis_length

        # 计算区域的次轴长度，即区域的短轴
        minor_axis_length = region.minor_axis_length

        # 计算区域的偏心率，即长轴和短轴的离心率，偏心率是一个0到1之间的值，值越接近1，区域形状越细长
        eccentricity = region.eccentricity

        # 计算区域的密实度，即区域面积与凸包面积的比值，密实度是一个0到1之间的值，值越接近1，区域形状越紧密
        solidity = region.solidity

        # 获取区域的标签值，通常用于标识区域
        label_value = region.label

        # 获取区域的类别，根据区域的坐标从标签图像中提取类别信息
        label_class = label_img[region.coords[0][0], region.coords[0][1]]

        features.append(
            [label_class,label_value,  area, perimeter, major_axis_length, minor_axis_length, eccentricity, solidity])


    return np.array(features)


def create_edge_index_adjacent(label_img, regions):
    num_regions = len(regions)
    edge_index = np.zeros((num_regions, num_regions), dtype=int)

    edge_index_mask = np.zeros((num_regions, num_regions), dtype=int)
    #dilated_label_img = dilation(label_img, square(3))

    for i in range(num_regions):
        region_i = (label_img == (i + 1))
        region_i = dilation(region_i, square(3))
        for j in range(i + 1, num_regions):
            region_j = (label_img == (j + 1))
            if np.any(region_j & region_i) and not edge_index_mask[i,j]:
                edge_index[i, j] = 1
                edge_index[j, i] = 1
            edge_index_mask[i, j] = 1
            edge_index_mask[j, i] = 1

    return edge_index
def create_edge_index(label_img, regions):
    n, m = label_img.shape

    current_node = 0
    edges = set()
    node_map = {}  # (i, j) -> node_index
   # regions = regionprops(labeled_array)
    # 遍历每个连通区域，给每个连通区域分配一个节点索引
    for region in regions:
        for coord in region.coords:
            node_map[tuple(coord)] = region.label
        #current_node += 1

    # 遍历每个连通区域，找到其边界像素，并确定与其他分类节点的连接
    for region in regions:
        current_node = node_map[tuple(region.coords[0])]
        for coord in region.coords:
            i, j = coord
            # 检查四个方向上的邻居
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    neighbor_coord = (ni, nj)
                    if neighbor_coord in node_map:
                        neighbor_node = node_map[neighbor_coord]
                        if neighbor_node != current_node:
                            edges.add((current_node, neighbor_node))
                            edges.add((neighbor_node, current_node))
    edges = list(edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    #convert edge_index to adjacent mat
    num_nodes = len(regions)   # region labels are 1-based
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges:
        adj_matrix[edge[0]-1, edge[1]-1] = 1
        adj_matrix[edge[1]-1, edge[0]-1] = 1

    return edge_index, adj_matrix
    # 转换为 PyTorch 的 edge_index 格式

    # Create a dictionary to store the coordinates of each region's boundary pixels
    boundaries = {region.label: set() for region in regions}

    for region in regions:
        coords = region.coords
        for coord in coords:
            x, y = coord
            if (x > 0 and markers[x - 1, y] != region.label) or \
                    (x < markers.shape[0] - 1 and markers[x + 1, y] != region.label) or \
                    (y > 0 and markers[x, y - 1] != region.label) or \
                    (y < markers.shape[1] - 1 and markers[x, y + 1] != region.label):
                boundaries[region.label].add((x, y))

    # Create an adjacency set to avoid duplicate edges
    adjacency_set = set()

    # Check for adjacent regions
    for region in regions:
        region_label = region.label
        for coord in boundaries[region_label]:
            x, y = coord
            neighbors = [
                markers[x - 1, y] if x > 0 else 0,
                markers[x + 1, y] if x < markers.shape[0] - 1 else 0,
                markers[x, y - 1] if y > 0 else 0,
                markers[x, y + 1] if y < markers.shape[1] - 1 else 0,
            ]
            for neighbor_label in neighbors:
                if neighbor_label != region_label and neighbor_label != 0:
                    # Ensure we add each edge only once
                    edge = tuple(sorted((region_label - 1, neighbor_label - 1)))
                    adjacency_set.add(edge)

    # Convert adjacency set to edge index array
    edge_index = np.array(list(adjacency_set)).T

    return edge_index


# Example usage with the `markers` array from image segmentation
# markers = segment_image(image)  # Assume `segment_image` is defined as in previous examples
# edge_index = create_edge_index(markers)
# print(edge_index)

def save_graph_data(lbl_img, features, edge_index, feature_file='node_features.npy', edge_file='edge_index.npy', lbl_file='label_img.npy'):
    np.save(feature_file, features)
    np.save(edge_file, edge_index)
    np.save(lbl_file, lbl_img)
def load_graph_data(feature_file='node_features.npy', edge_file='edge_index.npy'):
    features = np.load(feature_file)
    edge_index = np.load(edge_file)
    return features, edge_index
def image_to_graph( target_path):
    #image = cv2.imread(image_path)
    markers = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    label_img = measure.label(markers)
    regions = regionprops(label_img)

    geometric_features = extract_geometric_features(markers, regions)
    #改成计算邻接矩阵, 邻接矩阵的大小是 node_num * node_num
    edge_index,edge_adj = create_edge_index(label_img, regions)
    return edge_adj, geometric_features, edge_index



# def get_node_edge_features_from_paths(samples_path, output_dir, tag = "train"):
#     out_dir = os.path.join(output_dir, tag)
#     os.makedirs(out_dir, exist_ok=True)
#     for target_path in tqdm(samples_path):
#         image_name = os.path.basename(target_path).split(".")[0]
#         lbl_path = os.path.join(out_dir, image_name + "_label_img.npy")
#         node_path = os.path.join(out_dir, image_name + "_node_features.npy")
#         edge_path = os.path.join(out_dir, image_name + "_edge_index.npy")
#         label_img, node_feats, edges = image_to_graph(target_path)
#         save_graph_data(label_img, node_feats, edges, node_path, edge_path, lbl_path)

def get_node_edge_features(txt, all_paths, output_dir, tag = "train"):

    with open(txt,encoding='utf-8',mode='r') as f:
        lines = f.readlines()
    samples_path = []
    for line in lines:
        line = line.rstrip()
        if line in all_paths:
            samples_path.append(all_paths[line])
        else:
            print("Error: can not find path:{}".format(line))
    out_dir = os.path.join(output_dir,tag)
    os.makedirs(out_dir,exist_ok=True)
    for target_path in tqdm(samples_path):
        image_name = os.path.basename(target_path).split(".")[0]
        lbl_path = os.path.join(out_dir, image_name + "_label_img.npy")
        node_path = os.path.join(out_dir,image_name + "_node_features.npy")
        edge_path = os.path.join(out_dir,image_name + "_edge_index.npy")
        label_img, node_feats, edges = image_to_graph(target_path)
        save_graph_data(label_img, node_feats,edges,node_path, edge_path, lbl_path)
#import xml.etree.ElementTree as ET
# if __name__ == '__main__':

    # json_file = r"F:\data\02_1M2M_BHYB\1M2M_BHYB_linux.json"
    #
    #
    # split_train_val(json_file)
    #
    #
    # pass


    # from utils.tools.configer import Configer
    # from dataset.tools.cv2_aug_transform_chg import CV2AugCompose_CHG
    # config = Configer(configs=config)
    #
    # transforms = CV2AugCompose_CHG(config, split="train")
    # json_file = r"F:\data\02_1M2M_BHYB\1M2M_BHYB.json"
    #
    #
    #
    # dataset = SamClipCD_dataset(json_file, config, aug_transform= transforms)
    #
    # dataloader = DataLoader(dataset,
    #                         batch_size=1,
    #                         shuffle= False)
    #
    # iter = iter(dataloader)
    #
    # data = iter.next()
    # old_dir = "F:\data\SECOND_train_set\im1"
    # new_dir = "F:\data\SECOND_train_set\im2"
    # prelabel = "F:\data\SECOND_train_set\label1"
    # curlabel = "F:\data\SECOND_train_set\label2"
    #
    # oldpaths = glob.glob(os.path.join(old_dir,"*.png"))
    # all_paths = []
    # for path in tqdm(oldpaths):
    #     print("Process")
    #     oldpath = path
    #     basename = os.path.basename(oldpath)
    #     newpath = os.path.join(new_dir, basename)
    #     precls = os.path.join(prelabel, basename)
    #     curcls = os.path.join(curlabel, basename)
    #     all_paths.append(oldpath + "\t" + newpath + "\t" + curcls)


    #labels_to_coco(all_paths, )

import os
import os
from tqdm.contrib.concurrent import process_map


def process_image(target_path, out_dir):
    image_name = os.path.basename(target_path).split(".")[0]
    lbl_path = os.path.join(out_dir, image_name + "_label_img.npy")
    node_path = os.path.join(out_dir, image_name + "_node_features.npy")
    edge_path = os.path.join(out_dir, image_name + "_edge_index.npy")

    label_img, node_feats, edges = image_to_graph(target_path)  # 假设已经定义
    save_graph_data(label_img, node_feats, edges, node_path, edge_path, lbl_path)  # 假设已经定义


def get_node_edge_features_from_paths(samples_path, output_dir, tag="train"):
    out_dir = os.path.join(output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 使用 process_map 来映射函数并显示进度条
    process_map(process_image, samples_path, [out_dir] * len(samples_path), max_workers=os.cpu_count(), chunksize=1)


def visualize_graph(edge_index):
    # Create a directed graph from the edge index
    G = nx.DiGraph()

    # Add edges from the edge index
    G.add_edges_from(edge_index)

    # Draw the graph
    pos = nx.kamada_kawai_layout(G)
    #pos = nx.spring_layout(G)  # positions for all nodes using the Spring layout
    nx.draw(G, pos, with_labels=False, node_color='skyblue', edge_color='#3b0c8c', node_size=15, font_size=16,
            font_color='black')
    # edge_labels = dict([((u, v,), f'{u}-{v}')
    #                     for u, v, d in G.edges(data=True)])
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title('Graph Visualization')
    plt.show()

def crop_images( txt , all_paths, crop_size=(256, 256), save_dir=None):

    with open(txt,encoding='utf-8',mode='r') as f:
        lines = f.readlines()
    samples_path = []
    for line in lines:
        line = line.rstrip()
        if line in all_paths:
            samples_path.append(all_paths[line])
        else:
            print("Error: can not find path:{}".format(line))
    image_paths = [path[0] for path in samples_path]
    label_paths = [path[1] for path in samples_path]
    if save_dir is None:
        save_dir = os.path.join(data_dir, 'cropped')
    os.makedirs(save_dir, exist_ok=True)

    cropped_img_dir = os.path.join(save_dir, 'images')
    cropped_lbl_dir = os.path.join(save_dir, 'labels')
    os.makedirs(cropped_img_dir, exist_ok=True)
    os.makedirs(cropped_lbl_dir, exist_ok=True)

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths)):
        with rasterio.open(img_path) as src_img:
            with rasterio.open(lbl_path) as src_lbl:
                # Loop through a grid of crops
                for i in range(0, src_img.width, crop_size[0]):
                    for j in range(0, src_img.height, crop_size[1]):
                        # Define the window position and size
                        window = Window(i, j, crop_size[0], crop_size[1])
                        # Read the window and its transform
                        img_data = src_img.read(window=window)
                        lbl_data = src_lbl.read(window=window)
                        img_transform = src_img.window_transform(window)
                        lbl_transform = src_lbl.window_transform(window)

                        # Define file paths for saving
                        cropped_img_name = f"{os.path.basename(img_path).replace('.tif', '')}_{i}_{j}.tif"
                        cropped_lbl_name = f"{os.path.basename(lbl_path).replace('.tif', '')}_{i}_{j}.tif"

                        # Write the cropped image and label with the updated transform
                        with rasterio.open(
                                os.path.join(cropped_img_dir, cropped_img_name), 'w',
                                driver='GTiff', height=crop_size[1], width=crop_size[0],
                                count=src_img.count, dtype=src_img.dtypes[0],
                                crs=src_img.crs, transform=img_transform
                        ) as dst_img:
                            dst_img.write(img_data)

                        with rasterio.open(
                                os.path.join(cropped_lbl_dir, cropped_lbl_name), 'w',
                                driver='GTiff', height=crop_size[1], width=crop_size[0],
                                count=src_lbl.count, dtype=src_lbl.dtypes[0],
                                crs=src_lbl.crs, transform=lbl_transform
                        ) as dst_lbl:
                            dst_lbl.write(lbl_data)

if __name__ == "__main__":
    import torch
    import numpy as np

    import glob
    import pandas as pd
    #copy xbd dataset to OEM
    import shutil

    # 定义源文件夹和目标文件夹
    # source_folder = r'D:\dengkai\data\BJ20012B8VI_01020180206F_8BitRGB_samples'
    # labels_folder = os.path.join(source_folder, 'labels')
    # images_folder = os.path.join(source_folder, 'images')
    #
    # # 如果目标文件夹不存在，则创建它们
    # os.makedirs(labels_folder, exist_ok=True)
    # os.makedirs(images_folder, exist_ok=True)
    #
    # # 遍历源文件夹中的所有文件
    # for filename in os.listdir(source_folder):
    #     file_path = os.path.join(source_folder, filename)
    #
    #     # 跳过文件夹
    #     if os.path.isdir(file_path):
    #         continue
    #
    #     # 判断文件是否是标签文件
    #     if 'label' in filename:
    #         shutil.move(file_path, os.path.join(labels_folder, filename))
    #     else:
    #         shutil.move(file_path, os.path.join(images_folder, filename))
    #
    # print("Files have been sorted into 'labels' and 'images' folders.")
    #
    # sys.exit(-1)
    # xbd_dataset = [r"F:\data\OpenEarthMap\xView2\data\train_images_labels_targets\train\images",r"F:\data\OpenEarthMap\xView2\data\test_images_labels_targets\test\images",r"F:\data\OpenEarthMap\xView2\data\tier3\images",r"F:\data\OpenEarthMap\xView2\data\hold\images"]
    # output_dir = r"F:\data\OpenEarthMap\OpenEarthMap_wo_xBD"
    # csv_file = r"F:\data\OpenEarthMap\OpenEarthMap_wo_xBD\xbd_files.csv"
    # # 指定列名读取CSV文件
    #
    # copyxBD_to_OpenEarthMap(xbd_dataset, output_dir, csv_file)
    # sys.exit(1)
    # column_names = ["src Path", "dst Path"]
    # df_existing = pd.read_csv(csv_file, names=column_names, header=None)
    # #df_existing = pd.read_csv(csv_file)
    #
    # # 将每一列转换为列表
    # src_name = df_existing["src Path"].tolist()
    # dest_name = df_existing["dst Path"].tolist()
    #
    # # 获取所有.png文件路径并创建一个字典以文件名为键，完整路径为值
    # xbd_allpaths = {}
    # for dir in xbd_dataset:
    #     for path in glob.glob(os.path.join(dir, "*.png")):
    #         xbd_allpaths[os.path.basename(path)] = path
    #
    # # 创建输出目录
    # os.makedirs(output_dir, exist_ok=True)
    #
    # # 查找src_name中的文件并复制到output_dir
    # for src, dst in zip(src_name, dest_name):
    #     # 查找文件名在xbd_allpaths中的完整路径
    #     if src in xbd_allpaths:
    #         src_path = xbd_allpaths[src]
    #         # 构建目标文件路径
    #         img_folder = os.path.join(output_dir, dst.rsplit('_', 1)[0], "images")
    #         os.makedirs(img_folder, exist_ok=True)  # 创建目标文件夹（如果不存在）
    #         dst_path = os.path.join(img_folder, dst)
    #         # 复制文件
    #         shutil.copy(src_path, dst_path)
    #         print(f"Copied {src_path} to {dst_path}")
    #     else:
    #         print(f"File {src} not found in dataset.")
    #
    # print("文件复制完成。")
    #
    # sys.exit(1)

    from sklearn.cluster import KMeans
    import  numpy as np
    from sklearn.decomposition import PCA
    # edge_index = np.load(r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\train_graph\aachen_2_edge_index.npy").transpose(1,0)
    #
    # visualize_graph(edge_index)
    #
    # sys.exit(-1)


    data_dir = r"F:\data\OpenEarthMap\Size_256\train"
    train_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\train.txt"
    val_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\val.txt"
    test_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\test.txt"

    tag_dir = ["images", "labels"]

    img_paths = glob.glob(os.path.join(data_dir, "*/images/*.tif"))

    data_dir = r"F:\data\OpenEarthMap\Size_256\train\labels"
    output_base_dir = r"F:\data\OpenEarthMap\Size_256\train\classfications"

    lbl_paths = glob.glob(os.path.join(data_dir,  "*.tif"))

    num_original_classes = 9
    num_clusters = 20


    def extract_features(mask):
        features = []
        for i in range(num_original_classes):
            pixel_count = np.sum(mask == i)
            features.append(pixel_count)
        features = np.array(features) / np.prod(mask.shape)  # 归一化
        return features


    # 提取所有分割标注的特征
    features_list = []
    for lbl_path in tqdm(lbl_paths[:1000]):
        with rasterio.open(lbl_path) as src:
            mask = src.read(1)  # 读取第一个波段
            features = extract_features(mask)
            features_list.append(features)

    features = np.array(features_list)

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    # 获取聚类结果
    clusters = kmeans.labels_
    #可视化特征
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # 使用 t-SNE 进行降维（可选）
    # tsne = TSNE(n_components=2, random_state=42)
    # features_tsne = tsne.fit_transform(features)

    # 绘制聚类结果的二维散点图
    plt.figure(figsize=(10, 7))
    for cluster in range(num_clusters):
        cluster_indices = np.where(clusters == cluster)
        plt.scatter(features_pca[cluster_indices, 0], features_pca[cluster_indices, 1], label=f'Cluster {cluster}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Features and K-means Clusters')
    plt.legend()
    plt.show()



    # 将结果保存到字典或数据框中
    results = {'path': lbl_paths[:1000], 'cluster': clusters}
    results_df = pd.DataFrame(results)

    # 保存到CSV文件中
    results_df.to_csv(r'F:\data\OpenEarthMap\Size_256\train\classfications\clustering_results.csv', index=False)

    for cluster in range(num_clusters):
        cluster_dir = os.path.join(output_base_dir, f'cluster_{cluster}')
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

    for idx, row in results_df.iterrows():
        src_path = row['path'].replace("labels", "images")
        cluster = row['cluster']
        dst_dir = os.path.join(output_base_dir, f'cluster_{cluster}')
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)


    print("Clustering results saved to clustering_results.csv")
    sys.exit(1)
    # # 创建一个字典来存储图像和标签的配对
    # img_dict = {os.path.basename(path): path for path in img_paths}
    # lbl_dict = {os.path.basename(path): path for path in lbl_paths}
    #
    # # 寻找文件名相同的图像和标签并配对
    # all_paths = []
    # for img_name, img_path in img_dict.items():
    #     if img_name in lbl_dict:
    #         all_paths.append((img_path, lbl_dict[img_name]))
    #
    # # xbd_allpaths = {}
    # all_lbl_paths = {}
    # for path in all_paths:
    #         all_lbl_paths[os.path.basename(path[-1])] = path[-1]
    #
    # #clip all_paths(images and labels) to 256 or 512
    # all_paths_update = {}
    # for path in all_paths:
    #     all_paths_update[os.path.basename(path[-1])] = path
    # crop_images(train_name, all_paths_update,crop_size=(512,512), save_dir = r"F:\data\OpenEarthMap\Size_512\train")
    # crop_images(val_name, all_paths_update, crop_size=(512, 512), save_dir=r"F:\data\OpenEarthMap\Size_512\val")
    # crop_images(test_name, all_paths_update, crop_size=(512, 512), save_dir=r"F:\data\OpenEarthMap\Size_512\test")

    # get_node_edge_features(train_name,all_lbl_paths,output_dir=data_dir,tag = 'train_graph')
    # get_node_edge_features(val_name, all_lbl_paths, output_dir=data_dir, tag ='val_graph')
    # get_node_edge_features(test_name, all_lbl_paths, output_dir=data_dir, tag ='test_graph')

    train_256_dir = r"F:\data\OpenEarthMap\Size_256\train"
    val_256_dir = r"F:\data\OpenEarthMap\Size_256\val"
    # train_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\train.txt"
    # val_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\val.txt"
    # test_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\test.txt"

    tag_dir = ["images", "labels"]

    #img_paths = glob.glob(os.path.join(data_dir, "images/*.tif"))
    train_lbl_paths_256 = glob.glob(os.path.join(train_256_dir, "labels/*.tif"))
    val_lbl_paths_256 = glob.glob(os.path.join(val_256_dir, "labels/*.tif"))

    get_node_edge_features_from_paths(train_lbl_paths_256, output_dir=train_256_dir, tag = "train_graph")
    get_node_edge_features_from_paths(val_lbl_paths_256, output_dir=val_256_dir, tag="val_graph")

    train_512_dir = r"F:\data\OpenEarthMap\Size_512\train"
    val_512_dir = r"F:\data\OpenEarthMap\Size_512\val"
    # train_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\train.txt"
    # val_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\val.txt"
    # test_name = r"F:\data\OpenEarthMap\OpenEarthMap_w_xBD\test.txt"

    tag_dir = ["images", "labels"]

    # img_paths = glob.glob(os.path.join(data_dir, "images/*.tif"))
    train_lbl_paths_512 = glob.glob(os.path.join(train_512_dir, "labels/*.tif"))
    val_lbl_paths_512 = glob.glob(os.path.join(val_512_dir, "labels/*.tif"))

    get_node_edge_features_from_paths(train_lbl_paths_512, output_dir=train_512_dir, tag="train_graph")
    get_node_edge_features_from_paths(val_lbl_paths_512, output_dir=val_512_dir, tag="val_graph")



    # 假设512x512土地利用覆盖图
    #land_cover = np.random.randint(0, 11, (512, 512))



    # print(edge_index.shape)
    # print(edge_index)

