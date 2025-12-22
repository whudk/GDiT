from torch.utils.data import Dataset
from PIL import Image
import os, glob
import numpy as np
import torch
from  pathlib import  Path
import json
try:
    from torch_geometric.data import Data
except:
    Data = None

def make_empty_graph(graph_dim=768):
    """
    构造一个没有节点/没有边的空 graph，
    但字段齐全，避免 GDiT graph=None 的炸点。
    """
    assert Data is not None, "need torch_geometric to build empty graph"
    g = Data()
    g.x = torch.zeros((0, graph_dim), dtype=torch.float32)          # 无语义 node feat
    g.edge_index = torch.zeros((2, 0), dtype=torch.long)           # 无边
    g.node_boxxes = torch.zeros((0, 4), dtype=torch.float32)       # 无box
    g.batch = torch.zeros((0,), dtype=torch.long)
    return g


class M2ITargetRText(Dataset):
    """
    目标域：只有 region 文本（caption/key） + image + vae_feats
    images.json: {prompt: [img_paths...]}
    vae_feats.json: {prompt: [vae_paths...]}  与 images.json 对应
    """
    def __init__(self, data_dir, graph_dim=768, in_channels=4):
        super().__init__()
        self.graph_dim = graph_dim
        self.in_channels = in_channels

        self.img_json_path = os.path.join(data_dir, "image_with_pesudo.json")
        self.vae_json_path = os.path.join(data_dir, "vae_feats.json")

        assert os.path.exists(self.img_json_path), f"images.json not found: {self.img_json_path}"
        assert os.path.exists(self.vae_json_path), f"vae_feats.json not found: {self.vae_json_path}"

        with open(self.img_json_path, "r", encoding="utf-8") as f:
            img_data = json.load(f)
        with open(self.vae_json_path, "r", encoding="utf-8") as f:
            vae_data = json.load(f)

        self.samples = []  # 每个元素: {"image_path":..., "vae_path":..., "caption":...}

        assert isinstance(img_data, dict) and isinstance(vae_data, dict), \
            "当前版本假设两个 json 都是 dict {prompt: [paths...]}"

        for dataset_name, image_info in img_data.items():
            if not isinstance(image_info, list):
                continue

            assert dataset_name in vae_data, f"vae_feats.json 缺少 key: {dataset_name}"
            vae_list = vae_data[dataset_name]

            # 建立 vae 索引：stem -> path
            vae_map = {}
            dup = []
            for vp in vae_list:
                s = Path(vp).stem
                if s in vae_map:
                    dup.append(s)
                vae_map[s] = vp
            if dup:
                print(f"[Warn] {dataset_name}: duplicated vae stems (show first 5): {dup[:5]}")

            miss = 0
            for img_item in image_info:
                img_path = img_item.get("image_path", "")
                if not img_path:
                    continue

                stem = Path(img_path).stem
                vae_path = vae_map.get(stem, None)

                if vae_path is None:
                    miss += 1
                    # 你可以选择：直接跳过，或 raise
                    # raise AssertionError(f"{dataset_name}: 找不到对应 vae for image {img_path} (stem={stem})")
                    continue

                pseudo_path = img_item.get("pseudo_label", None)
                caption = img_item.get("captions", "")

                self.samples.append({
                    "dataset_name": dataset_name,
                    "pseudo_path": pseudo_path,
                    "image_path": img_path,
                    "vae_path": vae_path,
                    "caption": caption,
                })

            if miss > 0:
                print(f"[Warn] {dataset_name}: {miss} images have no matched vae_feats by stem")
            print(f"[M2ITargetRText] loaded {len(self.samples)} samples from {self.img_json_path}")
        # for dataset_name, image_info in img_data.items():
        #     if not isinstance(image_info, list):
        #         continue
        #
        #     assert dataset_name in vae_data, f"vae_feats.json 缺少 key: {dataset_name}"
        #     vae_list = vae_data[dataset_name]
        #     assert len(image_info) == len(vae_list), f"{dataset_name}: image 数量与 vae 数量不一致"
        #
        #     for img_item, vae_path in zip(image_info, vae_list):
        #         img_path = img_item.get("image_path")
        #         pseudo_path = img_item.get("pseudo_label")
        #         caption = img_item.get("captions", "")
        #
        #         self.samples.append({
        #             "dataset_name": dataset_name,
        #             "pseudo_path" : pseudo_path,
        #             "image_path": img_path,
        #             "vae_path": vae_path,
        #             "caption": caption,
        #         })
        #
        # print(f"[M2ITargetRText] loaded {len(self.samples)} samples from {self.img_json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        pseudo_path = sample["pseudo_path"]
        vae_path = sample["vae_path"]
        caption = sample["caption"]

        # --- image ---
        target = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
        target = torch.from_numpy(target).permute(2, 0, 1)  # [3,H,W]
        _, H, W = target.shape

        if pseudo_path is None or pseudo_path == "" or (not os.path.exists(pseudo_path)):
            cond_seg = torch.zeros((1, H, W), dtype=torch.float32)
        else:
            pseudo_img = Image.open(pseudo_path)  # 灰度标签图
            # 尺寸不一致就最近邻缩放到与 image 相同
            if pseudo_img.size != (W, H):
                pseudo_img = pseudo_img.resize((W, H), resample=Image.NEAREST)

            pseudo = np.array(pseudo_img, dtype=np.uint8)  # [H,W], 0..8
            pseudo_t = torch.from_numpy(pseudo).long()
            cond_seg = pseudo_t.float().unsqueeze(0)
        # # --- cond_seg: 目标域没有语义图，用全0 ---
        # cond_seg = torch.zeros((1, H, W), dtype=torch.float32)

        # --- cond_graph / cond_inst: 没有图结构，用占位 ---
        cond_graph = None
        cond_inst = torch.zeros((H, W), dtype=torch.long)

        # --- vae_feats: 一定存在 ---
        vae_feats_np = np.load(vae_path)
        vae_feats = torch.from_numpy(vae_feats_np).squeeze(0)

        img_records = {
            "img_path": image_path,
            "vae_path": vae_path,
            "captions": caption,   # 这里就是你说的 Region 文本
            "domain": "target",
        }

        return dict(
            image=target,
            cond_seg=cond_seg,
            cond_graph=cond_graph,
            cond_inst=cond_inst,
            vae_feats=vae_feats,
            img_records=img_records,
        )
# class M2ITargetRText(Dataset):
#     """
#     目标域：只有 region 文本（caption）+ image
#     不带语义图 S，不带图结构 G
#     """
#     def __init__(self, data_dir, graph_dim=768, in_channels=4):
#         self.img_json = os.path.join(data_dir, "images.json")
#         self.vae_dir = os.path.join(data_dir, "vae_feats.json")  # 如果没有可留空
#         self.graph_dim = graph_dim
#         self.in_channels = in_channels
#
#         self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.tif")))
#
#         # 目标域可能没有 vae_feats
#         self.has_vae = os.path.exists(self.vae_dir) and len(glob.glob(os.path.join(self.vae_dir, "*.npy"))) > 0
#         if self.has_vae:
#             self.vae_paths = {os.path.basename(p).split(".")[0]: p
#                               for p in glob.glob(os.path.join(self.vae_dir, "*.npy"))}
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.img_paths[idx]
#         target = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
#         target = torch.from_numpy(target).permute(2, 0, 1)  # [3,H,W]
#
#         # ---- 关键：目标域无语义 S ----
#         _, H, W = target.shape
#         cond_seg = torch.zeros((1, H, W), dtype=torch.float32)
#
#         # ---- 关键：目标域无 G（用空 graph 占位）----
#         cond_graph = make_empty_graph(self.graph_dim)
#
#         # ---- 关键：目标域无 instance（占位即可）----
#         cond_inst = torch.zeros((H, W), dtype=torch.long)
#
#         # ---- VAE feats：没有就返回 0（loop 里兜底）----
#         if self.has_vae:
#             key = os.path.basename(image_path).split(".")[0]
#             vae_feats = torch.from_numpy(np.load(self.vae_paths[key])).squeeze(0)
#         else:
#             vae_feats = torch.zeros((1,), dtype=torch.float32)  # dummy
#
#         # ---- R=文本 region caption ----
#         region_caption = os.path.basename(image_path).split("_")[0]
#
#         img_records = {
#             "img_path": image_path,
#             "captions": region_caption,
#             "domain": "target"
#         }
#
#         return dict(
#             image=target,
#             cond_seg=cond_seg,
#             cond_graph=cond_graph,
#             cond_inst=cond_inst,
#             vae_feats=vae_feats,
#             img_records=img_records,
#             flag="regions_graph"   # 屏蔽 semantic 分支
#         )


#dataset = M2ITargetRText(data_dir= r"E:\clipsamcd\zero_changedetection\LEVIR_CD")


# generate cd
# A:le
# import os, glob, json
#
# A = r"E:\clipsamcd\zero_changedetection\LEVIR_CD\vae_feats\A"
# B = r"E:\clipsamcd\zero_changedetection\LEVIR_CD\vae_feats\B"
# text_prompt = ["LEVIR_CD_A", "LEVIR_CD_B"]
#
# def list_imgs(folder, exts=("png", "jpg", "jpeg", "tif", "tiff","npy")):
#     paths = []
#     for e in exts:
#         paths += glob.glob(os.path.join(folder, f"*.{e}"))
#     # 排序保证稳定一致
#     paths = sorted(paths, key=lambda p: os.path.basename(p))
#     return paths
#
# pngs_A = list_imgs(A, exts=("npy",))
# pngs_B = list_imgs(B, exts=("npy",))
#
# print("A count:", len(pngs_A), "B count:", len(pngs_B))
#
# # 可选：检查文件名能否对齐
# names_A = [os.path.basename(p) for p in pngs_A]
# names_B = [os.path.basename(p) for p in pngs_B]
# if len(names_A) != len(names_B):
#     print("[WARN] A/B 数量不一致!")
# else:
#     mismatch = [(a,b) for a,b in zip(names_A, names_B) if a != b]
#     if mismatch:
#         print("[WARN] A/B 文件名存在不对应的项，前5个：", mismatch[:5])
#
# out_json = {
#     text_prompt[0]: pngs_A,
#     text_prompt[1]: pngs_B
# }
#
# save_path = r"E:\clipsamcd\zero_changedetection\LEVIR_CD\vae_feats.json"
# with open(save_path, "w", encoding="utf-8") as f:
#     json.dump(out_json, f, ensure_ascii=False, indent=2)
#
# print("saved to:", save_path)




# src_ds = M2ITargetRText(data_dir= r"C\LEVIR_CD")
#
# # 拿一个迭代器
# it = iter(src_ds)
#
# # 取第一个样本
# sample = next(it)
