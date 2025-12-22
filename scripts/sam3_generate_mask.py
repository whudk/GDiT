import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import re, json
from pathlib import Path


from tqdm import  tqdm
OEM_dict = {
    0: 'Background',
    1: 'Bareland',
    2: 'Rangeland',
    3: 'Developed space',
    4: 'Road',
    5: 'Tree',
    6: 'Water',
    7: 'Agriculture land',
    8: 'Building',
}
NAME2ID = {v.lower(): k for k, v in OEM_dict.items()}
# 每类给多个概念短语（SAM3 是 concept prompt）
PROMPTS = {
    1: ["bare land", "barren land", "bare soil", "sand", "dirt"],
    2: ["rangeland", "grassland", "pasture", "meadow"],
    3: ["develope area", "built-up area", "urban area", "impervious surface", "yard"],
    4: ["road"],
    #5: ["trees", "forest", "woodland"],
    5: ["trees"],
    6: ["water"],
    7: ["agriculture land", "farmland", "cropland", "agricultural field"],
    8: ["building"],
}
# 低优先级 -> 高优先级（后写覆盖前写）
LOW_TO_HIGH = [3, 1, 2, 7,  6, 5, 4, 8]  # Developed 最低，Road 最高（你可按需调整）

# 每类阈值（可调）
THR = {
    1: 0.35,  # Bareland
    2: 0.35,  # Rangeland
    3: 0.35,  # Developed space（设高点，防抢占）
    4: 0.5,  # Road（设低点，尽量别漏）
    5: 0.35,  # Tree
    6: 0.35,  # Water
    7: 0.35,  # Agriculture land
    8: 0.4,  # Building
}
BG_THR = 0.20   # 全类别最大分数 < BG_THR -> 背景
MARGIN = 0.00   # 覆盖优势幅度（0=纯优先级覆盖；0.05更谨慎）
color_palette = {
    "0-1": [0, 0, 0],
    "1-2": [128, 0, 0],
    "2-3": [0, 255, 36],
    "3-4": [148, 148, 148],
    "4-5": [255, 255, 255],
    "5-6": [34, 97, 38],
    "6-7": [0, 69, 255],
    "7-8": [75, 181, 73],
    "8-9": [222, 31, 7],
}

def save_color_label(label_np: np.ndarray, out_color_path: str, palette_dict=color_palette):
    h, w = label_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for k in range(0, 9):
        rgb = palette_dict[f"{k}-{k+1}"]
        color[label_np == k] = rgb
    Image.fromarray(color, mode="RGB").save(out_color_path)


# --------------------------
# 4) 从 caption 解析类别
#    支持: "containing A, B, C" 或任意位置出现 OEM 类名
# --------------------------
def parse_classes_from_caption(caption: str):
    s = (caption or "").lower()

    # 先尝试截取 containing 后面的列表
    m = re.search(r"containing\s+(.+)$", s)
    if m:
        tail = m.group(1)
    else:
        tail = s

    # 用 OEM 名称做匹配（允许逗号分隔、任意顺序）
    found = set()
    for name_lc, cid in NAME2ID.items():
        if cid == 0:
            continue
        # 词边界：避免 "tree" 匹配到 "street" 之类（简单保护）
        if re.search(rf"\b{re.escape(name_lc)}\b", tail):
            found.add(cid)

    # 兜底：如果 caption 里直接写了 "Building" "Road" 等，仍可匹配
    if not found:
        for name_lc, cid in NAME2ID.items():
            if cid == 0:
                continue
            if re.search(rf"\b{re.escape(name_lc)}\b", s):
                found.add(cid)

    return sorted(found)


# --------------------------
# 5) SAM3：单类多 prompt 取 max，得到 [H,W] 概率图
# --------------------------
@torch.no_grad()
def sam3_score_map_from_instances(
    model, processor, image_pil, text, device,
    threshold=0.05,        # 遥感场景通常要比0.5低很多
    mask_threshold=0.30,   # 0.2~0.5试
    fuse="max"             # "max" or "sum"
):
    W, H = image_pil.size
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs["original_sizes"].tolist()
    )[0]

    # 没有实例就返回全0
    if len(results.get("masks", [])) == 0:
        return torch.zeros((H, W), dtype=torch.float32, device=device)

    masks = results["masks"]   # (N,H,W)
    scores = results["scores"] # (N,)

    if isinstance(masks, np.ndarray):
        masks_t = torch.from_numpy(masks).to(device)
    else:
        masks_t = masks.to(device)

    masks_t = masks_t.float()
    scores_t = torch.as_tensor(scores, device=device, dtype=torch.float32).view(-1, 1, 1)

    if fuse == "sum":
        smap = (masks_t * scores_t).sum(dim=0).clamp(0, 1)
    else:
        smap = (masks_t * scores_t).max(dim=0).values

    return smap


@torch.no_grad()
def sam3_semantic_score_for_prompts(model, processor, image_pil, prompts, device,
                                    threshold=0.05, mask_threshold=0.30, fuse="max"):
    best = None
    for p in prompts:
        smap = sam3_score_map_from_instances(
            model, processor, image_pil, text=p, device=device,
            threshold=threshold, mask_threshold=mask_threshold, fuse=fuse
        )
        best = smap if best is None else torch.maximum(best, smap)

    if best is None:
        W, H = image_pil.size
        best = torch.zeros((H, W), dtype=torch.float32, device=device)
    return best  # [H,W]

# --------------------------
# 6) 单图：只对 caption 中出现的类别做分割，优先级融合输出 0..8
# --------------------------
@torch.no_grad()
def infer_one_image(model, processor, image_path: str, caption: str, device: str,
                    inst_threshold=0.01, inst_mask_threshold=0.20):
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    active_cids = parse_classes_from_caption(caption)
    if len(active_cids) == 0:
        label = np.zeros((H, W), dtype=np.uint8)
        return label, []

    # 计算 active 类别的 score map（来自 post_process_instance_segmentation）
    score_by_cid = {}
    for cid in active_cids:
        smap = sam3_semantic_score_for_prompts(
            model, processor, image, PROMPTS[cid], device,
            threshold=inst_threshold, mask_threshold=inst_mask_threshold, fuse="max"
        )
        score_by_cid[cid] = smap

    # all_max 用于背景判定（只在 active 类里取 max）
    all_max = torch.stack(list(score_by_cid.values()), dim=0).max(dim=0).values  # [H,W]

    label_t = torch.zeros((H, W), dtype=torch.uint8, device=device)
    conf_t  = torch.zeros((H, W), dtype=torch.float32, device=device)

    # 按全局优先级，但只处理 active_cids
    for cid in [c for c in LOW_TO_HIGH if c in active_cids]:
        smap = score_by_cid[cid]
        thr = THR.get(cid, 0.35)
        if MARGIN > 0:
            m = (smap >= thr) & (smap >= conf_t + MARGIN)
        else:
            m = (smap >= thr)

        label_t[m] = cid
        conf_t[m] = smap[m]

    label_t[all_max < BG_THR] = 0

    label = label_t.detach().cpu().numpy().astype(np.uint8)
    return label, active_cids

def  predict_one_image(model_dir: str,img_path, caption, out_root, device: str = None):


    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 模型只加载一次
    model = Sam3Model.from_pretrained(model_dir).to(device)
    processor = Sam3Processor.from_pretrained(model_dir)


    # 输出路径
    stem = Path(img_path).stem
    out_label = out_root / "label" / f"{stem}.png"
    out_color = out_root / "color" / f"{stem}_color.png"

    (out_root / "label").mkdir(parents=True, exist_ok=True)
    (out_root / "color").mkdir(parents=True, exist_ok=True)


    label_np, active = infer_one_image(model, processor, img_path, caption, device)

    Image.fromarray(label_np, mode="L").save(out_label)
    save_color_label(label_np, str(out_color))




# --------------------------
# 7) 批处理 images.json
# --------------------------
def batch_run(
    json_path: str,
    model_dir: str,
    out_root: str,
    write_color: bool = True,  # 是否保存彩色图
    write_index: bool = True,  # 是否输出 pseudo_index.json
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 模型只加载一次
    model = Sam3Model.from_pretrained(model_dir).to(device)
    processor = Sam3Processor.from_pretrained(model_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 允许顶层是 dict: {dataset_name: [items...]}
    # 或直接 list: [items...]
    if isinstance(data, list):
        datasets = {"default": data}
    else:
        datasets = data

    summary = []
    index = {ds_name: [] for ds_name in datasets.keys()}
    for ds_name, items in datasets.items():
        ds_out = out_root / ds_name
        (ds_out / "label").mkdir(parents=True, exist_ok=True)
        (ds_out / "color").mkdir(parents=True, exist_ok=True)
       # items = items[:100]
        for it in tqdm(items, desc=f"Processing {ds_name}"):

            img_path = it.get("image_path", "")
            caption = it.get("captions", "")

            # 输出路径
            stem = Path(img_path).stem
            out_label = ds_out / "label" / f"{stem}.png"
            out_color = ds_out / "color" / f"{stem}_color.png"

            label_np, active = infer_one_image(model, processor, img_path, caption, device)

            Image.fromarray(label_np, mode="L").save(out_label)
            if write_color:
                save_color_label(label_np, str(out_color))
            it["pseudo_label"] = str(out_label)
            if write_color:
                it["pseudo_color"] = str(out_color)

            ordered = {
                "image_path": it.get("image_path", ""),
                "pseudo_label": it.get("pseudo_label", "")
            }
            if write_color:
                ordered["pseudo_color"] = it.get("pseudo_color", "")

            # 其余字段保持原样（通常包含 captions）
            for k, v in it.items():
                if k not in ordered:
                    ordered[k] = v

            it.clear()
            it.update(ordered)
            # =====================================================================

            if write_index:
                index[ds_name].append({
                    "image_path": img_path,
                    "pseudo_label": str(out_label),
                    "pseudo_color": str(out_color) if write_color else "",
                    "caption": caption,
                })



    # 保存一份结果索引，方便你后续训练直接读取
    if write_index:
        out_index = out_root / "pseudo_index.json"
        with open(out_index, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        print(f"Index saved to: {out_index}")
    print(f"Done. index saved to: {out_index}")


if __name__ == "__main__":
    batch_run(
        json_path=r"E:\clipsamcd\zero_changedetection\WHU_CD\images.json",
        model_dir=r"E:\Dengkai\sam3",
        out_root=r"E:\clipsamcd\zero_changedetection\WHU_CD\pseudo_seg",
        device="cuda"
    )
    # img_path = r"E:\clipsamcd\WHU-CD\train\A\train_1038.png"
    #
    # captions =  "a satellite image from the [WHUCD_A], containing Developed space"
    # predict_one_image(
    #     img_path  = img_path,
    #     caption = captions,
    #     model_dir=r"E:\Dengkai\sam3",
    #     out_root=r"E:\clipsamcd\zero_changedetection\LEVIR_CD\pseudo_sam3",
    #     device="cuda"
    #
    # )