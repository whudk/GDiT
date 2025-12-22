import torch
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git
from PIL import Image
import os, glob, json
from  tqdm import tqdm
def build_region_prompt(dataset_name, class_names):
    """
    根据数据集名称和包含的类别，生成一条区域文本提示（prompt）。

    例如：
        dataset_name = "Guangdong-Farmland-Change"
        class_names  = ["building", "greenspace", "road"]
    返回：
        "a satellite image from the Guangdong-Farmland-Change dataset containing building, greenspace and road"
    """
    dataset_name = str(dataset_name)

    # 去重并去掉空字符串
    class_names = [c.strip() for c in class_names if c and c.strip()]
    class_names = list(dict.fromkeys(class_names))  # 保顺序去重

    if len(class_names) == 0:
        return f"a satellite image from the {dataset_name} dataset"

    if len(class_names) == 1:
        cls_str = class_names[0]
        return f"a satellite image from the {dataset_name} dataset containing {cls_str}"

    cls_body = ", ".join(class_names[:-1])
    cls_last = class_names[-1]
    cls_str = f"{cls_body} and {cls_last}"

    prompt = f"a satellite image from the {dataset_name} dataset containing {cls_str}"
    return prompt

def predict_classes_and_build_prompt(
    image_path: str,
    dataset_name: str,
    class_names,
    clip_model,
    preprocess,
    device="cuda",
    top_k=3,
    score_threshold=0.2,
):
    """
    用 CLIP 预测一张遥感影像包含哪些类别，并生成带“数据集名称+类别”的文本提示。

    参数：
        image         : PIL.Image，一张影像（例如 256x256）
        dataset_name  : str，数据集名称，例如 "Guangdong-Farmland-Change"
        class_names   : list[str]，候选类别名称，如 ["building", "greenspace", "water", "road"]
        clip_model    : 已加载的 CLIP 模型（clip.load(...) 返回的 model）
        preprocess    : CLIP 的预处理函数（clip.load(...) 返回的 preprocess）
        device        : "cuda" 或 "cpu"
        top_k         : 至多取前 top_k 个类别
        score_threshold : 相似度阈值（0~1，越高越严格）

    返回：
        selected_classes : list[str]，CLIP 判定的该影像可能包含的类别
        prompt           : str，一条最终的区域文本提示
    """

    clip_model.eval()
    device = torch.device(device)
    image = Image.open(image_path).convert("RGB")
    # 1. 预处理图像并编码
    with torch.no_grad():
        img_in = preprocess(image).unsqueeze(0).to(device)   # [1, 3, H, W]
        image_feat = clip_model.encode_image(img_in)         # [1, D]
        image_feat = F.normalize(image_feat, dim=-1)         # 归一化

    # 2. 为每个类别构造文本提示，并编码
    #    注意：这里的 prompt 主要给 CLIP 判断类别，所以不要加 dataset_name，
    #    用简单、通用的句式即可。
    text_prompts = [f"a satellite image containing {cls}" for cls in class_names]
    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts).to(device)     # [K, L]
        text_feats = clip_model.encode_text(text_tokens)         # [K, D]
        text_feats = F.normalize(text_feats, dim=-1)

    # 3. 计算图像与每个类别文本的余弦相似度
    #    image_feat: [1, D], text_feats: [K, D] -> sim: [K]
    sims = (image_feat @ text_feats.T).squeeze(0)                # [K]
    sims_cpu = sims.detach().cpu()

    # 4. 选出 top_k 个分数最高的类别，并应用阈值过滤
    K = len(class_names)
    top_k = min(top_k, K)
    # 按相似度排序，取前 top_k 的索引
    sorted_idx = torch.argsort(sims_cpu, descending=True)[:top_k]

    selected_classes = []
    for idx in sorted_idx:
        score = sims_cpu[idx].item()
        if score >= score_threshold:
            selected_classes.append(class_names[int(idx)])

    # 如果一个都没过阈值，就至少取分数最高的一个，防止空
    if len(selected_classes) == 0:
        best_idx = int(torch.argmax(sims_cpu).item())
        selected_classes = [class_names[best_idx]]

    # 5. 把选出来的类别拼成“数据集名+类别”的区域文本提示
    prompt = build_region_prompt(dataset_name, selected_classes)

    return selected_classes, prompt




# 1. 加载 CLIP 模型
clip_model, preprocess = clip.load("ViT-L/14", device="cuda")


# generate cd
# A:le
# import os, glob, json
#
A = r"E:\clipsamcd\A-LEVIR-CD\train\A"
B = r"E:\clipsamcd\A-LEVIR-CD\train\B"
text_prompt = ["LEVIR_CD_A", "LEVIR_CD_B"]

def list_imgs(folder, exts=("png", "jpg", "jpeg", "tif", "tiff","npy")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, f"*.{e}"))
    # 排序保证稳定一致
    paths = sorted(paths, key=lambda p: os.path.basename(p))
    return paths

pngs_A = list_imgs(A, exts=("png",))
pngs_B = list_imgs(B, exts=("png",))

print("A count:", len(pngs_A), "B count:", len(pngs_B))

# 简单检查对齐
names_A = [os.path.basename(p) for p in pngs_A]
names_B = [os.path.basename(p) for p in pngs_B]
if len(names_A) != len(names_B):
    print("[WARN] A/B 数量不一致!")
else:
    mismatch = [(a, b) for a, b in zip(names_A, names_B) if a != b]
    if mismatch:
        print("[WARN] A/B 文件名存在不对应的项，前5个：", mismatch[:5])

# 类别空间
class_names = [
    "Background","Agriculture land", "Building", "Water", "Tree",
    "Road", "Developed space", "Rangeland", "Bareland"
]

# 用文件夹名当“数据集名 / 域名”
name_A = os.path.basename(os.path.normpath(A))
name_B = os.path.basename(os.path.normpath(B))

data_A = []
for p in tqdm(pngs_A):
    sel_cls, prompt = predict_classes_and_build_prompt(
        image_path=p,
        dataset_name=text_prompt[0],
        class_names=class_names,
        clip_model=clip_model,
        preprocess=preprocess,
        device="cuda",
        top_k=4,
        score_threshold=0.2,  # 可自己调
    )
    data_A.append({
        "path": p,
        "classes": sel_cls,
        "prompt": prompt,
    })

data_B = []
for p in tqdm(pngs_B):
    sel_cls, prompt = predict_classes_and_build_prompt(
            image_path=p,
            dataset_name=text_prompt[1],
            class_names=class_names,
            clip_model=clip_model,
            preprocess=preprocess,
            device="cuda",
            top_k=4,
            score_threshold=0.2,  # 可自己调
    )



    data_B.append({
        "path": p,
        "classes": sel_cls,
        "prompt": prompt,
    })

out_json = {
    name_A: {
        "images": data_A,
        "class_names": class_names,
    },
    name_B: {
        "images": data_B,
        "class_names": class_names,
    }
}

save_path = r"E:\clipsamcd\zero_changedetection\LEVIR_CD\images.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2)

print("saved to:", save_path)


# 2. 定义数据集名和候选类别
#dataset_name = "LEVIR_CD_A"

# OEM_dict = {
#     0:'Background',
#     1:'Bareland',
#     2:'Rangeland',
#     3:'Developed space',
#     4:'Road',
#     5:'Tree',
#     6:'Water',
#     7:'Agriculture land',
#     8:'Building',
# }
# 3. 读入一张目标域影像（PIL）
# from PIL import Image
# img = Image.open(r"E:\clipsamcd\A-LEVIR-CD+\train\B\train_1_crop_768_0.png").convert("RGB")
#
# # 4. 预测类别并生成 prompt
# selected_classes, prompt = predict_classes_and_build_prompt(
#     image=img,
#     dataset_name=dataset_name,
#     class_names=class_names,
#     clip_model=clip_model,
#     preprocess=preprocess,
#     device="cuda",
#     top_k=3,
#     score_threshold=0.22,  # 可自己调
# )
#
# print("Predicted classes:", selected_classes)
# print("Prompt:", prompt)
