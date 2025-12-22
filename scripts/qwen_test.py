import os
import json
import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache"
os.environ["HF_HUB_CACHE"] = r"E:\hf_cache\hub"
from PIL import Image

import torch
from transformers import AutoProcessor
try:
    # transformers>=4.50-ish: Qwen2.5-VL class name
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None

from qwen_vl_utils import process_vision_info
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"




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

TEMPLATE = "a satellite image from the [{src}], containing {cls_str}"


def list_images(image_dir: str,
                exts=(".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"),
                recursive: bool = False) -> List[str]:
    p = Path(image_dir)
    files = list(p.rglob("*")) if recursive else list(p.iterdir())
    imgs = [str(x) for x in files if x.is_file() and x.suffix.lower() in exts]
    imgs.sort()
    return imgs


def format_prompt_for_vlm() -> str:
    label_lines = "\n".join([f"{k}: {v}" for k, v in OEM_dict.items()])
    return f"""
You are analyzing a satellite image and must identify which land-cover classes are present.
Choose ONLY from the following label set (id:name):
{label_lines}

Return STRICT JSON only (no markdown, no explanation):
{{
  "class_ids": [int, ...],
  "cls_str": "..."
}}

Rules:
- class_ids must be sorted ascending and unique.
- cls_str must be a natural English list using ONLY the provided class names exactly.
- Include 2-5 major classes if possible; avoid "Background" unless nothing else is visible.
- If unsure about a class, do NOT include it.
""".strip()


def _extract_json(text: str) -> dict:
    """
    从模型输出里抠出第一个 JSON 对象（避免模型在前后加解释）。
    """
    s = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError(f"No JSON found. Raw output head: {s[:160]}")
    return json.loads(m.group(0))


def validate_and_fix_cls(data: dict) -> Tuple[List[int], str]:
    # 校验 ids
    ids = data.get("class_ids", [])
    try:
        ids = sorted(set(int(x) for x in ids))
    except Exception:
        ids = []

    ids = [i for i in ids if i in OEM_dict]

    # 校验 cls_str：只允许 OEM_dict 中的类名
    cls_str = (data.get("cls_str") or "").strip()
    valid_names = set(OEM_dict.values())

    parts = [p.strip() for p in cls_str.replace(" and ", ", ").split(",") if p.strip()]
    if (not cls_str) or any(p not in valid_names for p in parts):
        # 如果模型给的 cls_str 不靠谱，则用 ids 重建
        names = [OEM_dict[i] for i in ids]
        if len(names) == 0:
            cls_str = OEM_dict[0]
        elif len(names) == 1:
            cls_str = names[0]
        elif len(names) == 2:
            cls_str = f"{names[0]} and {names[1]}"
        else:
            cls_str = ", ".join(names[:-1]) + f", and {names[-1]}"

    return ids, cls_str


def _ensure_supported_image_path(img_path: str) -> Tuple[str, Optional[str]]:
    """
    Qwen 的 vision loader 对 tiff/GeoTIFF 不一定稳定。
    这里遇到 .tif/.tiff 就转成临时 JPEG，返回 (path_for_vlm, tmp_path_to_cleanup)
    """
    suffix = Path(img_path).suffix.lower()
    if suffix not in (".tif", ".tiff"):
        return img_path, None

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        im.save(tmp_path, format="JPEG", quality=95)
    return tmp_path, tmp_path


def build_qwen(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: int = 0):
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError(
            "Cannot import Qwen2_5_VLForConditionalGeneration. "
            "Please upgrade transformers (recommended: install from github)."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install CUDA PyTorch or use CPU mode.")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",                  # ✅ 替代 torch_dtype
        device_map=f"cuda:{device}",   # ✅ 强制放到指定 GPU
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def qwen_classify_oem(model, processor, img_path: str, sys_prompt: str,
                      max_new_tokens: int = 128) -> Tuple[List[int], str]:
    use_path, tmp_to_cleanup = _ensure_supported_image_path(img_path)
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": use_path},
                {"type": "text", "text": sys_prompt},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        out_text = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        data = _extract_json(out_text)
        return validate_and_fix_cls(data)

    finally:
        if tmp_to_cleanup and os.path.exists(tmp_to_cleanup):
            try:
                os.remove(tmp_to_cleanup)
            except Exception:
                pass


def run_folder(
    image_dir: str,
    out_json: str,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    src: str = "OpenEarthMap",
    recursive: bool = False,
    max_images: Optional[int] = 10,
):
    """
    将原 gemini 版替换为 Qwen2.5-VL（本地开源）：
    扫描影像目录 → 逐张图识别 OEM 类集合 → 拼 caption → 返回 records
    """
    imgs = list_images(image_dir, recursive=recursive)
    if max_images is not None:
        imgs = imgs[:max_images]

    print(f"[Qwen2.5-VL] found {len(imgs)} images in {image_dir}")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # ---- 只加载一次模型（缓存到函数属性）----
    cache = getattr(run_folder, "_cache", None)
    if cache is None or cache.get("model_name") != model_name:
        model, processor = build_qwen(model_id=model_name)


        run_folder._cache = {"model_name": model_name, "model": model, "processor": processor}
    else:
        model, processor = cache["model"], cache["processor"]

    sys_prompt = format_prompt_for_vlm()

    records = []
    for i, img_path in enumerate(imgs):
        print(f"[{i+1}/{len(imgs)}] captioning {img_path} ...")
        try:
            class_ids, cls_str = qwen_classify_oem(model, processor, img_path, sys_prompt)
            text = TEMPLATE.format(src=src, cls_str=cls_str)
        except Exception as e:
            text = f"[ERROR] {type(e).__name__}: {e}"

        records.append({"image_path": img_path, "captions": text})

    return records


if __name__ == "__main__":
    # 你的原调用逻辑保持不变
    image_dir_A = r"E:\clipsamcd\WHU-CD\train\A"
    image_dir_B = r"E:\clipsamcd\WHU-CD\train\B"
    src = ["WHUCD_A", "WHUCD_B"]
    data_A = run_folder(
        image_dir=image_dir_A,
        out_json=r"E:\clipsamcd\zero_changedetection\whucd.json",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        src="WHUCD_A",
        recursive=False,
        max_images=None,
    )

    data_B = run_folder(
        image_dir=image_dir_B,
        out_json=r"E:\clipsamcd\zero_changedetection\whucd.json",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        src="WHUCD_B",
        recursive=False,
        max_images=None,
    )

    out_json = {
        "WHUCD_A": data_A,
        "WHUCD_B": data_B,
    }

    save_path = r"E:\clipsamcd\zero_changedetection\WHU_CD\images.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"[Qwen2.5-VL] saved merged json to {save_path}")
