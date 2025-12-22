import os
import json
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from google import genai
from google.genai import types


OEM_dict = {
    0:'Background',
    1:'Bareland',
    2:'Rangeland',
    3:'Developed space',
    4:'Road',
    5:'Tree',
    6:'Water',
    7:'Agriculture land',
    8:'Building',
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


def pil_to_jpeg_bytes(img_path: str, quality: int = 95) -> bytes:
    # GeoTIFF / 灰度 / RGBA 统一转 RGB，再编码 JPEG
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def format_prompt_for_gemini() -> str:
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

    # 如果 Gemini 给的 cls_str 不靠谱，则用 ids 重建
    if not cls_str or any(name not in OEM_dict.values() for name in cls_str.replace(" and ", ", ").split(", ")):
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


def run_folder(
    image_dir: str,
    out_json: str,
    model_name: str = "gemini-2.5-flash",
    src: str = "OpenEarthMap",
    recursive: bool = False

):
    """
    扫描一个影像目录，对每张图生成一条 caption，写成 json：
    [
      {"image_path": "...", "captions": "..."},
      ...
    ]
    caption 格式：a satellite image from the [src], containing {cls_str}
    其中 cls_str 由 Gemini 基于 OEM_dict 自动生成。
    """

    imgs = list_images(image_dir, recursive=recursive)[:10]
    print(f"[Gemini] found {len(imgs)} images in {image_dir}")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # client：建议显式传 key，避免环境变量没生效/被 GOOGLE_API_KEY 抢优先级
    api_key = "AIzaSyA502PHYqm_Ta5SvCwotWp7Xv-sznRsRyE"
    #api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise RuntimeError("Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) or pass api_key=...")

    client = genai.Client(api_key=api_key)
    sys_prompt = format_prompt_for_gemini()

    records = []
    for i, img_path in enumerate(imgs):
        print(f"[{i+1}/{len(imgs)}] captioning {img_path} ...")
        try:
            img_bytes = pil_to_jpeg_bytes(img_path)
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    sys_prompt,
                ],
            )
            raw = (resp.text or "").strip()
            data = json.loads(raw)

            class_ids, cls_str = validate_and_fix_cls(data)
            text = TEMPLATE.format(src=src, cls_str=cls_str)

        except Exception as e:
            # 不中断：记录错误方便回查
            text = f"[ERROR] {type(e).__name__}: {e}"

        records.append({
            "image_path": img_path,
            "captions": text,
        })
    return records

    # with open(out_json, "w", encoding="utf-8") as f:
    #     json.dump(records, f, ensure_ascii=False, indent=2)
    # print(f"[Gemini] saved captions to {out_json}")


if __name__ == "__main__":
    image_dir = r"E:\clipsamcd\A-LEVIR-CD\train\B"
    out_json  = r"E:\clipsamcd\zero_changedetection\test.json"
    image_dir_A = r"E:\clipsamcd\A-LEVIR-CD\train\A"
    image_dir_B = r"E:\clipsamcd\A-LEVIR-CD\train\B"
    name_A = os.path.basename(os.path.normpath(image_dir_A))
    name_B = os.path.basename(os.path.normpath(image_dir_B))



    data_A = run_folder(
        image_dir=image_dir_A,
        out_json=out_json,
        model_name="gemini-2.5-flash",
        src="LEVIR-CD-A",     # 模板里的 [src]
        recursive=False
    )

    data_B = run_folder(
        image_dir=image_dir_B,
        out_json=out_json,
        model_name="gemini-2.5-flash",
        src="LEVIR-CD-B",  # 模板里的 [src]
        recursive=False
    )

    out_json = {
        "LEVIR-CD-A": data_A,
        "LEVIR-CD-B": data_B,
    }

    save_path = r"E:\clipsamcd\zero_changedetection\LEVIR_CD\images.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)