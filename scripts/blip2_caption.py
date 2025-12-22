import os
import json
from typing import List
from PIL import Image
from pathlib import Path
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
class BLIP2Captioner:
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = None,
    ):
        """
        基于 BLIP2 的图文对齐模型，用于生成遥感影像描述。
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device


        base = Path(r"E:\hf_cache\models--Salesforce--blip2-opt-2.7b\snapshots")
        snap_dirs = list(base.iterdir())
        assert len(snap_dirs) > 0, "snapshots 目录是空的，说明模型还没下完。"

        local_dir = snap_dirs[0]  # 一般只有一个 hash 目录
        print("Use local dir:", local_dir)
        print(f"[BLIP2] loading model: {model_name} on {device} ...")
        self.processor = Blip2Processor.from_pretrained(local_dir, local_files_only=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            local_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def describe_image(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")

        # prompt = (
        #     "You are a remote sensing expert. "
        #     "Describe the input as a high-resolution satellite image. "
        #     "Follow this template: "
        #     "'a high-resolution satellite image of ... "
        #     "In this region, the main land cover types include ...'. "
        #     "When mentioning land cover types, prefer these terms: "
        #     "buildings, roads, trees, water, agriculture land, bareland,rangeland and developed space."
        # )
        prompt = (
            "Question: In this high-resolution satellite image, which of the following land "
            "cover types are present? Choose only from this list: "
            "buildings, roads, trees, water, agriculture land, bareland, rangeland, developed space. "
            "Answer using only the land cover types in English, separated by commas, "
            "without any extra words."
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.model.dtype == torch.float16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )
        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        print("raw answer:", caption)

        # 1) 得到 pixel_values + input_ids + attention_mask
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        # 2) 只把张量移到 device；保持各自 dtype 不变
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 若模型是 fp16，只把 pixel_values 转成 fp16
        if self.model.dtype == torch.float16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # 3) 调用 generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,  # 先贪心生成，稳定一些
        )
        print("generated_ids shape:", generated_ids.shape)  # 调试一下

        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        print(caption)
        return caption


def list_images(folder: str, exts: List[str] = None) -> List[str]:
    if exts is None:
        exts = ["png", "jpg", "jpeg", "tif", "tiff"]
    paths = []
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = fn.lower().split(".")[-1]
            if ext in exts:
                paths.append(os.path.join(root, fn))
    paths = sorted(paths)
    return paths


def run_folder(
    image_dir: str,
    out_json: str,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    device: str = None,
):
    """
    扫描一个影像目录，对每张图生成一条 caption，写成 json：
    [
      {"image_path": "...", "captions": "..."},
      ...
    ]
    后面可以直接作为 img_records 读入。
    """
    from transformers import pipeline
    captioner = pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf", device=0)

    #captioner = BLIP2Captioner(model_name=model_name, device=device)
    imgs = list_images(image_dir)

    print(f"[BLIP2] found {len(imgs)} images in {image_dir}")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    records = []
    for i, img_path in enumerate(imgs):
        print(f"[{i+1}/{len(imgs)}] captioning {img_path} ...")
        image = Image.open(img_path)
        text =captioner(
            image,
            prompt="USER: <image>\Describe the contents of the image?\nASSISTANT:",
            generate_kwargs={"max_new_tokens": 76},
        )[0]["generated_text"].split("ASSISTANT:")[-1].strip()

        records.append({
            "image_path": img_path,
            "captions": text,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[BLIP2] saved captions to {out_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="单张影像路径，如果只想测试一张图。")
    parser.add_argument("--folder", type=str, default=None,
                        help="影像目录，批量生成 caption。")
    parser.add_argument("--out_json", type=str, default="captions_blip2.json",
                        help="批量模式输出的 JSON 路径。")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.image is not None:
        cap = BLIP2Captioner(model_name=args.model_name, device=args.device)
        text = cap.describe_image(args.image)
        print("Generated caption:")
        print(text)
    elif args.folder is not None:
        run_folder(
            image_dir=args.folder,
            out_json=args.out_json,
            model_name=args.model_name,
            device=args.device,
        )
    else:
        print("请指定 --image 或 --folder 其中之一。")
