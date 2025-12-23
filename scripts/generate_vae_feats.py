import torch
import glob
import numpy as np
import cv2
from diffusers.models import AutoencoderKL
import os
import argparse
from  tqdm import tqdm

from torchvision import transforms

from PIL import Image

transform = transforms.Compose([

    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])


def generate_vae_feats(data_dir, outdir):
    vae = AutoencoderKL.from_pretrained(r"./pretrained").to("cuda").eval()

    img_paths = glob.glob(os.path.join(data_dir, "images/*.tif"))
    #lbl_paths = glob.glob(os.path.join(data_dir, "labels/*.tif"))


    for image_path in tqdm(img_paths):
        x = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)[:, :, :3]
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x =  Image.fromarray(x)
        x = transform(x).cuda().unsqueeze(0)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        save_name = os.path.basename(image_path).replace(".tif", ".npy")
        save_path = os.path.join(outdir, save_name)
        np.save(save_path, x.cpu().numpy())

def parse_args():
    parser = argparse.ArgumentParser("Generate VAE features (latents) for dataset splits")

    # 方式1：给 root，然后自动拼 train/val
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=r'Dataset root containing splits, e.g. F:\data\OpenEarthMap\Size_256'
    )

    # 方式2：分别指定 train/val 的目录（优先生效）
    parser.add_argument("--train_dir", type=str, default=None, help="Path to train split directory")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to val split directory")

    # 输出子目录名（默认在 split 目录下创建 vae_feats）
    parser.add_argument("--out_name", type=str, default="vae_feats", help="Output folder name under each split")

    # 要处理哪些 split
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help='Splits to process, e.g. train val')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 解析 train/val 目录
    train_dir = args.train_dir
    val_dir = args.val_dir

    if train_dir is None or val_dir is None:
        if args.data_root is None:
            raise ValueError(
                "Please provide either --data_root (recommended) or both --train_dir and --val_dir."
            )
        # 如果用户没分别指定，就从 root 推断
        if train_dir is None:
            train_dir = os.path.join(args.data_root, "train")
        if val_dir is None:
            val_dir = os.path.join(args.data_root, "val")

    # 逐 split 处理
    for split in args.splits:
        if split.lower() == "train":
            data_dir = train_dir
        elif split.lower() == "val":
            data_dir = val_dir
        else:
            # 允许你未来扩展 test 等
            if args.data_root is None:
                raise ValueError(f"Unknown split '{split}' without --data_root to resolve it.")
            data_dir = os.path.join(args.data_root, split)

        out_dir = os.path.join(data_dir, args.out_name)
        os.makedirs(out_dir, exist_ok=True)

        generate_vae_feats(data_dir, out_dir)
        print(f"done generating vae feats for {split} -> {out_dir}")