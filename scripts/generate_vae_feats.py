import torch
import glob
import numpy as np
import cv2
from diffusers.models import AutoencoderKL
import os
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

if __name__ == "__main__":
    data_val_dir = r"F:\data\OpenEarthMap\Size_256\val"
    out_val_dir = r"F:\data\OpenEarthMap\Size_256\val\vae_feats"
    os.makedirs(out_val_dir, exist_ok=True)
    data_train_dir = r"F:\data\OpenEarthMap\Size_256\train"
    out_train_dir = r"F:\data\OpenEarthMap\Size_256\train\vae_feats"
    os.makedirs(out_train_dir, exist_ok=True)
    generate_vae_feats(data_val_dir, out_val_dir)
    print("done generating vae feats for val")
    generate_vae_feats(data_train_dir, out_train_dir)
    print("done generating vae feats for train")