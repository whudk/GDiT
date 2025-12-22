# MIT License

# Copyright (c) [2023] [Anima-Lab]

# This code is adapted from https://github.com/NVlabs/edm/blob/main/training/loss.py. 
# The original code is licensed under a Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License, which is can be found at licenses/LICENSE_EDM.txt. 

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import vae

from utils import *
from train_utils.helper import unwrap_model
import matplotlib.pyplot as plt

# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def show_noised(self, img, noised, vae):
        def recur_decode(z):
            try:
                return vae.decode(z / 0.18215).sample
            except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
                assert z.shape[2] % 2 == 0
                z1, z2 = z.tensor_split(2)
                return torch.cat([recur_decode(z1), recur_decode(z2)])

        noised_img = recur_decode(noised)
        ori_img = recur_decode(img)


        # Scale and clip the noised image for valid display
        denoised_img = torch.clamp(127.5 * noised_img + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",dtype=torch.uint8).numpy()
        original_img = torch.clamp(127.5 * ori_img + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        num_images = denoised_img.shape[0]  # Number of images in the batch
        fig, axes = plt.subplots(num_images, 2, figsize=(12, 6 * num_images))  # Create a grid of images

        # If there is only one image, axes will be a 1D array, handle it gracefully
        if num_images == 1:
            axes = axes.reshape(1, 2)

        for i in range(num_images):
            # Original image
            axes[i, 0].imshow(original_img[i])
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f"Original Image {i + 1}")

            # Noised image
            axes[i, 1].imshow(denoised_img[i])
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f"Denoised Image {i + 1}")

        plt.tight_layout()  # Adjust spacing between subplots for better appearance
        plt.savefig(r'D:\dengkai\code\MaskDiT-master\results\fid\output_images_all.png', dpi=300,
                    bbox_inches='tight')  # Save all images
        #plt.show()  # Optionally, show the plot if you want to display it interactively

    def __call__(self, net,
                 images,
                 seg = None,
                 labels=None,
                 mask_ratio=0,
                 mae_loss_coef=0,
                 feat=None,
                 augment_pipe=None,
                 graph = None,
                 train_steps = 0,
                 class_ids = None,
                 text_context =None,
                 instance_map = None,
                 return_state  = True,
                 **kwargs

                 ):


        #show n

        # rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # n = torch.randn_like(y) * sigma
        #
        # losses = dict()
        # model_out = net(y + n, sigma, labels, seg, mask_ratio= mask_ratio, mask_dict=None, feat=feat, graph = graph, context = text_context, class_ids = class_ids, instance_map = instance_map)
        # D_yn = model_out['x']
        # loss = weight * ((D_yn - y) ** 2)
        # # eps_pred = (x_noisy - x_clean_hat) / sigma
        # eps_pred = ((y + n) - D_yn) / sigma  # 形状 [B, C, H, W]
        #
        # losses["total"] = loss
        # losses["eps_pred"] = eps_pred
        #
        # return losses

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        x_noisy = y + n

        model_out = net(x_noisy, sigma, labels, seg,
                        mask_ratio=mask_ratio, mask_dict=None,
                        feat=feat, graph=graph, context=text_context,
                        class_ids=class_ids, instance_map=instance_map)
        D_yn = model_out['x']
        loss = weight * ((D_yn - y) ** 2)

        eps_pred = (x_noisy - D_yn) / sigma

        losses = dict()
        losses["total"] = loss
        losses["eps_pred"] = eps_pred

        if return_state:
            losses["y"] = y
            losses["sigma"] = sigma
            losses["n"] = n
            losses["x_noisy"] = x_noisy

        return losses


        # # if train_steps  % 100 == 0:
        # #self.show_noised(latents, D_yn, kwargs["vae"])
        # assert D_yn.shape == y.shape
        # losses = dict()
        #
        # loss = weight * ((D_yn - y) ** 2)  # (N, C, H, W)
        # loss = mean_flat(loss)  # (N)
        # losses["total"] = loss
        # losses["mae"] = loss * 0.0
        # return losses


# class EDMLoss:
#     def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
#         self.P_mean = P_mean
#         self.P_std = P_std
#         self.sigma_data = sigma_data
#
#     def __call__(self, net, images, labels=None, augment_pipe=None):
#         rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
#         sigma = (rnd_normal * self.P_std + self.P_mean).exp()
#         weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
#         y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
#         n = torch.randn_like(y) * sigma
#         D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
#         loss = weight * ((D_yn - y) ** 2)
#         return loss

# ----------------------------------------------------------------------------


Losses = {
    'edm': EDMLoss
}


# ----------------------------------------------------------------------------

def patchify(imgs, patch_size=2, num_channels=4):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p, c = patch_size, num_channels
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x


def mae_loss(net, target, pred, mask, norm_pix_loss=True):
    target = patchify(target, net.model.patch_size, net.model.out_channels)
    pred = patchify(pred, net.model.patch_size, net.model.out_channels)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)  # mean loss on removed patches, (N)
    assert loss.ndim == 1
    return loss
