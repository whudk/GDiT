# MIT License

# Copyright (c) [2023] [Anima-Lab]


# This code is adapted from https://github.com/NVlabs/edm/blob/main/generate.py. 
# The original code is licensed under a Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License, which is can be found at licenses/LICENSE_EDM.txt. 

import argparse
import os
import random

import PIL.Image
import lmdb
import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from tqdm import tqdm

from models.GDiT import Precond_models, DiT_models
from utils import *

import platform
from train_utils.datasets  import M2IBase

from torch.utils.data.distributed import DistributedSampler
from train_utils.helper import get_mask_ratio_fn, get_one_hot,requires_grad, update_ema, unwrap_model

from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader
# ----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.color_visualizer import apply_color_palette_torch
from torchvision.utils import save_image

import cv2
from omegaconf import OmegaConf
from train_utils.helper import _contrust_graph,extract_bboxes
from train_utils.helper import collate_graph
from train_utils.encoders import StabilityVAEEncoder


def preprocess_input(x_cond, nc):
    bs, _, h, w = x_cond.size()

    input_label = torch.FloatTensor(bs, nc, h, w).zero_().to(x_cond.device)
    input_semantics = input_label.scatter_(1, x_cond.long(), 1.0)



    x_cond = input_semantics

    return x_cond

def proprocess_simcond(memory, cluster, nc):
    select_path = random.choice(memory[cluster[0]])
    sim_img = cv2.imread(select_path,cv2.IMREAD_GRAYSCALE)
    sim_img = torch.from_numpy(sim_img).unsqueeze(0).unsqueeze(0)
    sim_cond = preprocess_input(sim_img, nc)
    return sim_cond

def _setup_process_group(args):
    from datetime import datetime
    local_rank = args.local_rank
    print('local_rank:{}'.format(local_rank))
    current_dir = os.path.join(os.getcwd(),"ddp")
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


    if (platform.system() == "Windows"):
        backends = 'gloo'
        init_method = f"file:///{os.path.join(current_dir, 'ddp_{}'.format(current_time))}"
    else:
        backends = 'nccl'
        init_method = f"file://{os.path.join(current_dir, 'ddp_{}'.format(current_time))}"
    torch.cuda.set_device(local_rank)

    print("use backends:{}".format(backends))

    torch.distributed.init_process_group(
        backend = backends,
        init_method=init_method,
        rank=local_rank,
        world_size =  len(args.gpu)
    )
    torch.distributed.barrier()
    print("load distributed success".format(backends))
import matplotlib.pyplot as plt


# def recur_decode(z, vae):
#     try:
#         return vae.decode(z / 0.18215).sample
#     except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
#         assert z.shape[2] % 2 == 0
#         z1, z2 = z.tensor_split(2)
#         return torch.cat([recur_decode(z1), recur_decode(z2)])
# def resample(noised, vae):
#
#
#     noised_img = recur_decode(noised, vae)
#
#
#     # Scale and clip the noised image for valid display
#     denoised_img = torch.clamp(127.5 * noised_img + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",
#                                                                                           dtype=torch.uint8).numpy()
#
#     return denoised_img


def edm_sampler(
        net, latents, seg = None, class_labels=None, cfg_scale=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, graph =None, vae = None,text_context = None, instance_map = None, feat=None,flag = 'regoins_graph_sem'
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
    # Create a figure for visualizing the denoised images at each step
    #fig, axes = plt.subplots(1, num_steps, figsize=(20, 5))  # Create subplots to show each denoised image

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat.float(), t_hat, class_labels, seg, cfg_scale, feat=feat, graph = graph, text_context = text_context, instance_map = instance_map, flag = flag)['x'].to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next.float(), t_next, class_labels, seg, cfg_scale, feat=feat, graph = graph, text_context = text_context, instance_map = instance_map, flag = flag)['x'].to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            #show_denoised(x_next.float(), vae)

    return x_next





def edm_sampler_two_stage(
        net,
        latents,
        seg=None,
        class_labels=None,
        cfg_scale=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
        graph=None,

        text_context=None,
        instance_map=None,
        feat=None,
        flag='regions_graph_sem',   # 默认行为（不开两段式）
        two_stage=True,             # ★ 是否启用“两段式：regions -> regoins_graph_sem”
        stage1_frac=0.5,            # ★ 前半段步数比例
        stage1_cfg_scale=None,      # ★ 前半段的 cfg_scale（不设就用 cfg_scale）
        stage2_cfg_scale=None,      # ★ 后半段的 cfg_scale（不设就用 cfg_scale）
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # 两段式拆分索引
    if two_stage:
        split_idx = int(num_steps * stage1_frac)
        split_idx = max(1, min(num_steps - 1, split_idx))
    else:
        split_idx = 0   # 不启用两段式，所有步都走原 flag

    if stage1_cfg_scale is None:
        stage1_cfg_scale = cfg_scale
    if stage2_cfg_scale is None:
        stage2_cfg_scale = cfg_scale

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0,...,N-1
        x_cur = x_next

        # ===== 根据阶段选择条件 =====
        if two_stage and i < split_idx:
            # 前半段：只用 regions 文本，不用 seg / graph
            cur_flag = "regions"
            cur_class_labels = class_labels      # 仍然用区域文本 embedding
            cur_seg = seg
            cur_graph = graph
            cur_cfg_scale = stage1_cfg_scale     # 你可以设小一点，比如 1.0
        elif two_stage:
            # 后半段：用完整 regoins_graph_sem
            cur_flag = flag
            cur_class_labels = class_labels
            cur_seg = seg
            cur_graph = graph
            cur_cfg_scale = stage2_cfg_scale
        else:
            # 不两段式：保持原 flag + 条件
            cur_flag = flag
            cur_class_labels = class_labels
            cur_seg = seg
            cur_graph = graph
            cur_cfg_scale = cfg_scale

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(
            x_hat.float(),
            t_hat,
            cur_class_labels,
            cur_seg,
            cur_cfg_scale,
            feat=feat,                    # ★ style/域向量，全程保留
            graph=cur_graph,
            text_context=text_context,
            instance_map=instance_map,
            flag=cur_flag,
        )['x'].to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Heun's correction.
        if i < num_steps - 1:
            denoised = net(
                x_next.float(),
                t_next,
                cur_class_labels,
                cur_seg,
                cur_cfg_scale,
                feat=feat,
                graph=cur_graph,
                text_context=text_context,
                instance_map=instance_map,
                flag=cur_flag,
            )['x'].to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next




# ----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
        net, latents, class_labels=None, cfg_scale=None, feat=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=None, sigma_max=None, rho=7,
        solver='heun', discretization='edm', schedule='linear', scaling='none',
        epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,graph =None, vae = None,text_context = None, instance_map = None,seg = None
):



    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']



    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (
                sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # rnd_normal = torch.randn([latents.shape[0], 1, 1, 1], device=latents.device)
    # sigma = (rnd_normal * 1.2 -1.2).exp()
    #
    #
    # n = latents * sigma
    #
    # model_out = net(n, sigma, None, seg,  feat=feat, graph=graph,context=text_context,  instance_map=instance_map)
    #return  model_out["x"]
    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(
            t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        ##denoised = net(x_hat.float() / s(t_hat), sigma(t_hat), class_labels, cfg_scale, feat=feat)['x'].to(torch.float64)
        denoised = net(x_hat.float(), t_hat, class_labels, seg, cfg_scale, feat=feat, graph=graph, text_context=text_context,instance_map=instance_map)['x'].to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(
            t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            #denoised = net(x_prime.float() / s(t_prime), sigma(t_prime), class_labels, cfg_scale, feat=feat)['x'].to(torch.float64)
            denoised = net(x_prime.float() / s(t_prime), sigma(t_prime), class_labels, seg, cfg_scale, feat=feat, graph=graph, text_context=text_context,instance_map=instance_map)['x'].to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(
                t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next


# ----------------------------------------------------------------------------

def retrieve_n_features(batch_size, feat_path, feat_dim, num_classes, device, split='train', sample_mode='rand_full'):
    env = lmdb.open(os.path.join(feat_path, split), readonly=True, lock=False, create=False)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
        if sample_mode == 'rand_full':
            image_ids = random.sample(range(length // 2), batch_size)
            image_ids_y = image_ids
        elif sample_mode == 'rand_repeat':
            image_ids = random.sample(range(length // 2), 1) * batch_size
            image_ids_y = image_ids
        elif sample_mode == 'rand_y':
            image_ids = random.sample(range(length // 2), 1) * batch_size
            image_ids_y = random.sample(range(length // 2), batch_size)
        else:
            raise NotImplementedError
        features, labels = [], []
        for image_id, image_id_y in zip(image_ids, image_ids_y):
            feat_bytes = txn.get(f'feat-{str(image_id)}'.encode('utf-8'))
            y_bytes = txn.get(f'y-{str(image_id_y)}'.encode('utf-8'))
            feat = np.frombuffer(feat_bytes, dtype=np.float32).reshape([feat_dim]).copy()
            y = int(y_bytes.decode('utf-8'))
            features.append(feat)
            labels.append(y)
        features = torch.from_numpy(np.stack(features)).to(device)
        labels = torch.from_numpy(np.array(labels)).to(device)
        class_labels = torch.zeros([batch_size, num_classes], device=device)
        if num_classes > 0:
            class_labels = torch.eye(num_classes, device=device)[labels]
        assert features.shape[0] == class_labels.shape[0] == batch_size
    return features, class_labels



def show_denoised(denoised, vae):
        def recur_decode(z):
            try:
                return vae.decode(z)
            except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
                assert z.shape[2] % 2 == 0
                z1, z2 = z.tensor_split(2)
                return torch.cat([recur_decode(z1), recur_decode(z2)])

        noised_img = recur_decode(denoised)



        # Scale and clip the noised image for valid display
        denoised_img = noised_img.permute(0, 2, 3, 1).to("cpu",dtype=torch.uint8).numpy()
#
        num_images = denoised_img.shape[0]  # Number of images in the batch
        fig, axes = plt.subplots(num_images, 2, figsize=(12, 6 * num_images))  # Create a grid of images

        # If there is only one image, axes will be a 1D array, handle it gracefully
        if num_images == 1:
            axes = axes.reshape(1, 2)

        for i in range(num_images):
            # Original image
            # axes[i, 0].imshow(original_img[i])
            # axes[i, 0].axis('off')
            # axes[i, 0].set_title(f"Original Image {i + 1}")

            # Noised image
            axes[i, 1].imshow(denoised_img[i])
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f"Denoised Image {i + 1}")

        plt.tight_layout()  # Adjust spacing between subplots for better appearance
        # plt.savefig(r'D:\dengkai\code\MaskDiT-master\results\fid\output_images_all.png', dpi=300,
        #             bbox_inches='tight')  # Save all images
        plt.show()  # Optionally, show the plot if you want to display it interactively


@torch.no_grad()
def generate_with_net(args, net, vae, device, rank, val_loader, num_expected = 50000):
    seeds = args.seeds

    text_context = None
    clip_model = None
    if args.use_clip:
        from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder
        clip_model = FrozenCLIPTextEmbedder("ViT-L/14")
        clip_model = clip_model.to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    net.eval()

    # Setup sampler
    sampler_kwargs = dict(num_steps=args.num_steps, S_churn=args.S_churn,
                          solver=args.solver, discretization=args.discretization,
                          schedule=args.schedule, scaling=args.scaling,
                          flag=args.eval_type)
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
    sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
    mprint(f"sampler_kwargs: {sampler_kwargs}, \nsampler fn: {sampler_fn.__name__}")
    # Setup autoencoder

    # generate images
    mprint(f'Generating {len(seeds)} images to "{args.outdir}"...')
    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)
    num_sum = 0
    for data in tqdm(val_loader ,disable=(rank != 0)):
        dist.barrier()
        if num_sum >= num_expected:
            break
        # x, x_cond, vae_feats, instance_map, bboxes, labels, img_records, batch = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5], data[6], data[7]
        x = data["image"].to(device)
        x_cond = data["cond_seg"].to(device)
        instance = data["cond_inst"].to(device)

        vae_feats = data["vae_feats"].to(device)
        graphs = data["cond_graph"]
        graphs = None
        # graphs = Batch.from_data_list(graphs).to(device) if len(
        #     graphs) > 0 else None


        img_records = data["img_records"]

        region_captions = [record["captions"] for record in img_records]
        exists = True
        for i in range(x.shape[0]):
            basename = os.path.basename(img_records[i]["img_path"]).replace(".tif", ".png")
            regoins = region_captions[i]
            save_dir = os.path.join(args.outdir, f"samples/{regoins}")
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, basename)
            if not os.path.exists(save_name):
                exists = False
                break
        if exists:
            continue

        ref = apply_color_palette_torch(x_cond)


        #all_regions = val_loader.dataset.get_regions()



        #region_captions = ['adelaide'] * len(region_captions)
        #"convert aachen to houston"
        #region_captions = [regoin.replace("aachen","mazowieckie") for regoin in region_captions]

        y = []

        with torch.no_grad():
            for text in region_captions:
                y.append(clip_model([text]))
            y = torch.cat(y).to(device)

        latents = torch.randn([x_cond.shape[0], 4, 32, 32], device=device)


        if args.cfg_scale is not None:
            x_cond = torch.cat([x_cond, torch.zeros_like(x_cond)], dim=0)
            y = torch.cat([y, torch.zeros_like(y)], dim=0)
            latents = torch.cat((latents, latents), dim=0)

        #x_cond = preprocess_input(x_cond, 9)
        # Generate images.
        def recur_decode(z):
            try:
                return vae.decode(z)
                #return vae.decode(samples / 0.18215).sample
            except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
                assert z.shape[2] % 2 == 0
                z1, z2 = z.tensor_split(2)
                return torch.cat([recur_decode(z1), recur_decode(z2)])




        with torch.no_grad():

            z = sampler_fn(net, latents.float(), seg=x_cond, class_labels=y,  text_context = text_context,  randn_like=torch.randn_like,instance_map = instance,
                       cfg_scale=args.cfg_scale,  graph=graphs, vae = vae, **sampler_kwargs).float()

            samples = recur_decode(z)

            if args.cfg_scale is not None:
                samples, _ = torch.split(samples, samples.shape[0] // 2, dim=0)

        num_sum  = num_sum + samples.shape[0]
        for i in range(samples.shape[0]):
            basename = os.path.basename(img_records[i]["img_path"]).replace(".tif", ".png")
            basename_ref = os.path.basename(img_records[i]["img_path"]).replace(".tif", "_ref.png")


            regoins = region_captions[i]
            save_dir = os.path.join(args.outdir, f"samples/{regoins}")
            os.makedirs(save_dir, exist_ok=True)

            save_name = os.path.join(save_dir,  basename)
            save_name_ref = os.path.join(args.outdir, "conditions", basename_ref)
            # if os.path.exists(save_name):
            #     continue
            save_image(samples[i].float(), save_name, nrow=1, normalize=True, value_range=(0, 255))
            #save_image(ref[i], save_name_ref, nrow=samples.shape[0], normalize=True, value_range=(-1, 1))


def eval_ddpm(args, net, diffusion, vae, device, rank, val_loader):
    for data in tqdm(val_loader ,disable=(rank != 0)):
        dist.barrier()

        # x, ref, vae_feats, instance_map, labels, img_records, batch = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5], data[6]

        x, ref, vae_feats, instance_map, labels, img_records, batch = data[0].to(device), data[1].to(device), data[
            2].to(device), data[3].to(device), data[4].to(device), data[5], data[6]

        y = None
        if args.num_classes > 0:
            y = get_one_hot(labels, args.num_classes)
            y_null = torch.zeros_like(y)
            y = torch.cat((y, y_null), dim=0)


        # Pick latents and labels.

        # cfg_cond = torch.zeros_like(x_cond)
        # x_cond = torch.cat((x_cond, cfg_cond), dim=0)

        # paths= img_records[0]["img_path"]

        if args.with_graph:
            graphs = batch
            for key, val in graphs.items():
                graphs[key] = graphs[key].to(device) if graphs[key] is not None else None
            # graphs = data[3].to(device)
            graphs_new, instance_map_new, x_cond_new = _contrust_graph(ref, instance_map, graphs, dropout=0.0)
        else:
            graphs = None
        latents = torch.randn([ref.shape[0], 4, 32, 32], device=device)
        if args.cfg_scale is not None:
            graphs_cfg, instance_map_cfg, x_cond_cfg = _contrust_graph(ref, instance_map, graphs, dropout=1.0)
            graphs = concatenate_graphs([graphs_new, graphs_cfg])
            instance_map = torch.cat([instance_map_new, instance_map_cfg], dim=0)
            x_cond = torch.cat([x_cond_new, x_cond_cfg], dim=0)
            latents = torch.cat((latents, latents), dim=0)
        else:
            graphs = graphs_new
            instance_map = instance_map_new
            x_cond = x_cond_new
        # retrieve features from training set [support random only]
        feat = None
        x_cond = preprocess_input(x_cond, 9)

        # Generate images.
        def recur_decode(z):
            try:
                return vae.decode(z / 0.18215).sample
            except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
                assert z.shape[2] % 2 == 0
                z1, z2 = z.tensor_split(2)
                return torch.cat([recur_decode(z1), recur_decode(z2)])

        model_kwargs = dict(seg=x_cond, y=y, graph=graphs, instance_map=instance_map)
        with torch.no_grad():


            z = diffusion.ddim_sample_loop(
                net.forward if args.cfg_scale is None else net.forward_with_scale,
                latents.shape,
                latents,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True, device=device
            )

            samples = recur_decode(z)

            if args.cfg_scale is not None:
                samples, _ = torch.split(samples, samples.shape[0] // 2, dim=0)

            # Save images.
            # images_np = images.add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        ref = apply_color_palette_torch(ref)

        ref = ref / 255.0 - 1
        # .mul_(std[:, None, None]).add_(mean[:, None, None])
        # samples = torch.cat((ref, samples), dim=0)

        for i in range(samples.shape[0]):
            basename = os.path.basename(img_records[i]["img_path"]).replace(".tif", ".png")
            basename_ref = os.path.basename(img_records[i]["img_path"]).replace(".tif", "_ref.png")
            save_name = os.path.join(args.outdir, "samples", basename)
            save_name_ref = os.path.join(args.outdir, "conditions", basename_ref)
            save_image(samples[i], save_name, nrow=samples.shape[0], normalize=True, value_range=(-1, 1))
            # save_image(ref[i], save_name_ref, nrow=samples.shape[0], normalize=True, value_range=(-1, 1))


def generate(args):
    device = torch.device("cuda")

    mprint(f'cf_scale: {args.cfg_scale}')
    if args.global_rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        logger = Logger(file_name=f'{args.outdir}/log.txt', file_mode="a+", should_flush=True)

    # Create model:
    net = Precond_models[args.precond](
        img_resolution=args.image_size,
        img_channels=args.image_channels,
        num_classes=args.num_classes,
        model_type=args.model_type,
        use_decoder=args.use_decoder,
        mae_loss_coef=args.mae_loss_coef,
        pad_cls_token=args.pad_cls_token,
        ext_feature_dim=args.ext_feature_dim

    ).to(device)
    mprint(
        f"{args.model_type} (use_decoder: {args.use_decoder}) Model Parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Load checkpoints
    ckpt = torch.load(args.ckpt_path, map_location=device)
    net.load_state_dict(ckpt['model'])
    mprint(f'Load weights from {args.ckpt_path}')

    generate_with_net(args, net, device)

    # Done.
    cleanup()
    if args.global_rank == 0:
        logger.close()

# def save_image(x, path):
#     c,h,w = x.shape
#     assert c==3
#     x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
#     PIL.Image.fromarray(x).save(path)
def concatenate_graphs(data_list):
    x = torch.cat([data.x for data in data_list], dim=0)
    edge_index = torch.cat([data.edge_index + cumsum_nodes for data, cumsum_nodes in zip(data_list, torch.tensor([0] + [data.num_nodes for data in data_list[:-1]]).cumsum(0))], dim=1)

    cur_num = 0
    new_batch = []
    for data in data_list:
        batch = data.batch + cur_num
        new_batch.append(batch)
        cur_num += batch.max() + 1
    batch = torch.cat(new_batch, dim=0)
    #batch = torch.cat([data.batch for data in enumerate(data_list)])

    # Assuming all graphs have the same features, otherwise you would need to handle features separately
    return Data(x=x, edge_index=edge_index, batch=batch)



def sample(args):
    config = OmegaConf.load(args.config)
    if platform.system() == 'Windows':
        _setup_process_group(args)
    else:
        dist.init_process_group("nccl")

        # _setup_process_group(args)
        assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    global_batch_size = args.global_batch_size

    mprint('start training...')

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = 23
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")





    clip_model = None

    transform, label_transform = None, None



    dataset =  M2IBase(
        data_dir=config.data.val.data_dir
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_graph
    )

    mprint(f"Dataset contains {len(dataset)}")

    steps_per_epoch = len(dataset) // global_batch_size
    mprint(f"{steps_per_epoch} steps per epoch")





    from models.GDiT import Precond_models

    net = Precond_models[config.model.precond](
        **config.model.params
    ).to(device)

    assert args.ckpt_path is not None



    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    net = DDP(net, device_ids=[rank], find_unused_parameters=True)
    if 'model' in ckpt.keys():
        net.module.load_state_dict(ckpt['model'], strict=False)
    else:
        net.module.load_state_dict(ckpt, strict=False)
    #net.load_state_dict(ckpt['model'], strict=False)
    # ema.load_state_dict(ckpt['ema'], strict=args.use_strict_load)
    mprint(f'Load weights from {args.ckpt_path}')

    net.eval()

    # Setup sampler
    sampler_kwargs = dict(num_steps=args.num_steps, S_churn=args.S_churn,
                          solver=args.solver, discretization=args.discretization,
                          schedule=args.schedule, scaling=args.scaling)
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
    sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
    mprint(f"sampler_kwargs: {sampler_kwargs}, \nsampler fn: {sampler_fn.__name__}")
    # Setup autoencoder

    vae = StabilityVAEEncoder(vae_name=r"./pretrained")
    #vae = AutoencoderKL.from_pretrained(r"./pretrained").to(device).eval()
    # z = torch.randn((1,4,32,32)).to(device)
    # vae.decode(z / 0.18215).sample

    args.outdir = os.path.join(args.outdir , args.eval_type)

    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "conditions"), exist_ok=True)

    generate_with_net(args, net.module, vae, device, rank, loader)


    # Done.
    cleanup()
    print ("Done")




if __name__ == '__main__':

    parser = argparse.ArgumentParser('sampling parameters')

    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument("--version", type=str, choices=["v3", "GDiT"], default = "GDiT",required=True)
    parser.add_argument("--eval_type", type=str, choices=["regions", "regions_graph", "regions_sem","regions_graph_sem"], default="regions_graph_sem", required=False)





    #data

    # parser.add_argument("--data-eval", type=str, default=None, required=True)
    # parser.add_argument("--graph-eval", type=str, default=None, required=False)
    parser.add_argument("--with-graph", type=str2bool, default=False, help="weather use latent")
    parser.add_argument('--gpu', default=[0, 1, 2], nargs='+', type=int, dest='gpu', help='The gpu list used.')
    parser.add_argument("--use_strict_load", type=str2bool, default=True)
    parser.add_argument("--global-batch-size", type=int, default=5)
    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='address for master')

    # sampling
    parser.add_argument("--feat_path", type=str, default='')
    parser.add_argument("--ext_feature_dim", type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, required=True, help='Network pickle filename')
    parser.add_argument('--outdir', type=str, required=True, help='sampling results save filename')
    parser.add_argument('--seeds', type=parse_int_list, default='0-63', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class_idx', type=int, default=None, help='Class label  [default: random]')
    parser.add_argument('--max_batch_size', type=int, default=64, help='Maximum batch size per GPU')

    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=18, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'],
                        help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'],
                        help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/stable_diffusion/autoencoder_kl.pth',
                        help='Autoencoder ckpt')

    # model
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=0, help='0 means unconditional')
    parser.add_argument("--model_type", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--fusion_type", type=str,  default="AdaLn")
    parser.add_argument("--mode", type=str, default="seg")
    parser.add_argument('--precond', type=str, choices=['vp', 've', 'edm'], default='edm', help='precond train & loss')
    parser.add_argument("--use_decoder", type=str2bool, default=False)
    parser.add_argument("--pad_cls_token", type=str2bool, default=False)
    parser.add_argument('--mae_loss_coef', type=float, default=0, help='0 means no MAE loss')
    parser.add_argument('--sample_mode', type=str, default='rand_full', help='[rand_full, rand_repeat]')
    parser.add_argument("--use_clip", action='store_true', default=False)

    #kmeans
    parser.add_argument("--ckpt_kmeans", type=str,
                        default=r"F:\data\OpenEarthMap\Size_256\train\classfications\kmeans_model.pkl",
                        help="Optional path to a kmeans model.")
    parser.add_argument("--classfications", type=str,
                        default=r"F:\data\OpenEarthMap\Size_256\train\classfications\clustering_result.csv",
                        help="Optional path to a kmeans model.")
    parser.add_argument("--use_kmeans", type=str2bool, default=False)
    args = parser.parse_args()



    torch.set_num_threads(1)







    eval_types = ["regions_graph_sem"]
    for eval_type in eval_types:
        print(f"eval as {eval_type}")
        args.eval_type = eval_type
        sample(args)

    # args.global_size = args.num_proc_node * args.num_process_per_node
    # size = args.num_process_per_node
    #
    # if size > 1:
    #     processes = []
    #     for rank in range(size):
    #         args.local_rank = rank
    #         args.global_rank = rank + args.node_rank * args.num_process_per_node
    #         p = Process(target=init_processes, args=(generate, args))
    #         p.start()
    #         processes.append(p)
    #
    #     for p in processes:
    #         p.join()
    # else:
    #     print('Single GPU run')
    #     assert args.global_size == 1 and args.local_rank == 0
    #     args.global_rank = 0
    #     init_processes(generate, args)
