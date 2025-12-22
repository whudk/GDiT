# MIT License

# Copyright (c) [2023] [Anima-Lab]

# This code is adapted from https://github.com/NVlabs/edm/blob/main/fid.py. 
# The original code is licensed under a Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License, which is can be found at licenses/LICENSE_EDM.txt. 

"""Script for calculating Frechet Inception Distance (FID)."""
import argparse
import os.path
from multiprocessing import Process

import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from utils import *
from train_utils.datasets import ImageFolderDataset
import platform
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
#----------------------------------------------------------------------------
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
def calculate_inception_stats(
        image_path, num_expected=None, seed=0, max_batch_size=64,
        num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        dist.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = r'./cache/hub/pyiqa/inception-2015-12-05.pt'

    # Load the Inception-v3 model
    mprint('Loading Inception-v3 model...')
    detector_kwargs = dict(return_features=True)
    detector_net = torch.load(detector_url, map_location=device)
    detector_net = detector_net.to(device)
    feature_dim = 2048
    # List images.
    mprint(f'Loading images from "{image_path}"...')
    dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        dist.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    mprint(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        dist.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    dist.all_reduce(mu)
    dist.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

def inception_score(  image_path, num_expected=None, seed=0, max_batch_size=64,
        num_workers=3, prefetch_factor=2, device=torch.device('cuda') , resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    mprint('Loading Inception-v3 model...')
    detector_kwargs = dict(return_features=True)
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    feature_dim = 2048
    # List images.
    mprint(f'Loading images from "{image_path}"...')
    dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        dist.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]
    data_loader = DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers,
                             prefetch_factor=prefetch_factor)

    N = len(dataset_obj)

    batch_size = all_batches[0].shape[0]
    dtype = torch.cuda.FloatTensor
    # Set up dtype
    if device:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader


    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x).cpu()

    # Get predictions
    preds = []

    for i, batch in enumerate(data_loader, 0):
        batch = batch[0].type(dtype) / 255.0 - 1.0
        #batchv = torch.Variable(batch)
        batch_size_i = batch.size()[0]

        preds.append(get_pred(batch))

    preds = torch.cat(preds, dim=0).numpy()
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
def calc(image_path, ref_path, num_expected, seed, batch, Eval_list = ['FID', 'IS']):
    """Calculate FID for a given set of images."""
    if dist.get_rank() == 0:
        name = '_'.join(Eval_list)
        logger = Logger(file_name=f'{image_path}/log_{name}.txt', file_mode="a+", should_flush=True)

    mprint(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        assert ref_path.endswith('.npz')
        ref = dict(np.load(ref_path))



    for Eval in Eval_list:

        mprint(f'Calculating {Eval}...')
        if Eval == 'FID':
            mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed,
                                                  max_batch_size=batch)
            res = None
            if dist.get_rank() == 0:
                res = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])
                print(f'{res:g}')

            dist.barrier()
            if dist.get_rank() == 0:
                logger.close()
        else:
            res = None
            if dist.get_rank() == 0:
                res = inception_score(image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)[0]
                print(f'{res:g}')

            dist.barrier()
            if dist.get_rank() == 0:
                logger.close()
    return res

#----------------------------------------------------------------------------


def ref(dataset_path, dest_path, batch, index = ""):
    """Calculate dataset reference statistics needed by 'calc'."""

    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch)
    mprint(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    dist.barrier()
    mprint('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('fid parameters')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='address for master')

    # fid
    parser.add_argument('--mode', type=str, required=True, choices=['calc', 'ref'], help='Calcalute FID or store reference statistics')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the images')
    parser.add_argument('--ref_path', type=str, default='assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz', help='Dataset reference statistics')
    parser.add_argument('--num_expected', type=int, default=50000, help='Number of images to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for selecting the images')
    parser.add_argument('--batch', type=int, default=64, help='Maximum batch size per GPU')

    args = parser.parse_args()
    args.global_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    func = lambda args: calc(args.image_path, args.ref_path, args.num_expected, args.seed, args.batch) \
        if args.mode == 'calc' else lambda args: ref(args.image_path, args.ref_path, args.batch)

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            args.global_rank = rank + args.node_rank * args.num_process_per_node
            p = Process(target=init_processes, args=(func, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('Single GPU run')
        assert args.global_size == 1 and args.local_rank == 0
        args.global_rank = 0
        args.gpu = [0]
        if platform.system() == 'Windows':
            _setup_process_group(args)
        else:
            dist.init_process_group("nccl")

        #ref(args.image_path, args.ref_path, args.batch)
        calc(args.image_path,args.ref_path,None, args.seed,args.batch,Eval_list=['FID'])

        # processes = []
        # p = Process(target=init_processes, args=(func, args))
        # p.start()
        # processes.append(p)
        #
        # for p in processes:
        #     p.join()
        #init_processes(func, args)