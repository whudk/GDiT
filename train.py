# MIT License

# Copyright (c) [2023] [Anima-Lab]
'''
Training MaskDiT on latent dataset in LMDB format. Used for experiments on Imagenet256x256.
'''

import argparse
import os.path

from copy import deepcopy
from time import time
from omegaconf import OmegaConf
import torch
import torch.optim as optim
from fid import calc
from train_utils.loss import Losses
from train_utils.helper import get_mask_ratio_fn, get_one_hot,requires_grad, update_ema, unwrap_model
from train_utils.datasets  import M2IBase,M2IOSM
from sample import generate_with_net
from utils import dist, mprint, get_latest_ckpt, Logger, sample, \
    str2bool, parse_str_none, parse_int_list, parse_float_none
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.GDiT import Precond_models
from diffusers.models import AutoencoderKL
import platform

import  logging
from torch.cuda.amp import autocast, GradScaler


from train_utils.helper import collate_graph
from tqdm import tqdm

# ------------------------------------------------------------
def preprocess_input(x_cond, nc):
    bs, _, h, w = x_cond.size()

    input_label = torch.FloatTensor(bs, nc, h, w).zero_().to(x_cond.device)
    input_semantics = input_label.scatter_(1, x_cond.long(), 1.0)

    # mask = (torch.rand([input_semantics.shape[0], 1, 1, 1]) > 0.5).float().to(x_cond.device)
    # input_semantics = input_semantics * mask

    x_cond = input_semantics

    return x_cond



def get_unique_classids(labels, num_classes = 9):

    class_id_finals = []
    for b in range(labels.shape[0]):
        class_id = sorted(torch.unique(labels[b]))
        class_id_final = torch.zeros(num_classes).long()
        if class_id[0] == 0:
            class_id = class_id[1:]
        for i in range(len(class_id)):
            class_id_final[class_id[i].long()] = 1
        class_id_finals.append(class_id_final)
    return torch.stack(class_id_finals, dim = 0).to(labels.device)




def __val__(model,clip_model,vae, loss_fn, val_loader, num_expected = 2000):
    model.eval()
    clip_model.eval()
    device = model.device
    val_losses = 0
    mprint("evaluating on val set")

    for i, (data) in tqdm(enumerate(val_loader)):
        # x, x_cond, vae_feats, instance_map, bboxes, labels, img_records, batch = data[0].to(device), data[1].to(device), \
        # data[2].to(device), data[3].to(device), data[4].to(device), data[5], data[6], data[7]
        x = data["image"].to(device)
        x_cond = data["cond_seg"].to(device)
        instance = data["cond_inst"].to(device)

        vae_feats = data["vae_feats"].to(device)
        graphs = data["cond_graph"].to(device)
        img_records = data["img_records"]

        region_captions = [ record["captions"] for record in img_records]

        y = []

        with torch.no_grad():
            for text in region_captions:
                y.append(clip_model([text]))
            y = torch.cat(y).to(device)




        class_ids = None

        if vae_feats is None:
            x = vae.encode_latents(vae.encode_pixels(x.to(device)))
        else:
            x = vae_feats


        x = x.half()


        with torch.no_grad():
            val_loss = loss_fn(net=model, images=x, seg=x_cond, labels=y, feat=None,
                           class_ids=class_ids, instance_map=instance,
                           mask_ratio=0.0,
                           mae_loss_coef=.0, graph=graphs, train_steps=0, vae=vae)

        avg_loss = val_loss["total"].mean()
        # Synchronize losses across distributed GPUs
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        loss = avg_loss.item() / dist.get_world_size()
        # if dist.get_world_size() > 1:
        #     #avg_loss = torch.tensor(val_loss, device=device)
        #
        # else:
        #     loss = val_loss["total"].mean()
        val_losses += loss

    model.train()
    dist.barrier()
    return  val_losses / len(val_loader)

def train_loop(args):
    # load configuration
    config = OmegaConf.load(args.config)
    
    if not args.no_amp:
        config.train.amp = 'fp16'
        #torch.set_default_dtype(torch.float16)
    else:
        config.train.amp = 'no'
    
    if config.train.tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    torch.set_num_threads(1)
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
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")



    mprint(f"enable_amp: {not args.no_amp}, TF32: {config.train.tf32}")
    # Select batch size per GPU
    # Select batch size per GPU

    #class_dropout_prob = 0.1
    class_dropout_prob = config.model.params.seg_dropout_prob


    log_every = config.log.log_every
    ckpt_every = config.log.ckpt_every
    mask_ratio_fn = get_mask_ratio_fn(config.model.params.mask_ratio_fn, config.model.params.node_dropout_prob,0.0)

    data_name = config.data.dataset
    if args.ckpt_path is not None and args.use_ckpt_path:  # use the existing exp path (mainly used for fine-tuning)
        checkpoint_dir = os.path.dirname(args.ckpt_path)
        experiment_dir = os.path.dirname(checkpoint_dir)
        exp_name = os.path.basename(experiment_dir)
    else:  # start a new exp path (and resume from the latest checkpoint if possible)
        #cond_gen = str(config.model.mode)
        # exp_name = f'{model_name}-{config.model.precond}-{data_name}-{cond_gen}-m{config.model.mask_ratio}-de{int(config.model.use_decoder)}' \
        #            f'-mae{config.model.mae_loss_coef}-bs-{global_batch_size}-lr{config.train.lr}{config.log.tag}'
        exp_name = f'{config.model.params.model_type}-fusion_type-{config.model.params.fusion_type}'
        experiment_dir = f"{args.results_dir}/{exp_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        # if args.ckpt_path is None:
        #     args.ckpt_path = get_latest_ckpt(checkpoint_dir)  # Resumes from the latest checkpoint if it exists
    mprint(f"Experiment directory created at {experiment_dir}")

    # Build CLIP model:
    clip_model = None
    if args.use_clip:
        from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder
        clip_model = FrozenCLIPTextEmbedder("ViT-L/14")
        clip_model = clip_model.to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad = False

    from train_utils.encoders import StabilityVAEEncoder

    vae = StabilityVAEEncoder(vae_name = r"./pretrained")

    train_transform = val_transform = None

    if config.data.dataset == 'OSM':
        dataset = M2IBase
    elif config.data.dataset == 'OEM':
        dataset = M2IOSM
    else:
        raise NotImplementedError

    train_dataset = dataset(
        data_dir=config.data.train.data_dir
    )
    val_dataset = dataset(
        data_dir=config.data.val.data_dir
    )
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_graph
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_graph
    )



    #
    # train_dataset = M2I_dataset(
    #     img_dir = config.data.train.img_dir,
    #     lbl_dir = config.data.train.lbl_dir,
    #     graph_dir = config.data.train.graph_dir,
    #     vae_dir = config.data.train.vae_dir,
    #     instance_dir=config.data.train.instance_dir,
    #     kmeans_path = config.data.kmeans.pretrained,
    #     clip_model=None,
    #     transform=train_transform,
    #     with_graph=args.with_graph,
    #     #graph_folder=args.graph_train,
    # )
    # val_dataset = M2I_dataset(
    #     img_dir=config.data.val.img_dir,
    #     lbl_dir=config.data.val.lbl_dir,
    #     graph_dir=config.data.val.graph_dir,
    #     vae_dir=config.data.val.vae_dir,
    #     instance_dir=config.data.val.instance_dir,
    #     kmeans_path=config.data.kmeans.pretrained,
    #     clip_model=None,
    #     transform=val_transform,
    #
    #     with_graph=args.with_graph,
    #
    #     sam_dir=None,
    #
    # )
    # sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    # loader = DataLoader(
    #     train_dataset,
    #     batch_size=int(args.global_batch_size // dist.get_world_size()),
    #     shuffle=False,
    #     sampler=sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    #     collate_fn=collate_graph
    # )
    # val_sampler = DistributedSampler(
    #     val_dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     sampler=val_sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    #     collate_fn=m2t_collate
    # )




    mprint(f"Dataset contains {len(train_dataset):,} images ({config.data.root})")

    steps_per_epoch = len(train_dataset) // global_batch_size
    mprint(f"{steps_per_epoch} steps per epoch")





    model = Precond_models[config.model.precond](
        **config.model.params
    ).to(device)


    # Load checkpoints
    train_steps_start = 0
    epoch_start = 0
    sampler.set_epoch(epoch_start)
    val_sampler.set_epoch(epoch_start)
    #args.ckpt_path = None
    # Prepare models for training:
    if args.ckpt_path is None:
        assert train_steps_start == 0
        raw_model = unwrap_model(model)
        #update_ema(ema, raw_model, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")

        if "model" in ckpt.keys():
            state_dict = ckpt['model']
            #print(state_dict)
        else:
            state_dict = ckpt


        model.load_state_dict(state_dict, strict=False)



        mprint(f"Forcefully loaded weights from {args.ckpt_path}")

        del state_dict # conserve memory

        #FID evaluation during training
        if args.enable_eval:
            start_time = time()
            args.outdir = os.path.join(experiment_dir, 'fid',
                                       f'edm-steps{args.num_steps}-ckpt{train_steps_start}_cfg{args.cfg_scale}_{args.eval_type}')
            os.makedirs(args.outdir, exist_ok=True)
            #generate_with_net(args, model.eval(), device, rank, 500)
            generate_with_net(args, model, vae, device, rank, val_loader,args.num_expected)
            dist.barrier()
            # fid = calc(args.outdir, config.eval.ref_path, args.num_expected, args.global_seed, args.fid_batch_size)
            # mprint(f"time for fid calc: {time() - start_time}, fid: {fid}")
            # mprint(f'Guidance: {args.cfg_scale}, FID: {fid}')
            #dist.barrier()


    #ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    #requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=3e-2)
    scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=config.train.max_num_steps // 4, eta_min=1e-5)


    ema = deepcopy(model.module).eval().requires_grad_(False).to(torch.float16)
    mprint(
        f"{config.model.params.model_type}  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup loss
    loss_fn = Losses[config.model.precond]()

    update_ema(ema, model.module, decay=0.999)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = train_steps_start
    log_steps = 0
    running_loss = 0
    running_loss_mae = 0
    start_time = time()
    mprint(f"Training for {config.train.epochs} epochs...")

    text_context = None

    best_loss =1000
    for epoch in range(epoch_start, config.train.epochs):
        mprint(f"Beginning epoch {epoch}...")

        loader.sampler.set_epoch(epoch)

        for i, (data) in enumerate(loader):


            x = data["image"].to(device)
            x_cond = data["cond_seg"].float().to(device)
            instance = data["cond_inst"].to(device)

            vae_feats = data["vae_feats"].to(device)
            graphs = data["cond_graph"].to(device)
            img_records = data["img_records"]


            region_captions = [ record["captions"] for record in img_records]


            y = []
            class_ids = None
            with torch.no_grad():
                for text in region_captions:
                    y.append(clip_model([text]))
                y = torch.cat(y).to(device)
                if class_dropout_prob > 0:
                    y = y * (torch.rand([y.shape[0], 1], device=device) >= class_dropout_prob)
            #vae_feats = None
            if vae_feats is None:
                x = vae.encode_latents(vae.encode_pixels(x.to(device)))
                #x = x/127.5 -1.0
                #x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            else:
                x = vae_feats

            if not args.no_amp:
                x = x.half()




            loss_batch = 0
            optimizer.zero_grad(set_to_none=True)
            if 'mask_ratio' in config.model.params.model_type:
                curr_mask_ratio = mask_ratio_fn((train_steps - train_steps_start) / config.train.max_num_steps)
                mae_loss_coef=config.model.params.mae_loss_coef
            else:
                curr_mask_ratio = mask_ratio_fn((train_steps - train_steps_start) / config.train.max_num_steps)
                mae_loss_coef = 0.0

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = loss_fn(net=model, images=x, seg = x_cond, labels=y,  text_context = text_context, class_ids = class_ids,mask_ratio=curr_mask_ratio, instance_map = instance,
                                   mae_loss_coef=mae_loss_coef, graph = graphs,  train_steps = train_steps, vae = vae)



            #print(torch.cuda.memory_summary())
            # if "mae" in loss.keys():
            #     loss_mean_mae = loss["mae"].mean()

            loss_mean = loss["total"].mean()


            # scaler.scale(loss_mean).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss_mean.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # loss_mean.backward()
            # optimizer.step()
            #raw_model = unwrap_model(model)

            scheduler.step()
            # Log loss values:
            loss_batch += loss_mean.item()
            running_loss += loss_batch

            log_steps += 1
            train_steps += 1
            if train_steps > (train_steps_start + config.train.max_num_steps):
                break
            if train_steps % log_every == 0:
                # Measure training speed:
                update_ema(ema, model.module)
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                for g in optimizer.param_groups:
                    lr = g['lr']
                mprint(f"(step={train_steps:07d}) Lr:{lr:.8f}, Train Loss: {avg_loss:.4f},  Train Steps/Sec: {steps_per_sec:.2f}")
                mprint(f"mask_ratio={curr_mask_ratio:.4f}")
                mprint(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')
                mprint(f'Reserved GPU memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB')


                # Reset monitoring variables:
                running_loss = 0
                running_loss_mae = 0
                log_steps = 0
                start_time = time()

                # Save checkpoint:

            if train_steps % ckpt_every == 0 and train_steps > train_steps_start:

                #eval_val_loader
                val_loss = __val__(model, clip_model, vae, loss_fn, val_loader)
                mprint(f"val_loss_on_valset: {val_loss:.4f}")

                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,

                    }
                    if val_loss < best_loss:
                        best_loss = val_loss
                        checkpoint_path = f"{checkpoint_dir}/best_{best_loss:04f}.pt"
                        torch.save(model.module.state_dict(), checkpoint_path)
                        print("current best loss is {}".format(best_loss))
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    mprint(f"Saved checkpoint to {checkpoint_path}")
                    del checkpoint  # conserve memory
                dist.barrier()

                #dist.barrier()

                # FID evaluation during training
            if args.enable_eval and train_steps % 10 == 0 :
                start_time = time()
                args.outdir = os.path.join(experiment_dir, 'fid',
                                           f'edm-steps{args.num_steps}-ckpt{train_steps_start}_cfg{args.cfg_scale}')
                os.makedirs(args.outdir, exist_ok=True)
                model.eval()
                with torch.no_grad():
                    generate_with_net(args, model.module, vae, device, rank, val_loader, args.num_expected)
                dist.barrier()
                # fid = calc(args.outdir, config.eval.ref_path, args.num_expected, args.global_seed, args.fid_batch_size)
                # mprint(f"time for fid calc: {time() - start_time}")
                # # if args.use_wandb:
                # #     accelerator.log({f'eval/fid': fid}, step=train_steps_start)
                # mprint(f'guidance: {args.cfg_scale} FID: {fid}')
                dist.barrier()
                model.train()
                start_time = time()
            torch.cuda.empty_cache()



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

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger





if __name__ == '__main__':
    parser = argparse.ArgumentParser('training parameters')
    # basic config
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument("--version", type=str, default="GDiT",  required=True)
    parser.add_argument("--eval_type", type=str,
                        choices=["regions", "regions_graph", "regions_sem", "regions_graph_sem"],
                        default="regions_graph_sem", required=False)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt-path", type=parse_str_none, default=None)

    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--no_amp', action='store_true', help="Disable automatic mixed precision.")

    parser.add_argument("--use_wandb", action='store_true', help='enable wandb logging')
    parser.add_argument("--use_ckpt_path", type=str2bool, default=False)
    parser.add_argument("--use_strict_load", type=str2bool, default=True)
    parser.add_argument("--use_clip", action='store_true', default=False)
    parser.add_argument("--tag", type=str, default='')

    parser.add_argument("--with-graph", type=str2bool, default=False, help="weather use latent")
    parser.add_argument('--enable_eval', action='store_true', help='enable fid calc during training')
    parser.add_argument('--seeds', type=parse_int_list, default='0-49999', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class_idx', type=int, default=None, help='Class label  [default: random]')
    parser.add_argument('--max_batch_size', type=int, default=50, help='Maximum batch size per GPU during sampling, must be a factor of 50k if torch.compile is used')

    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=18, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'], help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'], help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/stable_diffusion/autoencoder_kl.pth', help='Autoencoder ckpt')

    # parser.add_argument('--ckpt-path', type=str, default='assets/stable_diffusion/autoenco',
    #                     help='resume ckpt')
    parser.add_argument('--ref_path', type=str, default='assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz', help='Dataset reference statistics')
    parser.add_argument('--num_expected', type=int, default=500, help='Number of images to use')
    parser.add_argument('--fid_batch_size', type=int, default=64, help='Maximum batch size per GPU')

    parser.add_argument("--latent", type=str2bool, default=False, help="weather use latent")
    parser.add_argument('--gpu', default=[0, 1, 2], nargs='+', type=int, dest='gpu', help='The gpu list used.')
    parser.add_argument('--local_rank', type=int, default=0, dest='local_rank', help='local rank of current process')
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=0)

    #parser.add_argument("--image-size", type=int,  choices=[256, 512], default=256)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--global-batch-size", type=int, default=5)
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True
    train_loop(args)
