import argparse
import sys

import torch
from omegaconf import OmegaConf
import cv2
import numpy as np
from scripts.semantic_to_captions import semantic_to_instance_map,get_adjacency_matrix_from_instace_map,get_node_feats
from train_utils.encoders import StabilityVAEEncoder
from sample import edm_sampler
from torch_geometric.data import Data, Batch
import platform,os
import torch.distributed as dist
from torchvision.utils import save_image
import glob
import matplotlib.pyplot as plt
from train_utils.helper import extract_bboxes
import tqdm
oem = {
  0:"Background",
  1:"Bareland",
  2:"Rangeland",
  3:"Developed space",
  4:"Road",
  5:"Tree",
  6:"Water",
  7:"Agriculture land",
  8:"Building"
}
revert_oem = {
  "Background":0,
  "Bareland":1,
  "Rangeland":2,
  "Developed space":3,
  "Road":4,
  "Tree":5,
  "Water":6,
  "Agriculture land":7,
  "Building":8
}

from types import SimpleNamespace


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

def __init_process(args):
    if platform.system() == 'Windows':
        _setup_process_group(args)
    else:
        dist.init_process_group("nccl")
        assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    print('start training...')

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

def regions_Domain_Adaptation(args, from_regoin, dst_regions):
    img_dir = r"F:\data\OpenEarthMap\Size_256\val\labels"
    all_imgs = glob.glob(os.path.join(img_dir, '*.tif'))
    src_cls,dst_cls =None, None

    config = OmegaConf.load(args.config)
    device = dist.get_rank() % torch.cuda.device_count()



    from models.GDiT import Precond_models

    net = Precond_models[config.model.precond](
        **config.model.params
    ).to(device)

    assert args.ckpt_path is not None
    ckpt = torch.load(args.ckpt_path, map_location="cpu")


    state_name = 'student' if 'student' in ckpt else 'model'

    net.load_state_dict(ckpt[state_name], strict=False)
    net.eval()





    # load vae
    vae = StabilityVAEEncoder(vae_name=r"./pretrained")
    # load clip
    from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder
    clip_model = FrozenCLIPTextEmbedder().cuda().eval()



    select_imgs = [img for img in all_imgs if from_regoin in img]

    for img in tqdm.tqdm(select_imgs):
        args.ref_path = img
        #generate_sample(args, from_regoin, -1, src_cls, dst_cls, clip_model=clip_model, device=device, vae = vae, net = net)
        generate_sample(args, dst_regions, -1, src_cls, dst_cls,  clip_model=clip_model, device=device, vae = vae, net = net)


def generate_sample(args, regoins = 'chicago', obj_id = -1, src = 'Background', dst ='Building', clip_model = None, device = None, vae = None, net = None):

    if clip_model is None:
        from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder
        clip_model = FrozenCLIPTextEmbedder().cuda().eval()

    if net is None:
        config = OmegaConf.load(args.config)
        device = dist.get_rank() % torch.cuda.device_count()


        from models.GDiT import Precond_models
        net = Precond_models[config.model.precond](
            **config.model.params
        ).to(device)

        assert args.ckpt_path is not None
        ckpt = torch.load(args.ckpt_path, map_location="cpu")

        state_name = 'student' if 'student' in ckpt else 'teacher'

        net.load_state_dict(ckpt[state_name], strict=False)
        net.eval()


    sampler_kwargs = dict(num_steps=args.num_steps, S_churn=args.S_churn, flag=args.eval_type)
    y = []

    regions = [regoins] * args.batch_size

    with torch.no_grad():
        for text in regions:
            y.append(clip_model([text]))
        y = torch.cat(y).to(device)

    latents = torch.randn([args.batch_size, 4, 32, 32]).cuda()




    node_dict = get_node_feats(clip_model, oem)

    seg_map = cv2.imread(args.ref_path, cv2.IMREAD_GRAYSCALE)
    instance_map, node_captions = semantic_to_instance_map(seg_map, oem)

    # Display the instance map
    #cv2.imwrite(args.outdir + "/instance.png",instance_map)


    #show instance_map and id on instancemap


    if obj_id >= 0:
        seg_map, node_captions = change_object(seg_map, instance_map, node_captions, obj_id=obj_id, ori=src, dst=dst,class_map=revert_oem)
        instance_map, node_captions = semantic_to_instance_map(seg_map, oem)
    else:
        src = dst = None

    edges = get_adjacency_matrix_from_instace_map(instance_map)

    node_l1 = []
    for caption in node_captions:
        node_l1.append(node_dict[caption])  # 获取每个节点的特征

    # 确保将所有节点特征拼接成一个张量
    node_l1 = torch.cat(node_l1, dim=0)  # 如果 node_dict[caption] 是一个张量，这里应该能正确拼接

    # 你的边数据
    l1_edge = torch.from_numpy(edges)  # 应该是一个形状为 [2, num_edges] 的张量

    node_boxxes = extract_bboxes(torch.from_numpy(seg_map).float().unsqueeze(0).unsqueeze(0),
                                 torch.from_numpy(instance_map).unsqueeze(0))

    graph = Data(
        x=node_l1,  # 节点特征
        node_captions=node_captions,
        node_boxxes=node_boxxes,
        edge_index=l1_edge,  # 边的索引
        batch=torch.full((node_l1.shape[0],), 0, dtype=torch.long)
    ).cuda()
    # 创建图数据对象
    # graph = Data(
    #     x=node_l1,  # 节点特征
    #     edge_index=l1_edge,  # 边的索引
    #     batch=torch.full((node_l1.shape[0],), 0, dtype=torch.long)
    # ).cuda()



    x_cond = torch.from_numpy(seg_map).unsqueeze(0).unsqueeze(0).cuda().float()
    x_intance = torch.from_numpy(instance_map).unsqueeze(0).cuda()
    if args.cfg_scale is not None:
        x_cond = torch.cat([x_cond, torch.zeros_like(x_cond)], dim=0)
        y = torch.cat([y, torch.zeros_like(y)], dim=0)
        latents = torch.cat((latents, latents), dim=0)

    def recur_decode(z):
        try:
            return vae.decode(z)
        except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
            assert z.shape[2] % 2 == 0
            z1, z2 = z.tensor_split(2)
            return torch.cat([recur_decode(z1), recur_decode(z2)])

    with torch.no_grad():

        z = edm_sampler(net, latents.float(), seg=x_cond, class_labels=y, text_context=None,
                       randn_like=torch.randn_like, instance_map=x_intance,
                       cfg_scale=args.cfg_scale, graph=graph, vae=vae, **sampler_kwargs).float()

        samples = recur_decode(z)

        if args.cfg_scale is not None:
            samples, _ = torch.split(samples, samples.shape[0] // 2, dim=0)



    os.makedirs(args.outdir, exist_ok=True)
    for i in range(samples.shape[0]):
        basename = os.path.basename(args.ref_path).replace(".tif", ".png")
        prefix = f'{regions[0]}_obj_{obj_id}_{args.eval_type}_from_{src}_{dst}'

        basename = prefix + '_' + basename


        save_name = os.path.join(args.outdir, basename)
        save_image(samples[i].float(), save_name, nrow=1, normalize=True, value_range=(0, 255))






def change_object(semantic, instance_map, node_captions, obj_id = -1, ori = "Tree" , dst = "Water", class_map = oem):
    # Step 2: Check if `obj_id == -1`, meaning we want to replace all `ori` with `dst`
    if obj_id == 0:
        if ori not in class_map or dst not in class_map:
            raise ValueError(f"Class name '{ori}' or '{dst}' not found in class_map.")

        ori_id = class_map[ori]
        dst_id = class_map[dst]

        mask = semantic == ori_id
        # Step 3: Replace all occurrences of `ori_id` (Tree) in semantic map with `dst_id` (Water)
        semantic[mask] = dst_id



        select_indices = np.unique(instance_map[mask])
        for indice in select_indices:
            node_captions[indice - 1] = dst


        # Step 4: Replace `ori_id` (Tree) in instance map with `dst_id` (Water) if needed


    else:  # If obj_id is provided, replace `obj_id` in both semantic and instance_map
        # Replace `obj_id` in the semantic map with `dst_id`

        mask = instance_map == obj_id
        semantic[mask] = class_map[dst]
        # Replace `obj_id` in the instance map with `dst_id` (if needed)
        select_indices = np.unique(instance_map[mask])
        for indice in  select_indices:
            node_captions[indice - 1] = dst



    return semantic, node_captions

def run_edit_generation(args):
    # -------- defaults (can be overridden by args) --------
    regions = getattr(args, "regions", "LEVIR_CD_A")          # target region/dataset
    obj_id = getattr(args, "obj_id", 0)                       # 0: single edit, else: sweep
    src_cls = getattr(args, "src_cls", "Building")
    dst_cls = getattr(args, "dst_cls", "Agriculture land")

    cls_list = [
        "Background", "Bareland", "Rangeland", "Developed space", "Road",
        "Tree", "Water", "Agriculture land", "Building"
    ]

    # -------- optional: domain adaptation pre-step --------
    # (do NOT sys.exit here; just run if enabled)
    if getattr(args, "run_domain_adapt", False):
        from_region = getattr(args, "from_region", "aachen")
        dst_region = getattr(args, "dst_region", regions)
        regions_Domain_Adaptation(args, from_region=from_region, dst_regions=dst_region)

    # -------- edit generation --------
    # Case 1: single edit (obj_id==0): one pair (src->dst)
    if obj_id == 0:
        generate_sample(args, regions, obj_id, src_cls, dst_cls)
        return

    # Case 2: sweep edits (obj_id!=0): src fixed, dst goes through all classes
    for cls in cls_list:
        if cls == src_cls:
            continue
        generate_sample(args, regions, obj_id, src_cls, cls)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('sampling parameters')

    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--regions', type=str, required=False, default='chicago', help='region of image')
    parser.add_argument("--version", type=str, choices=["v3", "v_prompt"], default = "v_prompt",required=False)
    parser.add_argument("--eval_type", type=str, choices=["regions", "regions_graph", "regions_sem","regions_graph_sem"], default="regoins_graph_sem", required=False)
    parser.add_argument("--ref_path", type=str, default=True, help="path of semantic file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--obj-id", type=int, default=0)

    parser.add_argument('--gpu', default=[0, 1, 2], nargs='+', type=int, dest='gpu', help='The gpu list used.')
    parser.add_argument("--cfg_scale", type=float, default=None, help='None = no guidance, by default = 4.0')
    parser.add_argument('--num_steps', type=int, default=18, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--ckpt-path', type=str, required=True, help='Network pickle filename')
    parser.add_argument('--outdir', type=str, required=True, help='sampling results save filename')


    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'],
                        help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='address for master')
    # model
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=4)

    parser.add_argument('--precond', type=str, choices=['vp', 've', 'edm'], default='edm', help='precond train & loss')


    parser.add_argument("--use_clip", action='store_true', default=False)

    args = parser.parse_args()


    __init_process(args)

    # 只做一次编辑：obj_id==0，Building -> Agriculture land
    args = SimpleNamespace(
        regions="LEVIR_CD_A",
        obj_id=args.obj_id,
        src_cls="Building",
        dst_cls="Agriculture land",
        run_domain_adapt=False,
    )
    run_edit_generation(args)

    # #kmeans
    # regions = 'adelaide'
    # obj_id = 0
    # src_cls = 'Building'
    # dst_cls = 'Agriculture land'
    #
    # cls_list = ["Background","Bareland","Rangeland", "Developed space", "Road","Tree",  "Water", "Agriculture land", "Building"]
    #
    #
    #
    # #
    # regions_Domain_Adaptation(args, from_regoin="aachen", dst_regions="el_rodeo")
    # sys.exit(-1)
    #
    # regions = "LEVIR_CD_A"
    #
    # if obj_id == 0:
    #
    #     generate_sample(args, regions, obj_id, src_cls, dst_cls)
    #
    #
    #
    # else:
    #     for cls in cls_list:
    #         src_cls = src_cls
    #         dst_cls = cls
    #         if src_cls == dst_cls:
    #             continue
    #         generate_sample(args, regions, obj_id, src_cls, dst_cls)


    #敏感目标伪装？



